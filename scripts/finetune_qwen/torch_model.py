"""Load the unquantized local model without duplicating gigabytes of weights."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional


def load_local_mlx_base(path: Path, device: str):
    from accelerate import init_empty_weights
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    raw_config = json.loads((path / "config.json").read_text())
    if raw_config.get("quantization") or raw_config.get("tts_model_type") != "base":
        raise ValueError("Fine-tuning requires an unquantized Qwen Base checkpoint")
    config = Qwen3TTSConfig.from_dict(raw_config)
    config._attn_implementation = "sdpa"
    config.talker_config._attn_implementation = "sdpa"
    config.talker_config.code_predictor_config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3TTSForConditionalGeneration(config)
    weights = load_file(path / "model.safetensors")
    # MLX uses [output, kernel, input] for speaker Conv1d weights; PyTorch
    # uses [output, input, kernel]. Transformer tensors are already identical.
    for name, value in weights.items():
        if name.startswith("speaker_encoder.") and value.ndim == 3:
            weights[name] = value.transpose(1, 2).contiguous()
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.to(device)
    model.config.use_cache = False
    model.talker.config.use_cache = False
    model.talker.code_predictor.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer


class AdapterLinear(nn.Module):
    """FP32 trainable updates around unchanged BF16 base weights.

    The update can be merged into the original matrix for inference; it adds no
    inference layers, context tokens or reference audio after export.
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.scale = 2.0
        self.adapter_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.adapter_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.adapter_a, a=math.sqrt(5))

    def forward(self, inputs):
        update = functional.linear(functional.linear(inputs.float(), self.adapter_a), self.adapter_b)
        return self.base(inputs) + (update * self.scale).to(inputs.dtype)

    def merged_weight(self):
        return self.base.weight.float() + self.scale * (self.adapter_b @ self.adapter_a)


def install_adapters(model: nn.Module, rank: int) -> list[str]:
    model.requires_grad_(False)
    targets = []
    names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in names:
            parent_name, key = name.rsplit(".", 1)
            setattr(model.get_submodule(parent_name), key, AdapterLinear(module, rank))
            targets.append(name)
    if not targets:
        raise ValueError("No compatible transformer layers found")
    return targets


def next_codec_loss(logits: torch.Tensor, target_codes: torch.Tensor) -> torch.Tensor:
    """One explicit causal shift. Do not also use Hugging Face's shifted loss."""
    return functional.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), target_codes[:, 1:].reshape(-1), ignore_index=-100
    )


def fixed_speaker_tensor(model, speaker: torch.Tensor) -> torch.Tensor:
    """Match the frozen audio embedding, not the first (FP32 adapter) parameter."""
    weight = model.talker.get_input_embeddings().weight
    if speaker.numel() != weight.shape[1] or not torch.isfinite(speaker).all():
        raise ValueError("Invalid fixed speaker tensor")
    return speaker.detach().to(device=weight.device, dtype=weight.dtype)


def reference_embedding(model, path: Path) -> torch.Tensor:
    import soundfile as sf
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    audio, rate = sf.read(path, dtype="float32")
    if rate != 24000 or audio.ndim != 1:
        raise ValueError("Speaker reference must be mono 24 kHz")
    mels = mel_spectrogram(
        torch.from_numpy(audio)[None],
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)
    with torch.no_grad():
        model.speaker_encoder.eval()
        encoder_weight = next(model.speaker_encoder.parameters())
        speaker = model.speaker_encoder(mels.to(device=encoder_weight.device, dtype=encoder_weight.dtype))
        return fixed_speaker_tensor(model, speaker)


def training_inputs(model, tokenizer, text: str, codes: torch.Tensor, speaker: torch.Tensor):
    """Match Spanish CustomVoice's non-streaming inference prefix exactly."""
    config = model.config
    talker = model.talker
    device = model.device
    codec = config.talker_config
    if speaker.dtype != talker.get_input_embeddings().weight.dtype:
        raise ValueError("Fixed speaker dtype must match the frozen audio embeddings")

    def text_embeddings(ids):
        return talker.text_projection(talker.get_text_embeddings()(torch.tensor([ids], device=device)))

    def codec_embeddings(ids):
        return talker.get_input_embeddings()(torch.tensor([ids], device=device))

    role = text_embeddings(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
    tts_bos, tts_eos, tts_pad = text_embeddings(
        [config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]
    ).chunk(3, dim=1)
    tags = codec_embeddings(
        [
            codec.codec_think_id,
            codec.codec_think_bos_id,
            codec.codec_language_id["spanish"],
            codec.codec_think_eos_id,
        ]
    )
    speaker_tags = torch.cat([tags, speaker.view(1, 1, -1), codec_embeddings([codec.codec_pad_id])], dim=1)
    prefix_tags = torch.cat([tts_pad.expand(-1, 5, -1), tts_bos], dim=1) + speaker_tags
    text_part = torch.cat([text_embeddings(tokenizer.encode(text, add_special_tokens=False)), tts_eos], dim=1)
    text_part = text_part + codec_embeddings([codec.codec_pad_id]).expand(-1, text_part.shape[1], -1)
    prefix = torch.cat([role, prefix_tags, text_part, tts_pad + codec_embeddings([codec.codec_bos_id])], dim=1)
    audio = talker.get_input_embeddings()(codes[:, 0])[None]
    for group in range(1, 16):
        audio = audio + talker.code_predictor.get_input_embeddings()[group - 1](codes[:, group])[None]
    audio = audio + tts_pad
    inputs = torch.cat([prefix, audio, tts_pad + codec_embeddings([codec.codec_eos_token_id])], dim=1)
    targets = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
    targets[0, prefix.shape[1] : -1] = codes[:, 0]
    targets[0, -1] = codec.codec_eos_token_id
    return inputs, targets, prefix.shape[1]


def training_loss(model, tokenizer, rows: list[dict], speaker: torch.Tensor):
    examples = []
    code_rows = []
    for row in rows:
        codes = torch.tensor(row["audio_codes"], dtype=torch.long, device=model.device)
        examples.append(training_inputs(model, tokenizer, row["text"], codes, speaker))
        code_rows.append(codes)
    inputs = nn.utils.rnn.pad_sequence([example[0][0] for example in examples], batch_first=True)
    targets = nn.utils.rnn.pad_sequence([example[1][0] for example in examples], batch_first=True, padding_value=-100)
    lengths = torch.tensor([example[0].shape[1] for example in examples], device=model.device)
    mask = (torch.arange(inputs.shape[1], device=model.device)[None] < lengths[:, None]).long()
    outputs = model.talker(
        inputs_embeds=inputs,
        attention_mask=mask,
        output_hidden_states=True,
        use_cache=False,
    )
    main_loss = next_codec_loss(outputs.logits, targets)
    # The hidden state BEFORE a codec frame predicts that frame. Selecting the
    # state after ingesting its target residual codes would leak the answer.
    hidden = torch.cat(
        [
            outputs.hidden_states[0][-1][index, example[2] - 1 : example[2] - 1 + len(codes)]
            for index, (example, codes) in enumerate(zip(examples, code_rows, strict=True))
        ]
    )
    codes = torch.cat(code_rows)
    predictor_inputs = [hidden[:, None], model.talker.get_input_embeddings()(codes[:, :1])]
    for group in range(1, 15):
        predictor_inputs.append(
            model.talker.code_predictor.get_input_embeddings()[group - 1](codes[:, group : group + 1])
        )
    predictor = model.talker.code_predictor.forward_finetune(
        inputs_embeds=torch.cat(predictor_inputs, dim=1),
        use_cache=False,
    )
    # forward_finetune already selects each residual head at its causal input
    # position. Its labels must not pass through ForCausalLMLoss's extra shift.
    residual_loss = functional.cross_entropy(predictor.logits.float().reshape(-1, 2048), codes[:, 1:].reshape(-1))
    return main_loss + 0.3 * residual_loss, main_loss.detach(), residual_loss.detach()
