import pytest
import torch
from torch import nn

from scripts.finetune_qwen.torch_model import (
    AdapterLinear,
    install_adapters,
    next_codec_loss,
    training_inputs,
    training_loss,
)


def test_adapter_starts_identical_and_merges_without_an_extra_layer():
    torch.manual_seed(4)
    base = nn.Linear(8, 6)
    adapter = AdapterLinear(base, rank=4)
    inputs = torch.randn(3, 8)
    assert torch.equal(adapter(inputs), base(inputs))
    with torch.no_grad():
        adapter.adapter_b.normal_(std=0.03)
    merged = nn.Linear(8, 6)
    with torch.no_grad():
        merged.weight.copy_(adapter.merged_weight())
        merged.bias.copy_(base.bias)
    torch.testing.assert_close(adapter(inputs), merged(inputs))


def test_only_adapter_parameters_train():
    model = nn.Module()
    model.q_proj = nn.Linear(8, 8)
    model.v_proj = nn.Linear(8, 8)
    # Use the same nested shape as the actual transformer.
    root = nn.Module()
    root.layer = model
    assert install_adapters(root, 4) == ["layer.q_proj", "layer.v_proj"]
    parameters = [name for name, parameter in root.named_parameters() if parameter.requires_grad]
    assert parameters == [
        "layer.q_proj.adapter_a",
        "layer.q_proj.adapter_b",
        "layer.v_proj.adapter_a",
        "layer.v_proj.adapter_b",
    ]


def test_main_loss_predicts_exactly_one_token_ahead_including_eos():
    labels = torch.tensor([[-100, -100, 2, 3, 4]])
    logits = torch.full((1, 5, 6), -15.0)
    logits[0, 1, 2] = 15
    logits[0, 2, 3] = 15
    logits[0, 3, 4] = 15
    assert next_codec_loss(logits, labels).item() < 1e-5
    wrong = logits.roll(-1, dims=1)
    assert next_codec_loss(wrong, labels).item() > 10


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [1, 2, 3] if text == "<|im_start|>assistant\n" else [20, 21, 22, 23]


def _tiny_model():
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

    transformer = {
        # Qwen's codec control tokens live above the ordinary text vocabulary.
        "vocab_size": 4306,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 64,
    }
    config = Qwen3TTSConfig(
        tts_model_type="custom_voice",
        tts_bos_token_id=90,
        tts_eos_token_id=91,
        tts_pad_token_id=92,
        talker_config={
            **transformer,
            "text_hidden_size": 64,
            "text_vocab_size": 128,
            "spk_id": {"fabian": 3000},
            "spk_is_dialect": {"fabian": False},
            "codec_language_id": {"spanish": 2054},
            "rope_scaling": {"rope_type": "default", "mrope_section": [12, 10, 10], "interleaved": True},
            "code_predictor_config": {**transformer, "vocab_size": 2048, "num_code_groups": 16},
        },
    )
    return Qwen3TTSForConditionalGeneration(config).eval()


def test_training_prefix_matches_actual_qwen_custom_voice_inference(monkeypatch):
    model = _tiny_model()
    tokenizer = _Tokenizer()
    codes = torch.randint(0, 2048, (3, 16))
    speaker = model.talker.get_input_embeddings()(torch.tensor([3000]))
    inputs, targets, prefix = training_inputs(model, tokenizer, "Texto de prueba", codes, speaker)
    captured = []

    class CapturedError(Exception):
        pass

    def capture(**kwargs):
        captured.append(kwargs["inputs_embeds"])
        raise CapturedError

    monkeypatch.setattr(model.talker, "generate", capture)
    ids = torch.tensor([[1, 2, 3, 20, 21, 22, 23, 4, 3, 1, 2, 3]])
    with pytest.raises(CapturedError):
        model.generate(input_ids=[ids], languages=["Spanish"], speakers=["fabian"], non_streaming_mode=True)
    torch.testing.assert_close(inputs[:, :prefix], captured[0])
    assert targets[0, :prefix].eq(-100).all()
    assert torch.equal(targets[0, prefix:-1], codes[:, 0])
    assert targets[0, -1].item() == model.config.talker_config.codec_eos_token_id


def test_batched_training_has_finite_gradients_with_different_audio_lengths():
    model = _tiny_model()
    speaker = model.talker.get_input_embeddings()(torch.tensor([3000])).detach()
    install_adapters(model, rank=4)
    rows = [{"text": "Prueba", "audio_codes": torch.randint(0, 2048, (length, 16)).tolist()} for length in (3, 5)]
    loss, main, residual = training_loss(model, _Tokenizer(), rows, speaker)
    assert torch.isfinite(loss)
    assert main > 0
    assert residual > 0
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert any(parameter.grad.abs().max() > 0 for parameter in trainable)


def test_resume_preserves_bf16_speaker_despite_fp32_adapter_model_dtype():
    from scripts.finetune_qwen.torch_model import fixed_speaker_tensor

    model = _tiny_model().to(torch.bfloat16)
    speaker = model.talker.get_input_embeddings()(torch.tensor([3000])).detach()
    install_adapters(model, rank=4)
    assert model.dtype == torch.float32
    restored = fixed_speaker_tensor(model, speaker.cpu())
    assert restored.dtype == torch.bfloat16
    assert torch.equal(restored, speaker)
    codes = torch.randint(0, 2048, (3, 16))
    inputs, _targets, _prefix = training_inputs(model, _Tokenizer(), "Texto de prueba", codes, restored)
    assert inputs.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="Fixed speaker dtype"):
        training_inputs(model, _Tokenizer(), "Texto de prueba", codes, restored.float())
