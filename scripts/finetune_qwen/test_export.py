import json

import pytest
import torch

from scripts.finetune_qwen.checkpoint import publish_checkpoint, read_checkpoint
from scripts.finetune_qwen.export import merge_weights, retained_checkpoint


def test_export_merges_adapters_and_embeds_only_the_chosen_speaker():
    torch.manual_seed(9)
    original = torch.randn(6, 8, dtype=torch.bfloat16)
    a, b = torch.randn(2, 8), torch.randn(6, 2)
    speaker = torch.randn(1, 6)
    embedding = torch.randn(3072, 6, dtype=torch.bfloat16)
    merged = merge_weights(
        {
            "talker.layer.q_proj.weight": original,
            "talker.model.codec_embedding.weight": embedding,
            "speaker_encoder.conv.weight": torch.zeros(1),
        },
        {"talker.layer.q_proj.adapter_a": a, "talker.layer.q_proj.adapter_b": b},
        speaker,
    )
    expected = (original.float() + 2 * (b @ a)).to(torch.bfloat16)
    torch.testing.assert_close(merged["talker.layer.q_proj.weight"], expected, rtol=0, atol=0)
    torch.testing.assert_close(merged["talker.model.codec_embedding.weight"][3000], speaker[0].to(torch.bfloat16))
    torch.testing.assert_close(merged["talker.model.codec_embedding.weight"][:3000], embedding[:3000], rtol=0, atol=0)
    assert "speaker_encoder.conv.weight" not in merged
    assert not any("adapter" in name for name in merged)


def test_export_refuses_incomplete_adapter_pairs():
    with pytest.raises(ValueError, match="Incomplete adapter"):
        merge_weights({}, {"talker.q_proj.adapter_a": torch.ones(1, 2)}, torch.ones(1))


def test_retained_validation_snapshot_does_not_change_best_pointer(tmp_path):
    contract = {"base": "hash"}
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    (tmp_path / "status.json").write_text(json.dumps({"event": "complete"}))
    for step in (300, 400):
        publish_checkpoint(tmp_path, {"contract": contract, "step": step, "best_step": step}, kind="best")
    pointer = (tmp_path / "best.json").read_bytes()
    saved, identity = retained_checkpoint(tmp_path, 300)
    assert saved["step"] == 300
    assert len(identity["sha256"]) == 64
    assert (tmp_path / "best.json").read_bytes() == pointer
    assert read_checkpoint(tmp_path, kind="best")["step"] == 400
    with pytest.raises(ValueError, match="not uniquely retained"):
        retained_checkpoint(tmp_path, 200)
    (tmp_path / "status.json").write_text(json.dumps({"event": "paused"}))
    with pytest.raises(ValueError, match="completed run"):
        retained_checkpoint(tmp_path, 300)


def test_retained_snapshot_rejects_another_run_contract(tmp_path):
    (tmp_path / "contract.json").write_text(json.dumps({"base": "current"}))
    (tmp_path / "status.json").write_text(json.dumps({"event": "complete"}))
    publish_checkpoint(tmp_path, {"contract": {"base": "other"}, "step": 300, "best_step": 300}, kind="best")
    with pytest.raises(ValueError, match="contract or selection mismatch"):
        retained_checkpoint(tmp_path, 300)


@pytest.mark.parametrize(("loss", "baseline"), [(float("nan"), 3.0), (2.0, float("nan")), (2.0, float("inf"))])
def test_export_rejects_nonfinite_selection_metrics(tmp_path, monkeypatch, loss, baseline):
    from scripts.finetune_qwen import export as exporter

    saved = {"step": 1, "contract": {"base": {}}, "best_loss": loss, "baseline": {"loss": baseline}}
    monkeypatch.setattr(exporter, "read_checkpoint", lambda *_args, **_kwargs: saved)
    monkeypatch.setattr(exporter, "base_identity", lambda _path: {})
    with pytest.raises(ValueError, match="did not improve"):
        exporter.export(tmp_path, tmp_path, tmp_path / "output")
