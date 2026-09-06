import json

import pytest
import torch
from torch import nn

from scripts.finetune_qwen import checkpoint
from scripts.finetune_qwen.torch_model import install_adapters
from scripts.finetune_qwen.train import epoch_order, initial_state, validation_rows


def test_resume_identity_covers_data_and_checkpoint_code(monkeypatch):
    from scripts.finetune_qwen import train

    original = train.runtime_identity()
    assert set(original["code"]) == {"train.py", "torch_model.py", "data.py", "checkpoint.py"}
    real_hash = train.sha256
    monkeypatch.setattr(train, "sha256", lambda path: "changed-loader" if path.name == "data.py" else real_hash(path))
    assert train.runtime_identity() != original


def test_interrupted_pointer_commit_keeps_previous_optimizer_checkpoint(tmp_path, monkeypatch):
    original = {"step": 4, "momentum": torch.arange(8, dtype=torch.float32)}
    checkpoint.publish_checkpoint(tmp_path, original)

    def interrupt(_path, _value):
        raise OSError("interrupted before pointer commit")

    monkeypatch.setattr(checkpoint, "durable_json", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        checkpoint.publish_checkpoint(tmp_path, {"step": 5, "momentum": torch.ones(8)})
    saved = checkpoint.read_checkpoint(tmp_path)
    assert saved["step"] == 4
    torch.testing.assert_close(saved["momentum"], original["momentum"])


def test_checkpoint_corruption_is_rejected_before_deserialization(tmp_path, monkeypatch):
    path = checkpoint.publish_checkpoint(tmp_path, {"step": 1})
    path.write_bytes(b"truncated checkpoint")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("Corrupt bytes must not reach the deserializer")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="hash mismatch"):
        checkpoint.read_checkpoint(tmp_path)


def test_best_adapter_checkpoint_does_not_replace_resume(tmp_path):
    checkpoint.publish_checkpoint(tmp_path, {"step": 7})
    checkpoint.publish_checkpoint(tmp_path, {"step": 3}, kind="best")
    checkpoint.publish_checkpoint(tmp_path, {"step": 8})
    assert checkpoint.read_checkpoint(tmp_path)["step"] == 8
    assert checkpoint.read_checkpoint(tmp_path, kind="best")["step"] == 3
    pointer = json.loads((tmp_path / "resume.json").read_text())
    assert pointer["slot"] == 1


def test_optimizer_resume_matches_an_uninterrupted_update(tmp_path):
    def make_model():
        torch.manual_seed(42)
        model = nn.Module()
        model.layer = nn.Module()
        model.layer.q_proj = nn.Linear(8, 4)
        install_adapters(model, rank=2)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        return model, optimizer

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        model.layer.q_proj(torch.ones(2, 8)).square().mean().backward()
        optimizer.step()

    model, optimizer = make_model()
    update(model, optimizer)
    checkpoint.publish_checkpoint(
        tmp_path,
        {"step": 1, "adapters": checkpoint.adapter_state(model), "optimizer": optimizer.state_dict()},
    )
    update(model, optimizer)
    restored, resumed_optimizer = make_model()
    saved = checkpoint.read_checkpoint(tmp_path)
    checkpoint.restore_adapters(restored, saved["adapters"])
    resumed_optimizer.load_state_dict(saved["optimizer"])
    update(restored, resumed_optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_incompatible_adapter_does_not_partially_mutate_model():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.q_proj = nn.Linear(4, 4)
    install_adapters(model, rank=2)
    original = checkpoint.adapter_state(model)
    invalid = {name: torch.ones_like(value) for name, value in original.items()}
    invalid["layer.q_proj.adapter_b"] = torch.ones(1)
    with pytest.raises(ValueError, match="Invalid checkpoint adapter"):
        checkpoint.restore_adapters(model, invalid)
    for name, value in checkpoint.adapter_state(model).items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_resume_order_and_validation_selection_cover_both_chapters():
    order = epoch_order(101, 7, 2)
    assert order[:48] + epoch_order(101, 7, 2)[48:] == order
    assert sorted(order) == list(range(101))
    assert order != epoch_order(101, 7, 3)
    rows = [{"chapter": chapter, "id": f"{chapter}_{i}"} for chapter in (8, 25) for i in range(60)]
    selected = validation_rows(rows, 32, 7)
    assert sum(row["chapter"] == 8 for row in selected) == 16
    assert sum(row["chapter"] == 25 for row in selected) == 16
    assert selected == validation_rows(rows, 32, 7)


def test_validation_balances_sources_with_identical_chapter_numbers():
    rows = [{"source": source, "chapter": 8, "id": f"{source}{i}"} for source in ("", "sisifo") for i in range(40)]
    selected = validation_rows(rows, 32, 7)
    assert sum(row["source"] == "sisifo" for row in selected) == 16
    assert selected == validation_rows(rows, 32, 7)


@pytest.fixture
def completed_parent(tmp_path):
    contract = {
        "base": {"weights": "base-hash"},
        "rank": 2,
        "reference_id": "ch04_00077",
        "reference_sha256": "reference-hash",
    }
    saved = {
        "contract": contract,
        "step": 7,
        "best_step": 7,
        "best_loss": 2.0,
        "baseline": {"loss": 3.0},
        "adapters": {"adapter": torch.ones(2)},
        "speaker": torch.ones(4),
    }
    (tmp_path / ".training.lock").touch()
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    (tmp_path / "status.json").write_text(json.dumps({"event": "complete"}))
    checkpoint.publish_checkpoint(tmp_path, saved, kind="best")
    reference = {"id": "ch04_00077", "sha256": "reference-hash", "split": "train"}
    return tmp_path, contract, reference


def test_continuation_loads_selected_adapters_and_preserves_exact_reference(completed_parent):
    parent, contract, reference = completed_parent
    saved, identity, selected = initial_state(parent, contract["base"], 2, [reference])
    assert saved["step"] == 7
    assert "optimizer" not in saved
    assert selected == reference
    assert identity["best"]["step"] == 7
    assert identity["contract_sha256"] == checkpoint.sha256(parent / "contract.json")


def test_continuation_rejects_base_rank_reference_and_incomplete_parent(completed_parent):
    parent, contract, reference = completed_parent
    for base, rank in (({}, 2), (contract["base"], 3)):
        with pytest.raises(ValueError, match="mismatch"):
            initial_state(parent, base, rank, [reference])
    for rows in ([], [{**reference, "split": "test"}], [{**reference, "sha256": "changed"}]):
        with pytest.raises(ValueError, match="exact parent speaker reference"):
            initial_state(parent, contract["base"], 2, rows)
    (parent / "status.json").write_text(json.dumps({"event": "paused"}))
    with pytest.raises(ValueError, match="completed parent"):
        initial_state(parent, contract["base"], 2, [reference])
