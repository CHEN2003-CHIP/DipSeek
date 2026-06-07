import importlib.util
import inspect
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    print("Project root:", ROOT)
    print("Python:", sys.executable)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)

    old_spec = importlib.util.find_spec("model.model_minimind")
    assert old_spec is None, "model.model_minimind still exists or is still importable. Remove compatibility layer."

    from model.model_dipseek import DipSeekConfig, DipSeekForCausalLM

    assert hasattr(DipSeekConfig, "model_type"), "DipSeekConfig has no model_type"
    assert DipSeekConfig.model_type == "dipseek", f"model_type should be dipseek, got {DipSeekConfig.model_type}"

    # 按真实 __init__ 参数过滤，避免某些参数名和示例不一致导致报错
    proposed_kwargs = {
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 256,
        "max_position_embeddings": 128,
        "intermediate_size": 256,
        "dropout": 0.0,
        "use_moe": False,
        "flash_attn": False,
    }

    sig = inspect.signature(DipSeekConfig.__init__)
    kwargs = {
        k: v for k, v in proposed_kwargs.items()
        if k in sig.parameters
    }

    print("Config kwargs:", kwargs)
    config = DipSeekConfig(**kwargs)

    # 防止某些默认值过大，尽量压到 tiny
    for k, v in proposed_kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    if hasattr(config, "model_type"):
        assert config.model_type == "dipseek"

    model = DipSeekForCausalLM(config)
    model.eval()

    vocab_size = getattr(config, "vocab_size", 256)
    input_ids = torch.randint(0, vocab_size, (2, 16), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids)

    if hasattr(out, "logits"):
        logits = out.logits
    elif isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    print("Forward output type:", type(out))
    if hasattr(logits, "shape"):
        print("Logits shape:", tuple(logits.shape))

    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "_smoke_dipseek_state_dict.pth"

    torch.save(model.state_dict(), ckpt_path)
    model2 = DipSeekForCausalLM(config)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model2.load_state_dict(state, strict=False)

    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    assert len(missing) == 0, f"Missing keys: {missing}"

    print("State dict save/load strict-like check passed.")
    print("Checkpoint:", ckpt_path)
    print("Param count:", sum(p.numel() for p in model.parameters()))
    print("Smoke test passed.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        input_ids = input_ids.to(device)
        with torch.no_grad():
            _ = model(input_ids)
        print("CUDA forward passed.")
        print("Max CUDA memory MB:", round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2))

    mtp_config = DipSeekConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=32,
        intermediate_size=128,
        dropout=0.0,
        use_moe=False,
        flash_attn=False,
        tie_word_embeddings=False,
        mtp_depth=1,
        mtp_loss_weight=0.1,
        mtp_detach_lm_head=True,
        mtp_adapter_init=0.0,
    )
    mtp_model = DipSeekForCausalLM(mtp_config)
    mtp_model.train()
    mtp_input = torch.randint(0, mtp_config.vocab_size, (2, 8), dtype=torch.long)
    mtp_labels = mtp_input.clone()
    mtp_out = mtp_model(mtp_input, labels=mtp_labels)
    assert torch.isfinite(mtp_out.loss), "MTP total loss should be finite"
    assert torch.isfinite(mtp_out.main_loss), "MTP main loss should be finite"
    assert torch.isfinite(mtp_out.mtp_loss), "MTP aux loss should be finite"
    mtp_out.loss.backward()
    assert mtp_model.mtp_heads[0].gate.grad is not None, "MTP gate should receive gradients"

    mtp_model.zero_grad(set_to_none=True)
    mtp_only = mtp_model(mtp_input, labels=mtp_labels).mtp_loss
    mtp_only.backward()
    assert mtp_model.lm_head.weight.grad is None, "Detached MTP projection should not update lm_head.weight"

    mtp_model.eval()
    with torch.no_grad():
        eval_out = mtp_model(mtp_input, labels=mtp_labels)
    assert eval_out.mtp_loss.item() == 0.0, "Eval mode should not compute MTP loss"

    masked_labels = torch.full_like(mtp_input, -100)
    mtp_model.train()
    masked_out = mtp_model(mtp_input, labels=masked_labels)
    assert torch.isfinite(masked_out.loss), "Fully masked total loss should not be NaN"
    assert torch.isfinite(masked_out.mtp_loss), "Fully masked MTP loss should not be NaN"

    short_input = mtp_input[:, :2]
    short_out = mtp_model(short_input, labels=short_input.clone())
    assert torch.isfinite(short_out.mtp_loss), "Short sequence MTP loss should not be NaN"
    export_state = {
        k: v
        for k, v in mtp_model.state_dict().items()
        if not k.startswith("mtp_heads.")
    }
    assert any(k.startswith("mtp_heads.") for k in mtp_model.state_dict()), "MTP model should own train-time heads"
    assert not any(k.startswith("mtp_heads.") for k in export_state), "Export state should exclude MTP heads"
    print("MTP Lite stability checks passed.")


if __name__ == "__main__":
    main()
