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


if __name__ == "__main__":
    main()