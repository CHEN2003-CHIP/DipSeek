import argparse
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eval_llm import infer_arch_from_state_dict, load_checkpoint_state_dict
from model.model_dipseek import DipSeekConfig, DipSeekForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Smoke test DipSeek checkpoint architecture loading.")
    parser.add_argument('--checkpoint_path', required=True, type=str, help="Full checkpoint path.")
    parser.add_argument('--hidden_size', default=768, type=int, help="Model hidden size.")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Number of hidden layers.")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="Runtime device.")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="Whether to enable MoE.")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="Enable inference RoPE scaling.")
    parser.add_argument('--strict_load', default=1, type=int, choices=[0, 1], help="Whether to strict-load the checkpoint.")
    args = parser.parse_args()

    state_dict = load_checkpoint_state_dict(args.checkpoint_path, args.device)
    use_residual_scale, mtp_depth = infer_arch_from_state_dict(state_dict)

    config = DipSeekConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling,
        use_residual_scale=use_residual_scale,
        mtp_depth=mtp_depth,
        mtp_loss_weight=0.0,
    )
    model = DipSeekForCausalLM(config).to(args.device).eval()
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(args.strict_load))

    print(f"checkpoint_path={args.checkpoint_path}")
    print(f"use_residual_scale={int(use_residual_scale)}")
    print(f"mtp_depth={mtp_depth}")
    print(f"missing_keys_count={len(missing)}")
    print(f"unexpected_keys_count={len(unexpected)}")
    if missing:
        print(f"missing_examples={missing[:20]}")
    if unexpected:
        print(f"unexpected_examples={unexpected[:20]}")

    input_ids = torch.randint(0, config.vocab_size, (1, 8), device=args.device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
    print(f"forward_success={outputs.logits.shape == (1, 8, config.vocab_size)}")
    print(f"logits_shape={tuple(outputs.logits.shape)}")


if __name__ == "__main__":
    main()
