import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling


def apply_lora(
    model,
    rank=8,
    alpha=16,
    target_modules=("q_proj", "v_proj"),
    freeze_base=True,
):
    device = next(model.parameters()).device

    if freeze_base:
        for p in model.parameters():
            p.requires_grad = False

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not any(target in name for target in target_modules):
            continue

        lora = LoRA(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
        ).to(device)

        setattr(module, "lora", lora)

        original_forward = module.forward

        def forward_with_lora(x, layer1=original_forward, layer2=lora):
            return layer1(x) + layer2(x)

        module.forward = forward_with_lora

        for p in module.lora.parameters():
            p.requires_grad = True

    return model


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}

    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            for k, v in module.lora.state_dict().items():
                state_dict[f"{clean_name}.lora.{k}"] = v.detach().cpu().half()

    torch.save(state_dict, path)


def load_lora(model, path, strict=True):
    device = next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)

    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    raw_model = getattr(model, "_orig_mod", model)

    for name, module in raw_model.named_modules():
        if not hasattr(module, "lora"):
            continue

        prefix = f"{name}.lora."
        lora_state = {
            k.replace(prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

        if len(lora_state) == 0:
            if strict:
                raise ValueError(f"No LoRA weights found for module: {name}")
            continue

        module.lora.load_state_dict(lora_state, strict=True)

    return model


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)

    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}

    for name, param in raw_model.state_dict().items():
        if ".lora." not in name:
            state_dict[name] = param.detach().cpu().half()

    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "lora"):
            delta_weight = module.lora.B.weight.data @ module.lora.A.weight.data
            delta_weight = delta_weight * module.lora.scaling

            merged_weight = module.weight.data + delta_weight.to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            state_dict[f"{name}.weight"] = merged_weight.detach().cpu().half()

            if module.bias is not None:
                state_dict[f"{name}.bias"] = module.bias.data.detach().cpu().half()

    torch.save(state_dict, save_path)