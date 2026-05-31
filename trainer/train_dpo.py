import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_dipseek import DipSeekConfig
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def count_params(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def get_batch_logps(logits, labels, average_log_prob=False):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()

    loss_mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~loss_mask] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1),
        dim=2,
        index=safe_labels.unsqueeze(2)
    ).squeeze(2)
    per_token_logps = per_token_logps * loss_mask

    if average_log_prob:
        return per_token_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)

    return per_token_logps.sum(-1)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1,
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    reward_margins = chosen_rewards - rejected_rewards

    metrics = {
        'chosen_reward': chosen_rewards.mean(),
        'rejected_reward': rejected_rewards.mean(),
        'reward_acc': reward_accuracies.mean(),
        'reward_margin': reward_margins.mean(),
        'policy_chosen_logps': policy_chosen_logps.detach().mean(),
        'policy_rejected_logps': policy_rejected_logps.detach().mean(),
        'ref_chosen_logps': reference_chosen_logps.detach().mean(),
        'ref_rejected_logps': reference_rejected_logps.detach().mean(),
    }
    return losses.mean(), metrics


def save_policy_checkpoint(lm_config, model, optimizer, scaler, epoch, step, wandb=None):
    model.eval()
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = {
        k: v.half().cpu()
        for k, v in raw_model.state_dict().items()
        if not k.startswith("mtp_heads.")
    }
    torch.save(state_dict, ckp)
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
    model.train()
    del state_dict


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        chosen_input_ids = batch['chosen_input_ids'].to(args.device)
        chosen_labels = batch['chosen_labels'].to(args.device)
        rejected_input_ids = batch['rejected_input_ids'].to(args.device)
        rejected_labels = batch['rejected_labels'].to(args.device)
        input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            with torch.no_grad():
                ref_outputs = ref_model(input_ids)
                reference_logps = get_batch_logps(ref_outputs.logits, labels)

            outputs = model(input_ids)
            policy_logps = get_batch_logps(outputs.logits, labels)

            batch_size = chosen_input_ids.shape[0]
            policy_chosen_logps, policy_rejected_logps = policy_logps[:batch_size], policy_logps[batch_size:]
            reference_chosen_logps, reference_rejected_logps = reference_logps[:batch_size], reference_logps[batch_size:]

            dpo_loss_val, metrics = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=beta
            )
            aux_loss = outputs.aux_loss if outputs.aux_loss is not None else outputs.logits.new_zeros(())
            loss = (dpo_loss_val + aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            log_values = {
                'loss': current_loss,
                'dpo_loss': current_dpo_loss,
                'aux_loss': current_aux_loss,
                'chosen_reward': metrics['chosen_reward'].item(),
                'rejected_reward': metrics['rejected_reward'].item(),
                'reward_acc': metrics['reward_acc'].item(),
                'reward_margin': metrics['reward_margin'].item(),
                'policy_chosen_logps': metrics['policy_chosen_logps'].item(),
                'policy_rejected_logps': metrics['policy_rejected_logps'].item(),
                'ref_chosen_logps': metrics['ref_chosen_logps'].item(),
                'ref_rejected_logps': metrics['ref_rejected_logps'].item(),
                'learning_rate': current_lr,
                'epoch_time': eta_min,
            }

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, '
                f'chosen_reward: {log_values["chosen_reward"]:.4f}, rejected_reward: {log_values["rejected_reward"]:.4f}, '
                f'reward_acc: {log_values["reward_acc"]:.4f}, reward_margin: {log_values["reward_margin"]:.4f}, '
                f'policy_chosen_logps: {log_values["policy_chosen_logps"]:.4f}, policy_rejected_logps: {log_values["policy_rejected_logps"]:.4f}, '
                f'ref_chosen_logps: {log_values["ref_chosen_logps"]:.4f}, ref_rejected_logps: {log_values["ref_rejected_logps"]:.4f}, '
                f'lr: {current_lr:.8f}, epoch_time: {eta_min:.3f}min'
            )

            if wandb: wandb.log(log_values)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_policy_checkpoint(lm_config, model, optimizer, scaler, epoch, step, wandb)

        del chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, input_ids, labels
        del ref_outputs, reference_logps, outputs, policy_logps, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DipSeek DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_residual_scale', default=0, type=int, choices=[0, 1], help="是否启用Residual Scale（0=否，1=是）")
    parser.add_argument('--residual_scale_init', default=1.0, type=float, help="Residual Scale初始值")
    parser.add_argument('--mtp_depth', default=0, type=int, help="MTP-lite深度，DPO阶段默认关闭")
    parser.add_argument('--mtp_loss_weight', default=0.0, type=float, help="MTP-lite loss权重，DPO阶段默认0")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="DipSeek-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = DipSeekConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        use_residual_scale=bool(args.use_residual_scale),
        residual_scale_init=args.residual_scale_init,
        mtp_depth=args.mtp_depth,
        mtp_loss_weight=args.mtp_loss_weight,
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if args.dtype == "float32":
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"DipSeek-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    moe_suffix = '_moe' if lm_config.use_moe else ''
    weight_path = f'../out/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    Logger(f'Loading policy/ref model from: {weight_path}')
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    policy_params = count_params(model)
    policy_trainable_params = count_params(model, trainable_only=True)
    reference_params = count_params(ref_model)
    reference_trainable_params = count_params(ref_model, trainable_only=True)

    Logger('DPO config:')
    Logger(f'  from_weight={args.from_weight}, save_weight={args.save_weight}, save_dir={args.save_dir}')
    Logger(f'  use_residual_scale={lm_config.use_residual_scale}, residual_scale_init={lm_config.residual_scale_init}')
    Logger(f'  mtp_depth={lm_config.mtp_depth}, mtp_loss_weight={lm_config.mtp_loss_weight}')
    Logger(f'  beta={args.beta}, learning_rate={args.learning_rate}, batch_size={args.batch_size}, accumulation_steps={args.accumulation_steps}')
    Logger(f'  max_seq_len={args.max_seq_len}, dtype={args.dtype}, num_workers={args.num_workers}, grad_clip={args.grad_clip}')
    Logger(f'  policy params={policy_params / 1e6:.3f}M, policy trainable params={policy_trainable_params / 1e6:.3f}M')
    Logger(f'  reference params={reference_params / 1e6:.3f}M, reference trainable params={reference_trainable_params / 1e6:.3f}M')

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: skip {start_step} steps, start from step {start_step + 1}')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    if dist.is_initialized(): dist.destroy_process_group()
