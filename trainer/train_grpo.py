import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_dipseek import DipSeekConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def count_params(module, trainable_only=False):
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def rule_reward(prompt: str, response: str, args) -> float:
    text = response.strip()
    lower = text.lower()
    reward = 0.0

    if args.length_reward:
        reward += 0.5 if 20 <= len(text) <= 1200 else -0.5
    if args.format_reward:
        reward += 0.5 if "```" in text else 0.0
        reward += 0.5 if any(x in lower for x in ["思路", "复杂度", "时间复杂度", "explanation", "approach"]) else 0.0
    if args.code_reward:
        code_keywords = ["def ", "class ", "import ", "return ", "for ", "while ", "#include", "public static", "function ", "const "]
        reward += 0.5 if any(k in lower for k in code_keywords) else 0.0
        refusal = ["cannot", "can't", "无法", "不能", "抱歉", "sorry", "as an ai"]
        reward += 0.5 if not any(x in lower for x in refusal) else -0.5
    if args.repetition_penalty_reward:
        reward -= rep_penalty(text)

    if len(text) < 5:
        reward -= 0.5
    if len(text) > 2000:
        reward -= 0.5
    if not text or text.count("\ufffd") >= 2:
        reward -= 0.5

    return reward


def parse_messages_from_prompt(prompt):
    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
    matches = re.findall(pattern, prompt, re.DOTALL)
    return [{"role": role, "content": content.strip()} for role, content in matches]


def calculate_rewards(prompts, responses, reward_model=None, args=None):
    device = args.device
    rule_scores = torch.zeros(len(responses), device=device)
    model_scores = torch.zeros(len(responses), device=device)

    for i, prompt in enumerate(prompts):
        for j in range(args.num_generations):
            idx = i * args.num_generations + j
            if args.reward_mode in ("rule", "hybrid"):
                rule_scores[idx] = rule_reward(prompt, responses[idx], args)

    if args.reward_mode in ("model", "hybrid"):
        if reward_model is None:
            raise ValueError("reward_mode is model/hybrid but reward_model is not loaded.")
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                messages = parse_messages_from_prompt(prompt)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    model_scores[idx] = float(reward_model.get_score(messages, responses[idx]))

    rewards = args.rule_reward_weight * rule_scores + args.reward_model_weight * model_scores
    return rewards, rule_scores, model_scores


def compute_group_advantages(rewards, num_generations, eps=1e-4):
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True).clamp(min=eps)
    advantages = ((grouped - mean) / std).view(-1)
    return advantages, grouped


def compute_completion_logps(model, input_ids, completion_ids, prompt_lens, attention_mask):
    outputs = model(input_ids, attention_mask=attention_mask)
    token_logps = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
    token_logps = torch.gather(token_logps, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=input_ids.device).unsqueeze(0)
    return token_logps.gather(1, pos), outputs.aux_loss


def build_completion_mask(completion_ids, completion_mask, eos_token_id):
    completion_mask = completion_mask.bool()
    is_eos = (completion_ids == eos_token_id) & completion_mask
    eos_idx = torch.full((completion_ids.size(0),), completion_ids.size(1) - 1, dtype=torch.long, device=completion_ids.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    pos = torch.arange(completion_ids.size(1), device=completion_ids.device).unsqueeze(0)
    return ((pos <= eos_idx.unsqueeze(1)) & completion_mask).float()


def compute_grpo_loss(
    per_token_logps,
    old_per_token_logps,
    ref_per_token_logps,
    advantages,
    completion_mask,
    beta,
    epsilon,
    epsilon_high,
    loss_type,
):
    old_per_token_logps = old_per_token_logps.detach()
    ref_per_token_logps = ref_per_token_logps.detach()
    ratio = torch.exp(per_token_logps - old_per_token_logps)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    if loss_type == "cispo":
        # CISPO keeps a one-sided high-ratio clamp and optimizes logprob directly.
        clipped_ratio = torch.clamp(ratio, max=epsilon_high)
        per_token_loss = -(clipped_ratio.detach() * advantages.unsqueeze(1) * per_token_logps - beta * per_token_kl)
        clip_mask = ratio > epsilon_high
    else:
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        per_token_loss1 = ratio * advantages.unsqueeze(1)
        per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
        per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - beta * per_token_kl)
        clip_mask = (ratio < 1 - epsilon) | (ratio > 1 + epsilon)

    token_counts = completion_mask.sum(dim=1).clamp(min=1)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / token_counts).mean()
    denom = completion_mask.sum().clamp(min=1)
    metrics = {
        "kl_mean": ((per_token_kl * completion_mask).sum() / denom).detach(),
        "ratio_mean": ((ratio * completion_mask).sum() / denom).detach(),
        "clip_frac": ((clip_mask.float() * completion_mask).sum() / denom).detach(),
    }
    return loss, metrics


def save_policy_checkpoint(lm_config, model, optimizer, scheduler, epoch, step, wandb=None):
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
    lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                  epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
    model.train()
    del state_dict


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None):
    global global_step

    for step, batch in enumerate(loader, start=start_step + 1):
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=bool(args.do_sample),
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        completion_mask = build_completion_mask(
            completion_ids,
            rollout_result.completion_mask.to(args.device),
            tokenizer.eos_token_id,
        )

        rewards, rule_rewards, model_rewards = calculate_rewards(prompts, completions, reward_model, args)
        advantages, grouped_rewards = compute_group_advantages(rewards, args.num_generations)

        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            per_token_logps, aux_loss = compute_completion_logps(model_unwrapped, outputs, completion_ids, prompt_lens, full_mask)
            aux_loss = aux_loss if lm_config.use_moe and aux_loss is not None else outputs.new_zeros((), dtype=torch.float32)
            with torch.no_grad():
                ref_per_token_logps, _ = compute_completion_logps(ref_model, outputs, completion_ids, prompt_lens, full_mask)

            policy_loss, loss_metrics = compute_grpo_loss(
                per_token_logps,
                old_per_token_logps,
                ref_per_token_logps,
                advantages,
                completion_mask,
                args.beta,
                args.epsilon,
                args.epsilon_high,
                args.loss_type,
            )
            loss = (policy_loss + aux_loss) / args.accumulation_steps

        loss.backward()

        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            rollout_engine.update_policy(model)
            global_step += 1

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            Logger(f"[DEBUG] step={step}")
            Logger(prompts[0])
            Logger(completions[0])
            Logger(f"[DEBUG] reward={rewards[0].item():.4f}, rule={rule_rewards[0].item():.4f}, model={model_rewards[0].item():.4f}")

        if step % args.log_interval == 0 or step == iters:
            avg_response_len = completion_mask.sum(dim=1).float().mean().item()
            log_values = {
                "loss": (loss.item() * args.accumulation_steps),
                "policy_loss": policy_loss.item(),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item(),
                "rule_reward_mean": rule_rewards.mean().item(),
                "model_reward_mean": model_rewards.mean().item(),
                "kl_mean": loss_metrics["kl_mean"].item(),
                "ratio_mean": loss_metrics["ratio_mean"].item(),
                "clip_frac": loss_metrics["clip_frac"].item(),
                "adv_mean": advantages.mean().item(),
                "adv_std": advantages.std(unbiased=False).item(),
                "avg_response_len": avg_response_len,
                "learning_rate": optimizer.param_groups[0]['lr'],
            }
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {log_values["loss"]:.4f}, policy_loss: {log_values["policy_loss"]:.4f}, '
                f'reward_mean: {log_values["reward_mean"]:.4f}, reward_std: {log_values["reward_std"]:.4f}, '
                f'rule_reward_mean: {log_values["rule_reward_mean"]:.4f}, model_reward_mean: {log_values["model_reward_mean"]:.4f}, '
                f'kl_mean: {log_values["kl_mean"]:.4f}, ratio_mean: {log_values["ratio_mean"]:.4f}, '
                f'clip_frac: {log_values["clip_frac"]:.4f}, adv_mean: {log_values["adv_mean"]:.4f}, '
                f'adv_std: {log_values["adv_std"]:.4f}, avg_response_len: {avg_response_len:.2f}, '
                f'lr: {log_values["learning_rate"]:.8f}'
            )
            if wandb and is_main_process():
                wandb.log(log_values)

        if not args.dry_run and (step % args.save_interval == 0 or step == iters) and is_main_process():
            save_policy_checkpoint(lm_config, model, optimizer, scheduler, epoch, step, wandb)
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, completions, old_per_token_logps
        del prompt_lens, full_mask, completion_mask, rewards, rule_rewards, model_rewards
        del advantages, grouped_rewards, per_token_logps, ref_per_token_logps, loss

        if args.dry_run:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DipSeek GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--max_steps", type=int, default=0, help="最大优化步数，0表示不限制")
    parser.add_argument("--dry_run", action="store_true", help="只跑1个batch且不保存")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_residual_scale', default=0, type=int, choices=[0, 1], help="是否启用Residual Scale（0=否，1=是）")
    parser.add_argument('--residual_scale_init', default=1.0, type=float, help="Residual Scale初始值")
    parser.add_argument('--mtp_depth', default=0, type=int, help="MTP-lite深度，GRPO阶段默认关闭")
    parser.add_argument('--mtp_loss_weight', default=0.0, type=float, help="MTP-lite loss权重，GRPO阶段默认0")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO ratio上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--reward_mode", type=str, default="rule", choices=["rule", "model", "hybrid"], help="reward来源：规则、reward model、混合")
    parser.add_argument("--reward_model_path", type=str, default="", help="可选Reward模型路径")
    parser.add_argument("--reward_model_weight", type=float, default=1.0)
    parser.add_argument("--rule_reward_weight", type=float, default=1.0)
    parser.add_argument("--code_reward", default=1, type=int, choices=[0, 1], help="是否启用代码规则奖励")
    parser.add_argument("--format_reward", default=1, type=int, choices=[0, 1], help="是否启用格式奖励")
    parser.add_argument("--length_reward", default=1, type=int, choices=[0, 1], help="是否启用长度奖励")
    parser.add_argument("--repetition_penalty_reward", default=1, type=int, choices=[0, 1], help="是否启用重复惩罚")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--do_sample", default=1, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="DipSeek-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下打印间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    if args.reward_mode in ("model", "hybrid") and not args.reward_model_path:
        raise ValueError("--reward_model_path is required when --reward_mode is model or hybrid.")

    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = DipSeekConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
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
        wandb_run_name = f"DipSeek-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    moe_suffix = '_moe' if lm_config.use_moe else ''
    weight_path = f'../out/{args.from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    Logger(f'Loading policy model from: {weight_path}')
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'Loading reference model from: {weight_path}')
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    reward_model = None
    if args.reward_mode in ("model", "hybrid"):
        reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
        Logger(f'Loaded reward model from: {args.reward_model_path}')

    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    iters = math.ceil(len(train_ds) / args.batch_size)
    max_optimizer_steps = args.max_steps if args.max_steps > 0 else math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, max_optimizer_steps), eta_min=args.learning_rate / 10)

    global_step = 0
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    Logger('GRPO config:')
    Logger(f'  from_weight={args.from_weight}, save_weight={args.save_weight}, save_dir={args.save_dir}')
    Logger(f'  use_residual_scale={lm_config.use_residual_scale}, residual_scale_init={lm_config.residual_scale_init}')
    Logger(f'  mtp_depth={lm_config.mtp_depth}, mtp_loss_weight={lm_config.mtp_loss_weight}')
    Logger(f'  reward_mode={args.reward_mode}, rule_reward_weight={args.rule_reward_weight}, reward_model_weight={args.reward_model_weight}')
    Logger(f'  beta={args.beta}, lr={args.learning_rate}, batch_size={args.batch_size}, num_generations={args.num_generations}')
    Logger(f'  max_seq_len={args.max_seq_len}, max_gen_len={args.max_gen_len}, dtype={args.dtype}, grad_clip={args.grad_clip}')
    Logger(f'  rollout_engine={args.rollout_engine}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, do_sample={bool(args.do_sample)}')
    Logger(f'  policy params={count_params(model) / 1e6:.3f}M, policy trainable params={count_params(model, True) / 1e6:.3f}M')
    Logger(f'  reference params={count_params(ref_model) / 1e6:.3f}M, reference trainable params={count_params(ref_model, True) / 1e6:.3f}M')

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, skip, wandb)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
        if args.dry_run:
            break

    if not args.dry_run and is_main_process():
        save_policy_checkpoint(lm_config, model, optimizer, scheduler, args.epochs - 1, global_step, wandb)

    if dist.is_initialized(): dist.destroy_process_group()
