import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_dipseek import DipSeekConfig, DipSeekForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def load_checkpoint_state_dict(ckp, device):
    checkpoint = torch.load(ckp, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint at {ckp} is not a state_dict-compatible dict.")
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in checkpoint.items()
    }

def infer_arch_from_state_dict(state_dict):
    use_residual_scale = any(
        ("attn_res_scale" in k or "mlp_res_scale" in k)
        for k in state_dict.keys()
    )

    mtp_depth = 0
    for k in state_dict.keys():
        if k.startswith("mtp_heads."):
            parts = k.split(".")
            if len(parts) > 1:
                try:
                    idx = int(parts[1])
                    mtp_depth = max(mtp_depth, idx + 1)
                except ValueError:
                    pass

    return use_residual_scale, mtp_depth

def resolve_checkpoint_path(args):
    if args.checkpoint_path is not None:
        return args.checkpoint_path
    moe_suffix = '_moe' if args.use_moe else ''
    return f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        ckp = resolve_checkpoint_path(args)
        state_dict = load_checkpoint_state_dict(ckp, args.device)
        if args.auto_arch:
            auto_use_residual_scale, auto_mtp_depth = infer_arch_from_state_dict(state_dict)
            args.use_residual_scale = int(auto_use_residual_scale)
            args.mtp_depth = auto_mtp_depth
            print(f"[auto_arch] use_residual_scale={args.use_residual_scale}, mtp_depth={args.mtp_depth}")
        model = DipSeekForCausalLM(DipSeekConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling,
            use_residual_scale=bool(args.use_residual_scale),
            residual_scale_init=args.residual_scale_init,
            mtp_depth=args.mtp_depth,
            mtp_loss_weight=args.mtp_loss_weight,
        ))
        missing, unexpected = model.load_state_dict(state_dict, strict=bool(args.strict_load))
        if missing:
            print(f"[load warning] missing keys count={len(missing)}, examples={missing[:20]}")
        if unexpected:
            print(f"[load warning] unexpected keys count={len(unexpected)}, examples={unexpected[:20]}")
        if args.lora_weight != 'None':
            target_modules = tuple(
                x.strip() for x in args.lora_target.split(',') if x.strip()
            )
            apply_lora(
                model,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                target_modules=target_modules,
                freeze_base=True,
            )
            lora_dir = args.lora_dir if args.lora_dir is not None else args.save_dir
            load_lora(model, f'./{lora_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="DipSeek模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=8, type=int, help="LoRA rank，必须和训练时一致")
    parser.add_argument('--lora_alpha', default=16, type=int, help="LoRA alpha，必须和训练时一致")
    parser.add_argument('--lora_target', default='q_proj,v_proj', type=str, help="LoRA target modules，必须和训练时一致")
    parser.add_argument('--lora_dir', default=None, type=str, help="LoRA权重目录，None时使用save_dir")
    parser.add_argument('--checkpoint_path', default=None, type=str, help="完整checkpoint路径，提供后优先使用它")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--use_residual_scale', default=0, type=int, choices=[0, 1], help="是否启用Residual Scale结构")
    parser.add_argument('--residual_scale_init', default=1.0, type=float, help="Residual Scale初始值")
    parser.add_argument('--mtp_depth', default=0, type=int, help="MTP-lite深度，0表示关闭")
    parser.add_argument('--mtp_loss_weight', default=0.0, type=float, help="MTP-lite loss权重，推理时通常为0")
    parser.add_argument('--strict_load', default=1, type=int, choices=[0, 1], help="是否严格加载checkpoint")
    parser.add_argument('--auto_arch', default=1, type=int, choices=[0, 1], help="是否从checkpoint key自动推断residual_scale和mtp_depth")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--repetition_penalty', default=1.15, type=float, help="重复惩罚，建议1.10~1.25")
    parser.add_argument('--no_repeat_ngram_size', default=4, type=int, help="禁止重复n-gram，建议3~5")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
