# AutoDL 上配置 MiniMind 最新版训练环境完整流程

> 适用场景：在 AutoDL 租用 GPU 实例后，从零配置 MiniMind 最新版环境，完成代码拉取、Conda 环境创建、缓存目录设置、依赖安装、数据下载，并准备运行 MiniMind 64M 训练。  
> 推荐所有代码、环境、缓存、数据、日志、权重都放在数据盘 `/root/autodl-tmp`，避免 30GB 系统盘爆满。

---

## 0. 机器配置参考

本流程基于类似如下 AutoDL 实例：

```text
GPU: NVIDIA GeForce RTX 4080 SUPER / 32GB
CPU: 16 核
内存: 62GB
系统盘: 30GB
数据盘: /root/autodl-tmp，50GB 或更大
GPU 驱动: 580.x
CUDA: ≤ 13.0
```

MiniMind 64M / hidden_size=768 / num_hidden_layers=8 可以在 32GB 显存上较轻松运行。

---

## 1. 连接 AutoDL 实例

### 方式 A：JupyterLab Terminal

在 AutoDL 控制台：

```text
容器实例
→ 找到已开机实例
→ 点击 JupyterLab
→ 打开 Terminal
```

### 方式 B：本地 SSH

从 AutoDL 实例页面复制 SSH 命令，格式一般类似：

```bash
ssh -p 端口 root@region-xxx.autodl.com
```

第一次连接时输入：

```text
yes
```

然后输入实例密码。

---

## 2. 检查 GPU 和磁盘

连接后先执行：

```bash
nvidia-smi
```

确认能看到 GPU，例如：

```text
NVIDIA GeForce RTX 4080 SUPER
Memory-Usage: 0MiB / 32760MiB
```

查看磁盘：

```bash
pwd
df -h
ls -lah /root/autodl-tmp
```

重点确认数据盘 `/root/autodl-tmp` 存在。

---

## 3. 在数据盘创建目录

所有大文件都放到 `/root/autodl-tmp`：

```bash
cd /root/autodl-tmp

mkdir -p projects
mkdir -p conda_envs
mkdir -p conda_pkgs
mkdir -p pip_cache
mkdir -p hf_cache
```

目录说明：

```text
/root/autodl-tmp/projects       # 项目代码
/root/autodl-tmp/conda_envs     # Conda 环境
/root/autodl-tmp/conda_pkgs     # Conda 包缓存
/root/autodl-tmp/pip_cache      # pip 缓存
/root/autodl-tmp/hf_cache       # Hugging Face 缓存
```

---

## 4. 设置缓存目录到数据盘

当前终端执行：

```bash
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

写入 `~/.bashrc`，以后新终端自动生效：

```bash
echo 'export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache' >> ~/.bashrc
echo 'export HF_HOME=/root/autodl-tmp/hf_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub' >> ~/.bashrc
echo 'export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets' >> ~/.bashrc
```

如果 Hugging Face 官方源无法访问，建议后面使用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

也可以写入 `~/.bashrc`：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

---

## 5. 配置 Conda 环境和缓存到数据盘

加载 Conda：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
```

配置 Conda 环境目录和包缓存目录：

```bash
conda config --add envs_dirs /root/autodl-tmp/conda_envs
conda config --add pkgs_dirs /root/autodl-tmp/conda_pkgs
```

创建独立环境：

```bash
conda create -p /root/autodl-tmp/conda_envs/minimind_latest python=3.10 -y
```

激活环境：

```bash
conda activate /root/autodl-tmp/conda_envs/minimind_latest
```

确认 Python 路径在数据盘：

```bash
which python
python -V
```

期望输出类似：

```text
/root/autodl-tmp/conda_envs/minimind_latest/bin/python
Python 3.10.x
```

---

## 6. 拉取 MiniMind 最新版代码

进入项目目录：

```bash
cd /root/autodl-tmp/projects
```

Clone 最新版 MiniMind：

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind.git minimind_latest
```

进入项目：

```bash
cd /root/autodl-tmp/projects/minimind_latest
```

查看目录结构：

```bash
ls -lah
```

最新版应包含：

```text
dataset/
model/
trainer/
scripts/
eval_llm.py
requirements.txt
README.md
```

---

## 7. 安装 PyTorch CUDA 版

你的 AutoDL 驱动可能显示 CUDA 13.0，但 PyTorch wheel 自带 CUDA runtime，不需要单独安装系统 CUDA Toolkit。

推荐安装 PyTorch 2.6.0 + cu126：

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

如果网络下载依赖很慢，可以先用清华源安装普通依赖，再安装 PyTorch：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  numpy==1.26.4 \
  pillow==11.3.0 \
  sympy==1.13.1 \
  mpmath==1.3.0 \
  networkx==3.4.2 \
  jinja2==3.1.6 \
  filelock \
  fsspec \
  typing_extensions
```

然后再安装 PyTorch：

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

期望输出：

```text
2.6.0+cu126
True
NVIDIA GeForce RTX 4080 SUPER
```

---

## 8. 安装 MiniMind 依赖

进入项目根目录：

```bash
cd /root/autodl-tmp/projects/minimind_latest
```

安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装绘图工具：

```bash
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装 Hugging Face CLI：

```bash
pip install "huggingface_hub==0.36.2" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如遇到 `rich` 版本冲突，可修复：

```bash
pip install "rich==13.7.1" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

检查核心包：

```bash
python -c "import torch, transformers, datasets, huggingface_hub; print(torch.__version__); print(transformers.__version__); print(datasets.__version__); print(huggingface_hub.__version__)"
```

---

## 9. 注意：不要随便升级 huggingface_hub

不要执行：

```bash
pip install -U huggingface_hub
```

它可能会把 `huggingface_hub` 升到 `1.x`，导致和 `transformers 4.57.x` 冲突。

如果你已经升级坏了，修复：

```bash
pip install "huggingface_hub==0.36.2" "rich==13.7.1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 10. 下载 MiniMind 训练数据

进入项目根目录：

```bash
cd /root/autodl-tmp/projects/minimind_latest
```

设置 Hugging Face 镜像和缓存：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

创建数据目录：

```bash
mkdir -p dataset
```

下载 mini 预训练数据：

```bash
hf download jingyaogong/minimind_dataset pretrain_t2t_mini.jsonl \
  --repo-type dataset \
  --local-dir dataset
```

下载 mini SFT 数据：

```bash
hf download jingyaogong/minimind_dataset sft_t2t_mini.jsonl \
  --repo-type dataset \
  --local-dir dataset
```

检查：

```bash
ls -lh dataset
```

期望看到：

```text
pretrain_t2t_mini.jsonl
sft_t2t_mini.jsonl
```

---

## 11. 可选：下载作者官方医疗 LoRA 数据

作者官方提供了医疗 LoRA 数据：

```bash
hf download jingyaogong/minimind_dataset lora_medical.jsonl \
  --repo-type dataset \
  --local-dir dataset
```

检查：

```bash
ls -lh dataset/lora_medical.jsonl
wc -l dataset/lora_medical.jsonl
```

---

## 12. 创建输出目录

进入项目根目录：

```bash
cd /root/autodl-tmp/projects/minimind_latest
```

创建日志、图片、权重和断点目录：

```bash
mkdir -p runs/logs
mkdir -p runs/plots
mkdir -p out
mkdir -p checkpoints
```

---

## 13. 使用 screen 防止断连

长时间训练不要裸跑，建议使用 `screen`：

```bash
screen -S minimind_latest
```

进入 screen 后重新激活环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/minimind_latest
cd /root/autodl-tmp/projects/minimind_latest
```

重新设置缓存：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
```

退出 screen 但不中断任务：

```text
Ctrl + A，然后按 D
```

重新进入 screen：

```bash
screen -r minimind_latest
```

查看 screen：

```bash
screen -ls
```

---

## 14. 运行 64M Pretrain

进入 trainer 目录：

```bash
cd /root/autodl-tmp/projects/minimind_latest/trainer
```

运行 Pretrain：

```bash
python -u train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --epochs 1 \
  --batch_size 32 \
  --accumulation_steps 1 \
  --max_seq_len 340 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --dtype float16 \
  --num_workers 8 \
  --log_interval 50 \
  --save_interval 999999 \
  --save_weight pretrain_64m_latest \
  > ../runs/logs/pretrain_64m_latest.log 2>&1
```

查看日志：

```bash
tail -f /root/autodl-tmp/projects/minimind_latest/runs/logs/pretrain_64m_latest.log
```

查看 GPU：

```bash
nvidia-smi -l 1
```

---

## 15. 运行 64M SFT

Pretrain 结束后运行：

```bash
cd /root/autodl-tmp/projects/minimind_latest/trainer

python -u train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --epochs 1 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --dtype float16 \
  --num_workers 8 \
  --log_interval 50 \
  --save_interval 999999 \
  --from_weight pretrain_64m_latest \
  --save_weight full_sft_64m_latest \
  > ../runs/logs/sft_64m_latest.log 2>&1
```

查看 SFT 日志：

```bash
tail -f /root/autodl-tmp/projects/minimind_latest/runs/logs/sft_64m_latest.log
```

---

## 16. 运行作者官方医疗 LoRA

确保 `full_sft_64m_latest_768.pth` 已存在：

```bash
cd /root/autodl-tmp/projects/minimind_latest
ls -lh out | grep full_sft
```

新开 screen：

```bash
screen -S lora_medical
```

进入 screen 后：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/minimind_latest
cd /root/autodl-tmp/projects/minimind_latest/trainer
```

运行医疗 LoRA：

```bash
python -u train_lora.py \
  --data_path ../dataset/lora_medical.jsonl \
  --epochs 1 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --dtype float16 \
  --num_workers 8 \
  --log_interval 50 \
  --save_interval 999999 \
  --from_weight full_sft_64m_latest \
  --lora_name lora_medical_64m_latest \
  > ../runs/logs/lora_medical_64m_latest.log 2>&1
```

查看日志：

```bash
tail -f /root/autodl-tmp/projects/minimind_latest/runs/logs/lora_medical_64m_latest.log
```

---

## 17. 可选：准备计算机 LoRA 数据

如果使用 Magicoder 转换数据，需要安装 datasets：

```bash
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
```

创建转换脚本：

```bash
cd /root/autodl-tmp/projects/minimind_latest
mkdir -p scripts_local

cat > scripts_local/convert_magicoder_to_minimind_lora.py <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

out_path = Path("dataset/lora_computer_magicoder.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")

print(ds)
print("columns:", ds.column_names)

max_samples = 30000
written = 0

with out_path.open("w", encoding="utf-8") as f:
    for ex in ds:
        instruction = (
            ex.get("instruction")
            or ex.get("problem")
            or ex.get("prompt")
            or ex.get("question")
            or ""
        )
        response = (
            ex.get("response")
            or ex.get("solution")
            or ex.get("output")
            or ex.get("answer")
            or ""
        )

        instruction = str(instruction).strip()
        response = str(response).strip()

        if not instruction or not response:
            continue

        obj = {
            "conversations": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        }

        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        written += 1

        if written >= max_samples:
            break

print(f"saved={out_path}")
print(f"written={written}")
PY
```

运行转换：

```bash
cd /root/autodl-tmp/projects/minimind_latest

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets

python scripts_local/convert_magicoder_to_minimind_lora.py
```

检查：

```bash
ls -lh dataset/lora_computer_magicoder.jsonl
wc -l dataset/lora_computer_magicoder.jsonl
head -n 1 dataset/lora_computer_magicoder.jsonl
```

注意：Magicoder 数据偏英文代码题，不一定适合中文计算机问答 LoRA。如果目标是中文计算机基础问答，建议重新制作中文高质量数据。

---

## 18. 可选：运行计算机 LoRA

```bash
screen -S lora_computer
```

进入 screen：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/minimind_latest
cd /root/autodl-tmp/projects/minimind_latest/trainer
```

运行训练：

```bash
python -u train_lora.py \
  --data_path ../dataset/lora_computer_magicoder.jsonl \
  --epochs 1 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --max_seq_len 768 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --dtype float16 \
  --num_workers 8 \
  --log_interval 50 \
  --save_interval 999999 \
  --from_weight full_sft_64m_latest \
  --lora_name lora_computer_64m_latest \
  > ../runs/logs/lora_computer_64m_latest.log 2>&1
```

查看日志：

```bash
tail -f /root/autodl-tmp/projects/minimind_latest/runs/logs/lora_computer_64m_latest.log
```

---

## 19. 创建 Loss 绘图脚本

进入项目根目录：

```bash
cd /root/autodl-tmp/projects/minimind_latest
mkdir -p scripts_local runs/plots
```

创建脚本：

```bash
cat > scripts_local/plot_loss.py <<'PY'
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--out", default="runs/plots")
args = parser.parse_args()

log_path = Path(args.log)
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

raw = log_path.read_bytes()
try:
    text = raw.decode("utf-8-sig")
except UnicodeDecodeError:
    text = raw.decode("gbk", errors="ignore")

text = re.sub(r"\x1b\[[0-9;]*m", "", text)
text = text.replace("\r", "\n")

rows = []

for line in text.splitlines():
    if "loss:" not in line:
        continue

    step_match = re.search(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)", line)
    loss_match = re.search(r"loss:\s*([0-9]+(?:\.[0-9]+)?)", line)
    logits_match = re.search(r"logits_loss:\s*([0-9]+(?:\.[0-9]+)?)", line)
    aux_match = re.search(r"aux_loss:\s*([0-9]+(?:\.[0-9]+)?)", line)
    lr_match = re.search(r"lr:\s*([0-9.]+)", line)

    if not loss_match:
        continue

    if step_match:
        epoch = int(step_match.group(1))
        step = int(step_match.group(3))
        total_steps = int(step_match.group(4))
    else:
        epoch = 1
        step = len(rows) + 1
        total_steps = 0

    rows.append({
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "loss": float(loss_match.group(1)),
        "logits_loss": float(logits_match.group(1)) if logits_match else "",
        "aux_loss": float(aux_match.group(1)) if aux_match else "",
        "lr": float(lr_match.group(1)) if lr_match else "",
    })

if not rows:
    print("没有解析到 loss。日志路径：", log_path)
    raise SystemExit(1)

csv_path = out_dir / f"{args.name}.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "step", "total_steps", "loss", "logits_loss", "aux_loss", "lr"]
    )
    writer.writeheader()
    writer.writerows(rows)

x = [r["step"] for r in rows]
y = [r["loss"] for r in rows]

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.xlabel("train step")
plt.ylabel("loss")
plt.title(args.name)
plt.grid(True)
plt.tight_layout()

png_path = out_dir / f"{args.name}_loss.png"
plt.savefig(png_path, dpi=160)

print(f"解析到 loss 点数: {len(rows)}")
print(f"CSV 已保存: {csv_path}")
print(f"PNG 已保存: {png_path}")
PY
```

---

## 20. 生成 Loss 图

```bash
cd /root/autodl-tmp/projects/minimind_latest

python scripts_local/plot_loss.py \
  --log runs/logs/pretrain_64m_latest.log \
  --name pretrain_64m_latest \
  --out runs/plots

python scripts_local/plot_loss.py \
  --log runs/logs/sft_64m_latest.log \
  --name sft_64m_latest \
  --out runs/plots

python scripts_local/plot_loss.py \
  --log runs/logs/lora_medical_64m_latest.log \
  --name lora_medical_64m_latest \
  --out runs/plots

python scripts_local/plot_loss.py \
  --log runs/logs/lora_computer_64m_latest.log \
  --name lora_computer_64m_latest \
  --out runs/plots
```

查看：

```bash
ls -lh runs/plots
```

---

## 21. 测试模型

### 21.1 测试 SFT 模型

```bash
cd /root/autodl-tmp/projects/minimind_latest

python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_64m_latest \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_new_tokens 256 \
  --device cuda
```

### 21.2 测试医疗 LoRA

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_64m_latest \
  --lora_weight lora_medical_64m_latest \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_new_tokens 256 \
  --device cuda
```

### 21.3 测试计算机 LoRA

```bash
python eval_llm.py \
  --load_from model \
  --save_dir out \
  --weight full_sft_64m_latest \
  --lora_weight lora_computer_64m_latest \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_new_tokens 256 \
  --device cuda
```

---

## 22. 打包结果下载

进入项目根目录：

```bash
cd /root/autodl-tmp/projects/minimind_latest
```

打包结果：

```bash
tar -czf minimind_local_continue.tar.gz \
  eval_llm.py \
  model \
  trainer \
  scripts \
  scripts_local \
  requirements.txt \
  runs/logs \
  runs/plots \
  out
```

检查：

```bash
ls -lh minimind_local_continue.tar.gz
```

可以通过 JupyterLab 文件栏下载，也可以用本地 PowerShell 的 `scp` 下载。

---

## 23. 本地继续测试

下载并解压到本地，例如：

```text
E:\Pycharm Project\MyGPT\minimind\fromAutodl
```

进入本地目录：

```powershell
conda activate "E:\conda_envs\minimind"

Set-Location "E:\Pycharm Project\MyGPT\minimind\fromAutodl"
```

测试 CUDA：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

本地测试 SFT：

```powershell
python eval_llm.py `
  --load_from model `
  --save_dir out `
  --weight full_sft_64m_latest `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_new_tokens 256 `
  --device cuda
```

本地测试医疗 LoRA：

```powershell
python eval_llm.py `
  --load_from model `
  --save_dir out `
  --weight full_sft_64m_latest `
  --lora_weight lora_medical_64m_latest `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_new_tokens 256 `
  --device cuda
```

本地测试计算机 LoRA：

```powershell
python eval_llm.py `
  --load_from model `
  --save_dir out `
  --weight full_sft_64m_latest `
  --lora_weight lora_computer_64m_latest `
  --hidden_size 768 `
  --num_hidden_layers 8 `
  --max_new_tokens 256 `
  --device cuda
```

---

## 24. 常见问题

### 24.1 `bash: ../runs/logs/xxx.log: No such file or directory`

原因：日志目录不存在。

解决：

```bash
cd /root/autodl-tmp/projects/minimind_latest
mkdir -p runs/logs runs/plots out checkpoints
```

### 24.2 `Network is unreachable`

原因：AutoDL 访问 Hugging Face 官方站点失败。

解决：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

再重新执行 `hf download`。

### 24.3 `huggingface_hub` 和 `transformers` 版本冲突

如果看到类似：

```text
transformers requires huggingface-hub<1.0,>=0.34.0
```

修复：

```bash
pip install "huggingface_hub==0.36.2" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 24.4 `rich` 和 `swanlab` 冲突

如果看到类似：

```text
swanlab requires rich<14.0.0
```

修复：

```bash
pip install "rich==13.7.1" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 24.5 笔记本关机训练会不会继续？

如果训练在 `screen` 里运行，并且 AutoDL 实例没有关机，训练会继续。

确认：

```bash
screen -ls
```

退出 screen 但不中断：

```text
Ctrl + A，然后按 D
```

重新进入：

```bash
screen -r minimind_latest
```

如果训练没有在 screen 里运行，关闭终端可能导致训练被杀掉。  
推荐：中断当前训练，在 screen 里重新启动。

### 24.6 计算机 LoRA 效果差

如果使用 Magicoder 数据，注意它主要是英文代码题，不适合中文计算机基础问答。  
如果目标是中文计算机领域 LoRA，建议重新制作中文高质量问答数据，例如：

```text
Python / 算法 / 数据结构
计算机网络
数据库 / MySQL / Redis
操作系统 / Linux / Docker / Git
PyTorch / Transformer / LLM 训练
```

---

## 25. 推荐执行顺序总结

```text
1. 连接 AutoDL
2. 检查 nvidia-smi 和 df -h
3. 在 /root/autodl-tmp 创建 projects / conda_envs / hf_cache 等目录
4. 创建 Conda 环境 minimind_latest
5. git clone 最新 MiniMind
6. 安装 PyTorch 2.6.0 + cu126
7. 安装 requirements.txt
8. 设置 HF_ENDPOINT=https://hf-mirror.com
9. 下载 pretrain_t2t_mini.jsonl 和 sft_t2t_mini.jsonl
10. 创建 runs/logs、runs/plots、out、checkpoints
11. screen -S minimind_latest
12. 跑 train_pretrain.py
13. 跑 train_full_sft.py
14. 下载 lora_medical.jsonl
15. 跑 train_lora.py 医疗 LoRA
16. 可选：准备计算机 LoRA 数据并训练
17. 画 loss 图
18. eval_llm.py 测试
19. tar 打包结果
20. 下载到本地继续分析
```
