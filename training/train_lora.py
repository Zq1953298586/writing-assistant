#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================
专属写作助手 · LoRA 微调脚本 v1
----------------------------------------------------------------
做什么：用 Unsloth + LoRA，在 Qwen3 上只训练「中文写作」能力，
        得到一支专属于你的笔。别的什么都不教它。

硬件：RTX 3090 24GB（下面超参按 24GB 显存调好，不用动）
环境：WSL2 里的 Ubuntu + Python 3.11
运行：python train_lora.py

一条命令跑完全流程：
  1. 下载基座模型          4. 保存 LoRA 权重（真正的"笔芯"，很小）
  2. 加载你的写作数据集      5. 合并成完整模型（16bit）
  3. LoRA 微调              6. 转成 GGUF（拖进 LM Studio / Ollama 就能用）

【数据红线】训练数据必须全部来自「蒸馏数据 + 你自己的语料」，
八类纯写作任务，不要混入通用问答、代码、闲聊数据，
否则会把"只做写作"的定位冲淡，笔就变钝了。
================================================================
"""

import json
import os
import sys

# ==================== ★ 顶部配置：只改这里 ====================
# 基座模型：默认 Qwen3-4B，先用它把全流程跑通。
# 跑通之后想上 8B：把下一行改成 "unsloth/Qwen3-8B"，并把 LORA_R 改成 32，
# 再把 GRAD_ACCUM 改成 8，其余不用动。
MODEL_NAME = "unsloth/Qwen3-4B"

LORA_R     = 16     # LoRA 的 rank：4B 用 16；8B 建议 32（更能装下你的文风）
LORA_ALPHA = 16     # 一般和 rank 取一样就行，不用动

MAX_SEQ_LENGTH = 2048  # 单条样本最长 2048 字；写超长文时可改 4096（显存占用会涨）

DATA_PATH = "./data/writing_data.jsonl"  # 你的 2000~5000 条写作数据放这里（一行一条）
OUTPUT_DIR = "./output"                   # 所有产物都写到这里

LORA_DIR   = "./output/writing-lora"          # 训练出的 LoRA 权重
MERGED_DIR = "./output/writing-merged-16bit"  # 合并后的完整模型
GGUF_DIR   = "./output/writing-gguf-q8_0"     # 转好的 GGUF，给 LM Studio / Ollama 用

NUM_EPOCHS    = 2      # 数据 3000 条以上用 2；不到 3000 条可以改成 3
LEARNING_RATE = 2e-4   # LoRA 经典学习率，不要乱改
BATCH_SIZE    = 2      # 每步吃 2 条样本
GRAD_ACCUM    = 4      # 攒 4 步再更新一次 → 有效 batch = 2x4 = 8；换 8B 时改成 8
SEED = 3407            # 随机种子，固定它保证可复现
# ==============================================================


# 写死的人设：只做写作。训练时每条样本都会带上它，
# 等于给模型立了一条"道"：我是笔，不是大脑。
SYSTEM_PROMPT = (
    "你是张义的专属写作助手，一支只会写作的笔。"
    "你只做中文写作：续写、改写润色、风格仿写、批评修改、扩写、缩写、列提纲、起标题。"
    "与写作无关的请求，一律礼貌拒绝，并把话题带回写作。"
    "写作时用自然流畅的现代中文，少用陈词滥调，多用具体细节。"
)

# 八类写作任务：数据里 task 字段就用这些名字
TASK_NAMES = ["续写", "改写润色", "风格仿写", "批评修改", "扩写", "缩写", "提纲", "标题"]


def load_writing_data(path):
    """读取 JSONL 写作数据，做格式检查，转成对话格式。

    每行两种写法都认：
      写法 A：{"task": "xxx", "instruction": "……", "input": "……（可空）", "output": "……"}
      写法 B：{"messages": [{"role": "user", "content": "……"},
                            {"role": "assistant", "content": "……"}]}
    """
    if not os.path.exists(path):
        print("\n[错误] 找不到数据文件：" + path)
        print("  请把你的写作数据存成这个路径再跑（详见 README-小白版.md 第 4 步）。")
        sys.exit(1)

    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("[错误] 第 %d 行不是合法 JSON，请检查。" % i)
                sys.exit(1)
            items.append(obj)

    if len(items) == 0:
        print("[错误] 数据文件是空的，请先放入写作数据。")
        sys.exit(1)

    # 统计八类任务分布，帮你一眼看出数据偏不偏科
    from collections import Counter
    dist = Counter(o.get("task", "未标注") for o in items if "messages" not in o)
    print("\n[数据] 共 %d 条，任务分布：" % len(items))
    for t in TASK_NAMES:
        print("  %s：%d 条" % (t, dist.get(t, 0)))
    unknown = [k for k in dist if k not in TASK_NAMES]
    if unknown:
        print("  [警告] 发现未知 task 名字 %s，请改成上面八类之一。" % unknown)

    # 转成带 system 人设的对话
    convos = []
    for o in items:
        if "messages" in o:
            messages = o["messages"]
        else:
            if "instruction" not in o or "output" not in o:
                print("[错误] 有数据行缺少 instruction 或 output 字段，请检查。")
                sys.exit(1)
            user_text = o["instruction"]
            if o.get("input"):  # 有原文就拼在后面
                user_text += "\n\n【原文】\n" + o["input"]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": o["output"]},
            ]
        convos.append(messages)
    return convos


def main():
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ---- 0. 看看显卡在不在 ----
    print("\n[显卡] " + torch.cuda.get_device_name(0))
    print("  显存：%.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3))

    # ---- 1. 下载并加载基座模型（4bit 量化，省显存） ----
    print("\n[1/6] 加载基座模型：" + MODEL_NAME + "（首次运行会下载约 2~3GB，请耐心）")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,         # 自动选最合适的精度（3090 会用 bfloat16）
        load_in_4bit=True,  # 4bit 加载：显存占用不到一半，写作质量几乎无损
    )

    # ---- 2. 挂上 LoRA（只训练一小部分参数，又快又省） ----
    print("\n[2/6] 挂载 LoRA：rank=%d，alpha=%d" % (LORA_R, LORA_ALPHA))
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # 注意力+前馈全覆盖
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,      # 0 是 Unsloth 优化过的最快选项
        bias="none",
        use_gradient_checkpointing="unsloth",  # 省显存大法，3090 必备
        random_state=SEED,
    )

    # ---- 3. 加载并格式化写作数据 ----
    print("\n[3/6] 读取写作数据：" + DATA_PATH)
    convos = load_writing_data(DATA_PATH)

    def formatting_prompts_func(examples):
        # Qwen3 是"会思考"的模型，训练写作时必须关掉思考模式，
        # 否则它会在正文里夹带 think 标签。enable_thinking=False 就是干这个的。
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            ) + tokenizer.eos_token
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    dataset = Dataset.from_dict({"conversations": convos})
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)
    print("  格式化完成，示例长度：%d 字符" % len(dataset[0]["text"]))

    # ---- 4. 开始训练 ----
    print("\n[4/6] 开始训练……中途不要关机，3090 上 4B 模型大约 1~4 小时")
    print("  盯着 loss 看：它应该一路往下走（比如从 1.5 降到 0.3 左右）")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",   # 用上面格式化好的 text 列
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,  # 有效 batch = 2x4 = 8
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",  # 学习率先升后降，收敛更稳
            warmup_steps=20,             # 前 20 步热身，别一上来就猛冲
            optim="adamw_8bit",          # 8bit Adam，省显存
            weight_decay=0.01,
            logging_steps=10,            # 每 10 步打印一次 loss
            save_steps=200,              # 每 200 步存一个检查点
            save_total_limit=2,          # 只留最近 2 个，省硬盘
            output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
            seed=SEED,
            report_to="none",            # 不用 wandb，省事
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
        ),
    )
    trainer.train()
    # 万一中途断掉，下次想接着训，把上一行改成下面这行再跑：
    # trainer.train(resume_from_checkpoint=True)

    # ---- 5. 保存 LoRA 权重（这就是你的"笔芯"，只有几十 MB） ----
    print("\n[5/6] 保存 LoRA 权重到：" + LORA_DIR)
    model.save_pretrained(LORA_DIR)
    tokenizer.save_pretrained(LORA_DIR)

    # ---- 6. 合并成完整模型（16bit，需要一点内存，128GB 绰绰有余） ----
    print("\n[6/6] 合并 LoRA 到基座模型：" + MERGED_DIR + "（约 8GB，稍等几分钟）")
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")

    # ---- 7. 转成 GGUF，给 LM Studio / Ollama 用 ----
    # q8_0：体积约 4.3GB（4B），质量最接近原版，推荐。
    # 想要更小更快：把 "q8_0" 改成 "q4_k_m"（约 2.5GB），改完单独重跑这一步就行，不用重训。
    print("\n[7/6] 转 GGUF（q8_0）：" + GGUF_DIR)
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q8_0")

    gguf_files = [f for f in os.listdir(GGUF_DIR) if f.endswith(".gguf")]
    gguf_file = os.path.join(GGUF_DIR, gguf_files[0]) if gguf_files else GGUF_DIR
    print("\n" + "=" * 60)
    print("全部完成！你的专属写作模型做好了。")
    print("  GGUF 文件在：" + gguf_file)
    print("  下一步：打开 LM Studio，把这个 .gguf 文件拖进去就能用；")
    print("  详细步骤见 README-小白版.md 第 8 步。")
    print("  别忘了按《评测方案.md》跑一遍盲测，确认笔是变利了而不是变钝了。")
    print("=" * 60)


if __name__ == "__main__":
    main()


# ==================================================================
# 附：训练完想快速试一句（可选）
# 把下面这段单独存成 test.py 跑，会用刚训好的 LoRA 写一段话。
# ------------------------------------------------------------------
# from unsloth import FastLanguageModel
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="./output/writing-lora",  # 直接加载 LoRA，自动找基座
#     max_seq_length=2048, dtype=None, load_in_4bit=True,
# )
# FastLanguageModel.for_inference(model)
# messages = [
#     {"role": "system", "content": "你是张义的专属写作助手，一支只会写作的笔。"},
#     {"role": "user", "content": "给下面这句起三个标题：雨点敲在铁皮屋顶上，像谁在门外轻轻叩门。"},
# ]
# inputs = tokenizer.apply_chat_template(messages, tokenize=True,
#     add_generation_prompt=True, return_tensors="pt").to("cuda")
# out = model.generate(inputs, max_new_tokens=256, temperature=0.7)
# print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
# ==================================================================
