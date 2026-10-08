# 专属写作助手 · LoRA 微调小白指南

> 给张义的说明：这份指南假设你只会开关机、打字、复制粘贴。
> 每一步都写了"做什么、点哪里/敲什么、成功长什么样、失败了看哪里"。
> 不要跳步，按顺序走。**全程约：装环境 1 小时 + 训练 1~4 小时**（训练时电脑不用管，去干别的）。

---

## 先搞懂三件事（30 秒）

1. **WSL2 是什么**：Windows 自带的一个"小 Ubuntu 系统"，训练必须在它里面跑。你不用懂 Linux，照着复制命令就行。
2. **训练时电脑在干嘛**：显卡在反复读你的写作数据，学你的文风。屏幕上会滚动数字，**不要关机、不要关窗口**。
3. **最后得到什么**：一个 `.gguf` 文件（约 4GB），拖进 LM Studio 就能用——你的专属写作模型诞生。

训练数据必须全部来自「蒸馏数据 + 你自己的语料」，八类纯写作任务。
不要混入通用问答、代码、闲聊数据，否则"只做写作"的定位会被冲淡。

---

## 第 0 步：确认显卡驱动装好（Windows 里做，5 分钟）

**做什么**：训练用的是显卡，没有驱动等于没有发动机。

1. 在 Windows 桌面按 `Win` 键，输入"设备管理器"，打开它。
2. 展开"显示适配器"，确认能看到 **NVIDIA GeForce RTX 3090**。
3. 没看到 → 去 NVIDIA 官网下载驱动装上（搜"NVIDIA 驱动下载"，选 GeForce RTX 3090 / Win10 64位）。
4. 看到了 → 直接去第 1 步。

✅ **成功长这样**：设备管理器里有 RTX 3090，没有黄色感叹号。

---

## 第 1 步：安装 WSL2（Windows 里做，约 10 分钟）

**做什么**：请 Windows 把那个"小 Ubuntu 系统"装好。

1. 点"开始"菜单，输入 `powershell`，在"Windows PowerShell"上**右键 → 以管理员身份运行**，点"是"。
2. 在弹出的黑窗口里，复制粘贴下面这行（右键 = 粘贴），按回车：
```
wsl --install
```
3. 它会自动下载安装，**中途会让你重启电脑** → 重启。
4. 重启后会自动弹出一个 Ubuntu 窗口，让你**设置用户名和密码**：
- 用户名：随便起，比如 `zhangyi`（只能用小写英文）
- 密码：输两遍（输入时屏幕上**什么都不显示**，这是正常的，输完按回车就行）
5. 看到 `zhangyi@电脑名:~$` 这样的提示符 → 安装成功。

✅ **成功长这样**：黑窗口里出现 `~$` 结尾的提示符，能打字。
❌ **失败了看哪里**：如果提示"虚拟化未启用"，重启电脑进 BIOS 打开 VT-x（华硕/微星主板一般在 Advanced → CPU 设置里），这一步如果卡住就喊我。

---

## 第 2 步：给 Ubuntu 装 Python（Ubuntu 窗口里做，约 5 分钟）

**做什么**：装运行训练脚本需要的"地基"。

1. 打开 Ubuntu（开始菜单搜 `Ubuntu`）。
2. 一行一行复制粘贴下面三行，每行粘贴后按回车，等它跑完再粘贴下一行：
```
sudo apt update
```
```
sudo apt install -y python3.11 python3.11-venv python3-pip
```
```
python3.11 --version
```
- `sudo` 开头的命令会问你要密码 → 输入第 1 步设的密码（输入时不显示，输完回车）。
3. 最后看到 `Python 3.11.x` → 成功。

✅ **成功长这样**：最后一行显示 `Python 3.11` 开头的版本号。
❌ **失败了看哪里**：如果 `apt update` 报错网络问题，多等一会儿重跑一次；还不行就喊我。

---

## 第 3 步：确认 Ubuntu 能看到显卡（Ubuntu 窗口里做，1 分钟）

**做什么**：确认"小系统"能用上你的 3090。这是最关键的一步。

复制粘贴运行：
```
nvidia-smi
```

✅ **成功长这样**：出现一张表格，第一行写着 `NVIDIA GeForce RTX 3090`，显存 `24576MiB`。
❌ **失败了看哪里**：
- 提示 `command not found` → 回到第 0 步，Windows 里重装一遍 NVIDIA 驱动，然后 Ubuntu 里重跑这条命令。
- 表格出来了但没有 3090 → 喊我。

---

## 第 4 步：建工作文件夹，放进数据和脚本（Ubuntu 窗口里做，5 分钟）

**做什么**：建一个"工地"，把训练脚本和你的写作数据放进去。

1. 你会收到我给你的两个东西：`train_lora.py` 和你的数据文件 `writing_data.jsonl`。
- `writing_data.jsonl`：你的 2000~5000 条写作数据，**一行一条**，格式见本目录 `data/示例_5条.jsonl`（打开看看就懂）。
- ⚠️ 数据里不要掺通用问答/代码/闲聊，只要八类写作任务：续写、改写润色、风格仿写、批评修改、扩写、缩写、提纲、标题。
2. 在 Ubuntu 里运行下面几行（建文件夹）：
```
mkdir -p ~/writing-tune/data
cd ~/writing-tune
```
3. 把 `train_lora.py` 和 `writing_data.jsonl` **复制到**对应位置：
- 最省事的办法：在 Windows 文件资源管理器地址栏输入 `\\wsl$\Ubuntu\home\zhangyi\writing-tune`（把 `zhangyi` 换成你的用户名），回车，就能像普通文件夹一样把文件拖进去。
- `train_lora.py` → 直接放在 `writing-tune` 里
- `writing_data.jsonl` → 放在 `writing-tune\data` 里
4. 回到 Ubuntu 窗口，运行检查：
```
ls ~/writing-tune ~/writing-tune/data
wc -l ~/writing-tune/data/writing_data.jsonl
```

✅ **成功长这样**：第一行显示 `train_lora.py` 和 `data`；最后一行显示一个数字（比如 `3200`），这就是你的数据条数。
❌ **失败了看哪里**：`wc -l` 报找不到文件 → 第 3 步的文件没放对位置，重新拖一次。

---

## 第 5 步：装训练环境（Ubuntu 窗口里做，约 20~40 分钟）

**做什么**：装 Unsloth（训练工具）。下载量大，耐心等。

一行一行运行：
```
cd ~/writing-tune
```
```
python3.11 -m venv venv
```
```
source venv/bin/activate
```
（看到行首出现 `(venv)` → 虚拟环境已进入，**以后每次打开 Ubuntu 跑训练前，都要先执行这两行**）
```
pip install unsloth
```

⏳ 这一步下载几 GB 的东西，**慢是正常的**，去喝杯水。看到 `Successfully installed...` → 成功。

✅ **成功长这样**：最后出现 `Successfully installed unsloth...`。
❌ **失败了看哪里**：
- 下载慢到超时 → 换国内源重装：
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple unsloth
```
- 报错缺东西 → 把完整的红色报错复制给我。

---

## 第 6 步：训练前最后检查（Ubuntu 窗口里做，2 分钟）

**做什么**：确认"发动机"真的点得着火。两条命令：

```
source ~/writing-tune/venv/bin/activate
```
```
python -c "import torch; print('显卡可用:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

✅ **成功长这样**：
```
显卡可用: True
NVIDIA GeForce RTX 3090
```
❌ **失败了看哪里**：显示 `False` → 回到第 3 步重查 `nvidia-smi`；还是不行就喊我，**不要强行往下走**。

---

## 第 7 步：开始训练（Ubuntu 窗口里做，1~4 小时）

**做什么**：真正开训。一条命令，之后不用管。

```
cd ~/writing-tune
source venv/bin/activate
python train_lora.py
```

会发生什么（按顺序）：
1. 先下载基座模型（约 2~3GB，**第一次慢，正常**）。
2. 打印"📊 数据共 N 条，任务分布"——**对一下数字**，和你的数据条数一致才对。
3. 开始滚动 `loss` 数字：**loss 应该一路往下走**（比如从 1.5 降到 0.3 左右），这就是"学进去了"。
4. 跑完自动合并、转 GGUF，最后打印 `🎉 全部完成！`

⏳ **期间**：不要关 Ubuntu 窗口，不要让电脑休眠（Windows 设置 → 电源 → 睡眠改成"从不"，训完再改回来）。你可以最小化窗口去干别的。

✅ **成功长这样**：最后看到 `🎉 全部完成！`，`~/writing-tune/output/writing-gguf-q8_0/` 里有一个 `.gguf` 文件。
❌ **失败了看哪里**：
| 看到的现象 | 怎么办 |
|---|---|
| `CUDA out of memory`（显存爆了） | 打开 `train_lora.py`，把 `BATCH_SIZE` 改成 1，重跑 |
| `loss` 变成 `nan` 或一路往上涨 | 停掉，把 `LEARNING_RATE` 改成 `1e-4`，重跑；还不行喊我 |
| 下载模型卡住不动超过 30 分钟 | 按 `Ctrl+C` 停掉，过会儿重跑（支持断点续传）；或喊我教你换国内镜像 |
| 中途断电/关机了 | 重跑 `python train_lora.py`，把 `trainer.train()` 那行按脚本里的注释改成 `resume_from_checkpoint=True` 再跑 |

---

## 第 8 步：把模型拖进 LM Studio（Windows 里做，5 分钟）

**做什么**：让训练成果真正能用。

1. 在 Windows 文件资源管理器地址栏输入 `\\wsl$\Ubuntu\home\zhangyi\writing-tune\output\writing-gguf-q8_0`，找到那个 `.gguf` 文件（约 4GB），**复制到** Windows 下一个好找的地方，比如 `D:\models\`。
2. 打开 **LM Studio** → 左侧点"💬 Chat" → 顶部模型选择框 → 点"导入模型"（或直接把 `.gguf` 文件拖进 LM Studio 窗口）。
3. 选中它，点 Load。**第一次加载要几十秒**，正常。
4. 在对话框里输入：`把下面这句续写下去：雨点敲在铁皮屋顶上，像谁在门外轻轻叩门。`
5. 看它写得像不像你的文风——这就是你的专属写作模型了。

✅ **成功长这样**：模型加载成功，对话框正常出字，文风对味。
❌ **失败了看哪里**：LM Studio 提示格式不支持 → 确认拖的是 `writing-gguf-q8_0` 文件夹里的 `.gguf` 文件，不是别的文件夹。

---

## 第 9 步（可选）：导入 Ollama（Windows 里做）

如果你更习惯用 Ollama：

1. 在 `.gguf` 文件旁边新建一个文本文件，命名为 `Modelfile`（没有后缀），内容三行：
```
FROM./writing-assistant-q8_0.gguf
PARAMETER temperature 0.7
SYSTEM "你是张义的专属写作助手，一支只会写作的笔。"
```
（第一行的文件名改成你实际的 gguf 文件名）
2. 在这个文件夹里打开 PowerShell，运行：
```
ollama create writing-assistant -f Modelfile
ollama run writing-assistant
```

---

## 第 10 步：跑评测（确认笔变利了，而不是变钝了）

训练完**不要直接宣布成功**。按同目录《评测方案.md》跑一遍 50 条盲测：
- 微调版明显更好 → 成功，可以开始用它写东西了。
- 差不多 → 数据加量或把 epoch 改成 3 再训一轮。
- 反而变差（训歪了）→ 按评测方案里的"回滚"操作，LM Studio 里切回原来的基座模型用，LoRA 权重留着，喊我一起看数据哪里出了问题。

---

## 换 8B 模型时要改的三处

全流程跑通、4B 效果满意后，想上 Qwen3-8B：
1. `train_lora.py` 顶部：`MODEL_NAME` 改成 `"unsloth/Qwen3-8B"`
2. `LORA_R` 改成 `32`
3. `GRAD_ACCUM` 改成 `8`

其余不动，重新跑第 7 步。8B 的 GGUF 约 8.5GB，训练时间约 2~6 小时。
