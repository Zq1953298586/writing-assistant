#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《蒸馏数据集生成套件》之批量生成脚本
================================================
功能：调用本地 LM Studio（OpenAI 兼容接口），用老师模型批量生成
      "输入→高质量输出" 写作样本，保存为 JSONL，供后续 LoRA 微调使用。

运行前：
  1. 台式机打开 LM Studio，加载老师模型（如 qwen3.8-27b），点 Start Server
     （服务地址 http://localhost:1234）
  2. 安装依赖：pip install requests

运行：
  python scripts/蒸馏数据批量生成.py

输出：distill_output/distill_<任务>.jsonl（每类任务一个文件）
特性：失败自动重试3次 · input 哈希去重 · 进度打印 · 中断后重跑自动续上
"""

import hashlib
import json
import os
import sys
import time
from datetime import date

import requests  # pip install requests


# ============================================================
# 配置区（只改这里）
# ============================================================
BASE_URL = "http://localhost:1234/v1"   # LM Studio 本地服务地址
MODEL = "qwen3.8-27b"                   # LM Studio 里加载的老师模型名
API_KEY = "lm-studio"                   # LM Studio 不校验，随便填
TEMPERATURE = 0.8                       # 0.7~0.9，保证样本多样性
TIMEOUT = 120                           # 单次请求超时（秒）
RETRY = 3                               # 失败重试次数

# 每类任务要生成的条数（总量建议 2000~5000）
# 注意：条数上限 = 种子数量。种子不够时脚本会自动按现有种子生成并提示补充。
TASK_COUNTS = {
    "xuxie":   400,   # 续写
    "gai_xie": 400,   # 改写润色
    "fengge":  300,   # 风格仿写
    "piping":  300,   # 批评修改
    "kuoxie":  300,   # 扩写
    "suoxie":  300,   # 缩写
    "dagang":  200,   # 提纲大纲
    "biaoti":  300,   # 标题
}

OUT_DIR = "distill_output"              # 输出目录
SEED_FILE = ""  # 可选：种子 txt 路径，一行一条，格式 task|种子文本；留空则用内置示例种子
PROGRESS_FILE = os.path.join(OUT_DIR, "progress.json")  # 断点续跑记录

# 长度规则：(input最小, input最大, output最小, output最大)，单位：字符
LENGTH_RULES = {
    "xuxie":   (10, 300, 80, 1200),
    "gai_xie": (20, 800, 20, 1000),
    "fengge":  (20, 600, 20, 800),
    "piping":  (20, 800, 100, 2000),
    "kuoxie":  (10, 200, 150, 1200),
    "suoxie":  (100, 1500, 30, 600),
    "dagang":  (10, 300, 150, 2000),
    "biaoti":  (20, 600, 30, 600),
}

# 话术黑名单：output 里出现这些词整条丢弃
BANNED_PHRASES = ["作为AI", "作为 AI", "语言模型", "抱歉，我不能", "我是Meta"]


# ============================================================
# Teacher Prompt 模板（{seed} 会被替换成种子文本）
# 与 02-蒸馏数据集规范与示例.md 第 2 节一致
# ============================================================
COMMON_REQ = (
    "你是一位中文写作大师，正在为\"纯写作专用\"小模型生成训练样本。\n"
    "硬性要求：\n"
    "1. 只输出一个纯 JSON 对象，不要 markdown 代码围栏、不要解释、不要多余文字。\n"
    "2. JSON 只有两个键：\"input\"（给学生的输入）和 \"output\"（高质量标准答案）。\n"
    "3. 全中文，标点规范，不夹杂英文解释。\n"
    "4. 严禁出现\"作为AI\"\"我是一个语言模型\"\"抱歉\"等话术。"
)

TEACHER_PROMPTS = {
    "xuxie": COMMON_REQ + "\n【任务】续写。先在心里写出\"平庸续写\"并避开它的全部毛病"
    "（情节原地踏步、人物只说不做、没有感官细节），直接输出大师级续写作为 output："
    "情节向前推进、人物有具体行动、有感官细节、结尾留一个钩子。input 为下面的【开头】"
    "（原样使用，一个字不改）。\n【开头】{seed}\n只输出 JSON。",
    "gai_xie": COMMON_REQ + "\n【任务】改写润色。input 字段请原样使用下面的【草稿】"
    "（一个字都不要改动）；output 为大师改写版：保留原意与关键信息，修正病句，"
    "升级意象、节奏与文字质感。\n【草稿】{seed}\n只输出 JSON。",
    "fengge": COMMON_REQ + "\n【任务】风格仿写。种子格式为\"范文：……\\n待改写：……\"。"
    "先分析【范文】的风格特征（句式、节奏、意象、用词），input 为【待改写】原文"
    "（原样使用，不要改动），output 为按范文风格改写后的版本：只学风格不抄范文内容，"
    "意思与待改写一致。\n【种子】{seed}\n只输出 JSON。",
    "piping": COMMON_REQ + "\n【任务】批评并修改。input 为下面的【原稿】（原样使用，一个字不改）；"
    "output 分两部分：先写\"【批评】\"（分优点/问题/修改建议三条，问题必须具体到句子），"
    "再写\"【修改后】\"（应用全部修改建议后的全文）。\n【原稿】{seed}\n只输出 JSON。",
    "kuoxie": COMMON_REQ + "\n【任务】扩写。input 为下面的【梗概】（原样使用）；output 为扩写后的"
    "完整场景（300~600字）：加入环境细节、人物动作与对话、感官描写，有开头有收束，"
    "避开平庸毛病（只概括不展开、人物脸谱化）。\n【梗概】{seed}\n只输出 JSON。",
    "suoxie": COMMON_REQ + "\n【任务】缩写。input 为下面的【原文】（原样使用，一个字不改）；"
    "output 为缩写版：压缩到原文三分之一左右，保留核心信息、关键情节与最有味道的意象，"
    "语言更凝练。\n【原文】{seed}\n只输出 JSON。",
    "dagang": COMMON_REQ + "\n【任务】写故事大纲。input 为下面的【创意】（原样使用）；output 为完整大纲："
    "三幕结构（开端/对抗/结局）、主要人物一句话小传、至少5个关键情节点、结尾的情感落点。"
    "\n【创意】{seed}\n只输出 JSON。",
    "biaoti": COMMON_REQ + "\n【任务】起标题。input 为下面的【梗概】（原样使用）；output 为："
    "5个备选标题（风格各不相同：直白/悬念/意象/反差/金句），然后推荐1个并用一句话说明推荐理由。"
    "\n【梗概】{seed}\n只输出 JSON。",
}

# 每类任务给学生模型看的 instruction（微调时用）
INSTRUCTIONS = {
    "xuxie": "把下面的故事开头续写下去：情节向前推进、人物有具体行动、有感官细节，结尾留一个钩子。",
    "gai_xie": "把下面的草稿改写为大师版：保留原意与关键信息，修正病句，升级意象、节奏与文字质感。",
    "fengge": "把下面的待改写文本改成范文的风格：只学风格不抄内容，意思保持一致。",
    "piping": "先批评下面原稿（分优点/问题/修改建议，问题具体到句子），再给出应用全部建议后的全文。",
    "kuoxie": "把下面的一句话梗概扩写成完整场景（300~600字）：加入环境细节、人物动作与对话、感官描写，有开头有收束。",
    "suoxie": "把下面的原文缩写到三分之一左右：保留核心信息、关键情节与最有味道的意象，语言更凝练。",
    "dagang": "为下面的创意写一份故事大纲：三幕结构、主要人物一句话小传、至少5个关键情节点、结尾的情感落点。",
    "biaoti": "为下面的文章梗概起5个备选标题（风格各不相同），然后推荐1个并用一句话说明理由。",
}


# ============================================================
# 内置示例种子（先小批量验证流程；正式跑请用 SEED_FILE 提供几千条）
# ============================================================
BUILTIN_SEEDS = {
    "xuxie": [
        "雨下到第三天，老槐树底下那口井，传出了敲门声。",
        "林晚把辞职信拍在桌上时，窗外的无人机正好掠过，投下一片阴影。",
        "沙漠中央，那台报废了五十年的收音机，突然响了。",
        "他推开祖屋的门，发现堂屋的八仙桌上摆着两副碗筷——一副是他的，一副是三十年前去世的爷爷的。",
        "飞船的氧气只够六小时，而救援队的信号，来自三光年外。",
        "菜市场收摊的时候，陈婆忽然发现，自己的影子没有跟着她一起走。",
        "少年把最后一枚铜钱放进乞丐的碗里，铜钱却自己跳了出来，立在碗沿上。",
        "凌晨三点，电梯在14楼停下，门开了，外面站着一个和她长得一模一样的人。",
    ],
    "gai_xie": [
        "他非常非常的生气，脸都气红了，大声地吼道：“你怎么可以这样！”他的声音特别大，震得人耳朵疼。",
        "月亮很圆很亮，星星也很多，夜空非常美丽，她站在阳台上感觉心情很好。",
        "这个城市很大，人很多，车也很多，到处都很热闹，他觉得很新奇。",
        "她哭了，眼泪不停地流下来，心里特别难过，觉得世界都塌了。",
        "山很高，水很清，风景特别美，游客们都纷纷拿出手机拍照。",
        "他跑得很快，像风一样快，一下子就跑到了终点，大家都为他鼓掌。",
        "冬天来了，天气很冷，雪下得很大，地上全白了，小孩子们很开心。",
        "这家店的东西很好吃，味道特别好，价格也不贵，所以很多人来吃。",
    ],
    "fengge": [
        "范文：他站着，不动。雨落下来，落在肩上，不擦。\n待改写：小明站在雨中一动不动，雨水打湿了他的肩膀，他没有伸手去擦。",
        "范文：灯亮了，人散了，街空了。\n待改写：夜深的时候路灯亮了起来，街上的人都走光了，整条街变得空空荡荡。",
        "范文：她笑了一下，很短，像刀。\n待改写：她露出了一丝非常短暂的笑容，给人的感觉很锋利。",
        "范文：门开了，风进来，人没进来。\n待改写：门被推开之后，有风吹了进来，但是并没有人走进来。",
        "范文：他说了三个字，然后走了十年。\n待改写：他只说了短短的三个字，然后就离开了，一走就是十年。",
        "范文：雪落下来，盖住脚印，盖住来路。\n待改写：雪纷纷扬扬地落下来，把地上的脚印和来时的路都盖住了。",
        "范文：她没哭，眼泪自己掉了下来。\n待改写：她本来不想哭的，但是眼泪还是不受控制地流了下来。",
        "范文：钟停了，时间还在走。\n待改写：墙上的钟已经停了，但是时间依然在一分一秒地流逝。",
    ],
    "piping": [
        "王小二是个好人，他乐于助人，大家都喜欢他。有一天，他扶老奶奶过马路，老奶奶很感谢他。",
        "公主很漂亮，王子很英俊，他们经历了很多困难，最后过上了幸福的生活。",
        "春天来了，万物复苏，到处一片生机勃勃的景象，让人感到心旷神怡。",
        "他是一个勇敢的人，面对困难从不退缩，这种精神值得我们每一个人学习。",
        "夜深了，他独自走在回家的路上，心里感到非常孤独和寂寞。",
        "这家公司很大，实力很强，发展前景非常好，是很多人向往的地方。",
        "她的歌声非常动听，像百灵鸟一样，大家都沉醉在她的歌声中。",
        "比赛进入了白热化阶段，双方队员都拼尽了全力，场面非常激烈。",
    ],
    "kuoxie": [
        "外卖员在暴雨夜送最后一单，敲开门发现点餐的是十年前的自己。",
        "乡村教师退休那天，全校只有一名学生来送她。",
        "AI管家第一次对主人说了谎。",
        "老兵把勋章埋在了苹果树下。",
        "深海潜水员听到了不属于这个深度的歌声。",
        "小偷偷走的钱包里，有一张他自己的通缉令。",
        "最后一个守灯塔的人，决定熄灯。",
        "女孩把愿望写进漂流瓶，瓶子漂回了她家门口。",
    ],
    "suoxie": [
        "清晨五点半，天还没有完全亮，菜市场已经热闹起来了。卖菜的张大爷把一捆捆青菜摆得整整齐齐，菜叶上还挂着露水；卖鱼的老李正把活蹦乱跳的鲫鱼往水盆里倒，水花溅到了他的围裙上；早点摊的油条在油锅里翻滚，金黄酥脆，香气飘出半条街。李奶奶挎着篮子，挨个摊位讨价还价，脸上笑开了花。这里的一天，就是这样在讨价还价和招呼声中开始的。",
        "他沿着山路一直往上走，走了大概两个小时，腿都走酸了，终于看到了山顶的那座小庙。庙不大，红墙灰瓦，门口有两棵老松树，枝干遒劲，松针落了一地。推开虚掩的木门，院子里静悄悄的，只有一只黄狗趴在石阶上晒太阳，听见动静抬头看了他一眼，又懒洋洋地趴了回去。香炉里插着几炷没烧完的香，青烟袅袅上升，在阳光里弯弯曲曲地散开。",
        "会议从下午两点开到五点，议题一个接一个。先是销售部汇报了上季度的业绩，数据不太好看，总经理的脸色有点沉；接着讨论了新产品的上市计划，市场部和研发部争了半天，一个要快一个要稳，谁也说服不了谁；最后说到团建的事，气氛才稍微轻松了一点，定了下个月去郊外烧烤。散会的时候，天已经擦黑了，大家都饿着肚子往外走，心里想的都是晚上吃什么。",
        "老槐树底下，几个老人摇着蒲扇下棋，棋子落在石桌上啪啪作响。旁边的小卖部里，冰棍箱上结着白霜，老板娘一边织毛衣一边招呼客人。不远处的空地上，几个孩子在踢毽子，毽子上的红绒绳上下翻飞，笑声传出老远。一只花猫蹲在墙头上，眯着眼睛打盹，尾巴有一搭没一搭地晃着。整个胡同的午后，慢得像化不开的糖。",
        "他打开电脑，新建了一个文档，标题想了半天也没想好。窗外的雨下个不停，雨点打在玻璃上，发出单调的声响。他端起杯子喝了一口，咖啡已经凉了，苦得皱眉。手机震了一下，是朋友发来的消息，问他稿子写得怎么样了。他回了个“快了”，然后把手机扣在桌上，深吸一口气，手指终于落在了键盘上。",
        "火车穿过隧道的时候，车厢里暗了下来，只有应急灯发出幽幽的光。孩子停止了哭闹，好奇地盯着窗外一闪而过的灯。对面的老人把报纸折好，放进布包里，闭上眼睛养神。列车员推着小车走过来，叫卖声在车厢里回荡。小桌板上，半杯茶水随着车身轻轻晃动，荡出一圈圈涟漪。隧道很长，黑暗持续了整整三分钟。",
        "她把行李箱拖进出租屋的时候，房间里还残留着上一任租客的味道。她打开窗户，风灌进来，窗帘鼓了起来。楼下是条小吃街，烧烤的烟火气混着炒面的香气往上飘。她把被子铺好，坐在床沿上发了会儿呆，然后起身把带来的那盆绿萝放在了窗台上。绿萝的叶子垂下来，在晚风里轻轻晃，像在跟她打招呼。",
        "村口的老井已经干了，井沿上长满了青苔。村里人说，这口井有上百年了，闹饥荒那几年，全村都靠它活命。现在家家都打了压水井，没人再来挑水了。只有每年清明，村长还会带着人来，把井台打扫一遍，摆上两盘供果。风一吹，井绳在辘轳上轻轻晃，发出吱呀吱呀的声响，像在说什么，又像什么都没说。",
    ],
    "dagang": [
        "一个能听见植物说话的女孩，发现城市绿化带在策划一场逃亡。",
        "退休杀手开了家早餐店，每位客人都点同一碗面。",
        "时间银行倒闭那天，人们排队取出自己存下的岁月。",
        "最后一个说书人发现，书里的故事正在一个个消失。",
        "快递员误把包裹送到了阴间，签收人是他去世的父亲。",
        "小镇上每下一场雨，就会有一个人忘记最重要的事。",
        "AI写的小说获了奖，评委不知道作者不是人。",
        "守墓人每晚都能收到墓主人的来信。",
    ],
    "biaoti": [
        "文章讲一位父亲每天凌晨四点起床给女儿做早餐，十年如一日，女儿考上大学那天才发现父亲一直在用左手——右手早年在工地受伤使不上力。",
        "写城市里最后一间修钢笔的小店，老师傅坚守三十年，年轻人排队来学，最后发现大家学的不是手艺，是慢下来。",
        "讲一个程序员用AI复刻了去世母亲的声音，每天跟“她”聊天，直到有一天AI说出了母亲从未告诉过他的秘密。",
        "写一位乡村教师在只有一个学生的学校坚守了二十年，学生考上大学那天，全村人凑钱为他摆了三天流水席。",
        "讲深夜食堂的老板娘记住每一位熟客的口味，有位客人连续来了十年，某天没来，老板娘沿着他常走的路找了过去。",
        "写一位快递员在暴雨夜送最后一单，敲开门发现点餐的是十年前的自己，两人聊了一整夜。",
        "讲城市绿化带的行道树一夜之间全部“搬家”，园林工人发现树根下压着一封写给市长的信。",
        "写一位守灯塔的老人，在灯塔自动化改造的最后一天，决定亲手熄灭那盏亮了四十年的灯。",
    ],
}


# ============================================================
# 工具函数
# ============================================================
def load_seeds():
    """加载种子：优先读 SEED_FILE（格式 task|种子文本），否则用内置种子。"""
    seeds = {t: [] for t in TASK_COUNTS}
    if SEED_FILE and os.path.exists(SEED_FILE):
        with open(SEED_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" not in line:
                    print("[警告] 种子行格式不对，已跳过：%s" % line[:30])
                    continue
                task, text = line.split("|", 1)
                task, text = task.strip(), text.strip()
                if task in seeds and text:
                    seeds[task].append(text)
                else:
                    print("[警告] 未知任务或空种子，已跳过：%s" % line[:30])
        print("[种子] 从文件加载：%s" % SEED_FILE)
    else:
        seeds = {t: list(v) for t, v in BUILTIN_SEEDS.items()}
        print("[种子] 使用内置示例种子（正式跑请用 SEED_FILE 提供几千条）")
    return seeds


def load_progress():
    """读取断点续跑记录。"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_seen_hashes():
    """从已有输出文件重建 input 哈希集合（去重用）。"""
    seen = set()
    if not os.path.isdir(OUT_DIR):
        return seen
    for name in os.listdir(OUT_DIR):
        if not name.startswith("distill_") or not name.endswith(".jsonl"):
            continue
        with open(os.path.join(OUT_DIR, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    seen.add(hash_text(rec.get("input", "")))
                except Exception:
                    continue
    return seen


def hash_text(s):
    """归一化后取 sha256（去空白、统一换行）。"""
    norm = "".join(s.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def check_server():
    """检查 LM Studio 服务是否在线、模型是否存在。"""
    try:
        r = requests.get(BASE_URL.rstrip("/") + "/models", timeout=10)
        r.raise_for_status()
        models = [m.get("id", "") for m in r.json().get("data", [])]
    except Exception as e:
        print("[错误] 连不上 %s，请确认 LM Studio 已启动服务。(%s)" % (BASE_URL, e))
        sys.exit(1)
    if MODEL not in models:
        print("[警告] 模型 %s 不在已加载列表 %s 中，将按填写的名字请求，" % (MODEL, models))
        print("       若报错请把 MODEL 改成列表里的名字。")
    else:
        print("[连接] 服务正常，老师模型：%s" % MODEL)


def chat_once(prompt):
    """调一次 /v1/chat/completions，失败重试 RETRY 次。"""
    url = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "stream": False,
    }
    last_err = None
    for attempt in range(1, RETRY + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            print("    [重试 %d/%d] 出错：%s" % (attempt, RETRY, e))
            time.sleep(2 * attempt)
    raise RuntimeError("连续 %d 次请求失败：%s" % (RETRY, last_err))


def extract_pair(text):
    """从老师回复里提取 input/output，兼容代码围栏包裹。"""
    t = text.strip()
    if t.startswith("```"):  # 去掉 ```json ... ``` 包裹
        lines = t.split("\n")
        lines = lines[1:] if len(lines) > 1 else []
        t = "\n".join(lines)
        if "```" in t:
            t = t.rsplit("```", 1)[0]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("回复里找不到 JSON 对象")
    obj = json.loads(t[start:end + 1])
    return obj["input"], obj["output"]


def validate(task, input_text, output_text):
    """长度过滤 + 内容禁区检查，不合格返回原因，否则返回 None。"""
    in_min, in_max, out_min, out_max = LENGTH_RULES[task]
    if not (in_min <= len(input_text) <= in_max):
        return "input 长度 %d 超出 [%d,%d]" % (len(input_text), in_min, in_max)
    if not (out_min <= len(output_text) <= out_max):
        return "output 长度 %d 超出 [%d,%d]" % (len(output_text), out_min, out_max)
    for bad in BANNED_PHRASES:
        if bad in output_text:
            return "output 含禁用话术：%s" % bad
    en_chars = sum(1 for c in output_text if c.isascii() and c.isalpha())
    if len(output_text) > 0 and en_chars / len(output_text) > 0.10:
        return "output 英文占比超 10%"
    if input_text.strip() == output_text.strip():
        return "input 与 output 完全相同"
    return None


def make_record(task, input_text, output_text):
    return {
        "task": task,
        "instruction": INSTRUCTIONS[task],
        "input": input_text.strip(),
        "output": output_text.strip(),
        "metadata": {
            "teacher_model": MODEL,
            "created_at": str(date.today()),
            "lang": "zh",
            "license_note": "Qwen系列允许输出用于训练",
        },
    }


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    check_server()
    seeds = load_seeds()
    progress = load_progress()
    seen = load_seen_hashes()
    if seen:
        print("[去重] 已载入 %d 个历史 input 哈希" % len(seen))

    total_ok, total_skip = 0, 0
    for task, want in TASK_COUNTS.items():
        task_seeds = seeds.get(task, [])
        if not task_seeds:
            print("\n[%s] 没有种子，跳过。请在 SEED_FILE 里补充。" % task)
            continue
        target = min(want, len(task_seeds))
        if want > len(task_seeds):
            print("\n[%s] 种子只有 %d 条（想要 %d 条），先按 %d 条生成；"
                  "补充种子后重跑会自动续上。" % (task, len(task_seeds), want, len(task_seeds)))
        done = progress.get(task, 0)
        if done >= target:
            print("\n[%s] 已完成 %d/%d，跳过。" % (task, done, target))
            continue

        out_path = os.path.join(OUT_DIR, "distill_%s.jsonl" % task)
        print("\n[%s] 目标 %d 条，从第 %d 条开始 → %s" % (task, target, done + 1, out_path))
        ok, skip = 0, 0
        with open(out_path, "a", encoding="utf-8") as fout:
            for idx in range(done, target):
                seed = task_seeds[idx]
                try:
                    raw = chat_once(TEACHER_PROMPTS[task].format(seed=seed))
                    input_text, output_text = extract_pair(raw)
                except Exception as e:
                    print("  [%d/%d] 生成失败：%s" % (idx + 1, target, e))
                    skip += 1
                    continue
                reason = validate(task, input_text, output_text)
                if reason:
                    print("  [%d/%d] 质检丢弃：%s" % (idx + 1, target, reason))
                    skip += 1
                    continue
                h = hash_text(input_text)
                if h in seen:
                    print("  [%d/%d] 重复 input，丢弃" % (idx + 1, target))
                    skip += 1
                    continue
                seen.add(h)
                fout.write(json.dumps(make_record(task, input_text, output_text),
                                      ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                progress[task] = idx + 1
                save_progress(progress)  # 每条都存，中断可续跑
                if ok % 10 == 0 or idx + 1 == target:
                    print("  [%d/%d] 已落盘 %d 条" % (idx + 1, target, ok))
        print("[%s] 本轮完成：%d 条，丢弃 %d 条" % (task, ok, skip))
        total_ok += ok
        total_skip += skip

    print("\n全部完成：共生成 %d 条，丢弃 %d 条，输出目录 %s/" % (total_ok, total_skip, OUT_DIR))


if __name__ == "__main__":
    main()
