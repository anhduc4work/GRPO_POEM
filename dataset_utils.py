"""
Dataset utilities for Vietnamese Luc-Bat poem GRPO training.
Loads and prepares prompts for poem generation.
"""

from datasets import load_dataset, Dataset
import random

POEM_START = "<POEM>"
POEM_END = "</POEM>"

SYSTEM_PROMPT = """Bạn là một nhà thơ người Việt chuyên sáng tác thơ lục bát.
Thơ lục bát gồm các cặp câu 6 chữ và 8 chữ xen kẽ nhau.
Quy tắc thơ lục bát:
1. Câu 6 chữ (lục): chữ thứ 6 phải vần với chữ thứ 6 của câu 8 tiếp theo.
2. Câu 8 chữ (bát): chữ thứ 8 phải vần với chữ thứ 6 của câu 6 tiếp theo.
3. Thanh điệu: chữ thứ 2, 4, 6 trong câu lục phải xen kẽ bằng-trắc đúng quy tắc.
4. Số dòng phải là số chẵn.

Hãy suy nghĩ kỹ trước khi viết. Đặt bài thơ trong thẻ <POEM> và </POEM>.
"""

TOPICS = [
    "tình yêu đôi lứa",
    "quê hương đất nước",
    "mùa xuân",
    "mùa thu lá vàng",
    "nỗi nhớ",
    "gia đình",
    "thiên nhiên",
    "biển cả",
    "đêm trăng",
    "mẹ",
    "tuổi thơ",
    "chiến tranh và hòa bình",
    "lòng biết ơn",
    "hoa mai",
    "sông nước",
    "buổi sáng sớm",
    "người lính",
    "cuộc đời",
    "ước mơ",
    "làng quê",
]

PROMPT_TEMPLATES = [
    "Hãy sáng tác một bài thơ lục bát về chủ đề: {topic}.",
    "Viết một bài thơ lục bát 4 câu (2 cặp lục bát) về {topic}.",
    "Sáng tác thơ lục bát với chủ đề {topic}, gồm ít nhất 4 dòng.",
    "Hãy làm một bài thơ lục bát về {topic} theo đúng quy tắc gieo vần và thanh điệu.",
    "Cho tôi một bài thơ lục bát về {topic}.",
]


def make_prompt(topic: str) -> str:
    template = random.choice(PROMPT_TEMPLATES)
    return template.format(topic=topic)


def load_hf_poem_dataset(dataset_name: str = "phongnt109/luc-bat-poem") -> Dataset:
    """
    Load a Vietnamese luc-bat poem dataset from HuggingFace.
    Falls back to synthetic prompts if unavailable.
    """
    try:
        ds = load_dataset(dataset_name, split="train")
        return ds
    except Exception:
        return None


def build_grpo_dataset(
    num_samples: int = 2000,
    hf_dataset_name: str = "phongnt109/luc-bat-poem",
    seed: int = 42,
) -> Dataset:
    """
    Build a GRPO training dataset. Each row contains:
      - prompt: list of chat messages (system + user)
      - answer: reference poem (optional, not used by reward fns)
    """
    random.seed(seed)
    data = []

    # Try loading real poems for topics
    hf_ds = load_hf_poem_dataset(hf_dataset_name)

    if hf_ds is not None and len(hf_ds) > 0:
        # Use real poems as reference + extract topics from prompts
        cols = hf_ds.column_names
        poem_col = next((c for c in ["poem", "text", "content", "luc_bat"] if c in cols), cols[0])
        poems = [hf_ds[i][poem_col] for i in range(min(len(hf_ds), num_samples))]

        for poem in poems:
            topic = random.choice(TOPICS)
            prompt_text = make_prompt(topic)
            data.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "answer": poem,
                "topic": topic,
            })
    else:
        # Synthetic prompts only
        for _ in range(num_samples):
            topic = random.choice(TOPICS)
            prompt_text = make_prompt(topic)
            data.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "answer": "",
                "topic": topic,
            })

    # Shuffle
    random.shuffle(data)
    return Dataset.from_list(data)
