# GRPO Fine-tuning: Qwen3-4B → Vietnamese Lục Bát Poem Generator

Fine-tunes **Qwen3-4B** using **GRPO** (Group Relative Policy Optimization) to generate Vietnamese **lục bát** poetry that follows traditional rules of rhyme and tone.

## Lục Bát Rules

Lục bát ("six-eight") is a Vietnamese poetic form where:
- Lines alternate between **6 words** (lục) and **8 words** (bát)
- **Rhyme**: the 6th word of a lục line must rhyme with the 6th word of the following bát line; the 8th word of the bát line must rhyme with the 6th word of the next lục line
- **Tone**: words at positions 2, 4, 6 follow strict even (bằng) / uneven (trắc) alternation rules
- Total lines must be **even**

## Pipeline

```
dataset_utils.py   →  Builds GRPO training prompts (HuggingFace + synthetic topics)
reward_functions.py →  5 reward functions based on lục bát rule checker
train_grpo.py       →  Main training script (Unsloth + TRL GRPOTrainer)
check_poem.py       →  Core Vietnamese poem rule checker (rhyme, tone, word count)
```

## Reward Functions

| Function | Description | Max Score |
|---|---|---|
| `reward_format` | Correct `<POEM>...</POEM>` tags | +2.0 |
| `reward_luc_bat_rules` | Full lục bát rule score (rhyme + tone) | +2.0 |
| `reward_line_count` | Even number of lines | +1.0 |
| `reward_word_count` | Alternating 6-8 word counts | +2.0 |
| `reward_think_present` | Uses `<think>` reasoning | +0.5 |

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
# Default (300 steps, LoRA 16bit)
python train_grpo.py

# QLoRA (less VRAM)
python train_grpo.py --load_in_4bit --max_steps 500

# Push LoRA to HuggingFace Hub
HF_TOKEN=your_token python train_grpo.py --push_to_hub
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `unsloth/Qwen3-4B` | Base model |
| `--max_steps` | `300` | Training steps |
| `--lora_rank` | `32` | LoRA rank |
| `--load_in_4bit` | `False` | Enable QLoRA |
| `--num_samples` | `2000` | Training prompts |
| `--push_to_hub` | `False` | Push to HF Hub |
| `--output_dir` | `grpo_poem_output` | Checkpoint dir |
| `--lora_save_path` | `grpo_poem_lora` | LoRA save dir |

### Inference

After training, the script automatically runs an inference test. To run manually:

```python
from unsloth import FastLanguageModel
from vllm import SamplingParams

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen3-4B", fast_inference=True)

messages = [
    {"role": "system", "content": "Bạn là nhà thơ chuyên sáng tác thơ lục bát..."},
    {"role": "user", "content": "Hãy sáng tác một bài thơ lục bát về mùa thu."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

output = model.fast_generate(
    [text],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
    lora_request=model.load_lora("grpo_poem_lora"),
)[0].outputs[0].text
print(output)
```

## References

- [fsoft-ailab/Poem-Generator](https://github.com/fsoft-ailab/Poem-Generator) — Vietnamese poem rules & scoring
- [Unsloth](https://github.com/unslothai/unsloth) — Fast LLM fine-tuning
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) — GRPO implementation
