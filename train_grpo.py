"""
GRPO Fine-tuning Pipeline for Qwen3-4B → Vietnamese Luc-Bat Poem Generator
============================================================================
Uses Unsloth + TRL GRPOTrainer with custom Vietnamese poem reward functions.

Usage:
    python train_grpo.py [--config config.yaml]

Or edit CONFIG below directly.
"""

import os
import gc
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional

from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from dataset_utils import build_grpo_dataset, SYSTEM_PROMPT
from reward_functions import (
    reward_format,
    reward_luc_bat_rules,
    reward_line_count,
    reward_word_count,
    reward_think_present,
)


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = "unsloth/Qwen3-4B"
    max_seq_length: int = 1024
    lora_rank: int = 32
    lora_alpha: int = 64
    load_in_4bit: bool = False          # True = QLoRA (less VRAM), False = LoRA 16bit
    gpu_memory_utilization: float = 0.75

    # Dataset
    dataset_name: str = "phongnt109/luc-bat-poem"
    num_samples: int = 2000
    seed: int = 42

    # Training
    learning_rate: float = 5e-6
    num_generations: int = 6            # Completions per prompt (GRPO K)
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 300
    warmup_ratio: float = 0.05
    max_grad_norm: float = 0.1
    temperature: float = 1.0
    max_prompt_length: int = 256
    max_completion_length: int = 512
    logging_steps: int = 10
    save_steps: int = 50

    # Output
    output_dir: str = "grpo_poem_output"
    lora_save_path: str = "grpo_poem_lora"
    push_to_hub: bool = False
    hub_model_id: str = "anhduc4work/qwen3-4b-viet-poem-grpo"
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))


# ─── Load model ──────────────────────────────────────────────────────────────

def load_model(cfg: TrainConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        fast_inference=True,
        max_lora_rank=cfg.lora_rank,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    return model, tokenizer


# ─── Build GRPO trainer ───────────────────────────────────────────────────────

def build_trainer(model, tokenizer, dataset, cfg: TrainConfig):
    vllm_sampling_params = SamplingParams(
        min_p=0.05,
        top_p=0.9,
        top_k=50,
        seed=cfg.seed,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
        max_tokens=cfg.max_completion_length,
    )

    training_args = GRPOConfig(
        # vLLM
        use_vllm=True,
        vllm_sampling_params=vllm_sampling_params,
        # Training
        learning_rate=cfg.learning_rate,
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        temperature=cfg.temperature,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        # Logging
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        output_dir=cfg.output_dir,
        # Misc
        seed=cfg.seed,
        report_to="none",
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_format,           # +2 for correct <POEM> tags
            reward_luc_bat_rules,    # main luc-bat rule score
            reward_line_count,       # even line count
            reward_word_count,       # 6-8 alternating word counts
            reward_think_present,    # small bonus for <think> usage
        ],
        args=training_args,
        train_dataset=dataset,
    )
    return trainer


# ─── Inference test ──────────────────────────────────────────────────────────

def test_inference(model, tokenizer, lora_path: str, cfg: TrainConfig):
    test_prompts = [
        "Hãy sáng tác một bài thơ lục bát về chủ đề tình yêu đôi lứa.",
        "Viết một bài thơ lục bát về mùa thu lá vàng.",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_tokens=512,
    )

    print("\n" + "="*60)
    print("INFERENCE TEST")
    print("="*60)

    for prompt_text in test_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        output = model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=model.load_lora(lora_path),
        )[0].outputs[0].text

        print(f"\nPrompt: {prompt_text}")
        print(f"Output:\n{output}\n")
        print("-"*40)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GRPO train Qwen3-4B for Vietnamese Luc-Bat poems")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--output_dir", type=str, default="grpo_poem_output")
    parser.add_argument("--lora_save_path", type=str, default="grpo_poem_lora")
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit,
        num_samples=args.num_samples,
        push_to_hub=args.push_to_hub,
        output_dir=args.output_dir,
        lora_save_path=args.lora_save_path,
    )

    print(f"[1/4] Loading model: {cfg.model_name}")
    model, tokenizer = load_model(cfg)

    print(f"[2/4] Building dataset (n={cfg.num_samples})")
    dataset = build_grpo_dataset(
        num_samples=cfg.num_samples,
        hf_dataset_name=cfg.dataset_name,
        seed=cfg.seed,
    )
    print(f"      Dataset size: {len(dataset)}")

    print("[3/4] Starting GRPO training...")
    trainer = build_trainer(model, tokenizer, dataset, cfg)
    trainer.train()

    print(f"[4/4] Saving LoRA to '{cfg.lora_save_path}'")
    model.save_lora(cfg.lora_save_path)

    if cfg.push_to_hub and cfg.hf_token:
        print(f"      Pushing to Hub: {cfg.hub_model_id}")
        model.push_to_hub_merged(
            cfg.hub_model_id,
            tokenizer,
            save_method="lora",
            token=cfg.hf_token,
        )

    # Optional: run inference test
    test_inference(model, tokenizer, cfg.lora_save_path, cfg)

    print("\nDone!")


if __name__ == "__main__":
    main()
