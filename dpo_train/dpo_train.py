import os
from typing import Dict

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import DPOTrainer

from config import ScriptArguments


def load_dpo_jsonl_dataset(train_file: str, eval_file: str):
    data_files = {"train": train_file}
    if eval_file:
        data_files["eval"] = eval_file
    dataset = load_dataset("json", data_files=data_files)

    def rename_reject(example: Dict):
        # Some generators produce 'reject' vs 'rejected'; normalize to 'rejected'
        if "rejected" not in example and "reject" in example:
            example["rejected"] = example["reject"]
        return example

    dataset = dataset.map(rename_reject)
    # TRL expects 'prompt', 'chosen', 'rejected'
    required_cols = {"prompt", "chosen", "rejected"}
    for split, ds in dataset.items():
        missing = required_cols.difference(set(ds.column_names))
        if missing:
            raise ValueError(f"Missing required columns {missing} in split '{split}'. Got {ds.column_names}")
    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model_name_or_path = script_args.model_name_or_path

    # Resolve HF token from args or environment
    hf_token = script_args.hf_token or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=script_args.trust_remote_code,
        token=hf_token or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=script_args.trust_remote_code,
        token=hf_token or None,
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    dataset = load_dpo_jsonl_dataset(script_args.train_file, script_args.eval_file)

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_train",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",  # Changed from out_proj
            "gate_proj",  # Added for Gemma-2
            "up_proj",    # Added for Gemma-2
            "down_proj",  # Added for Gemma-2
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,  # older TRL uses TrainingArguments-like config; no beta here
        train_dataset=dataset["train"],
        # Older transformers may not support eval during training without evaluation_strategy
        # If needed, you can run a separate eval after training.
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)