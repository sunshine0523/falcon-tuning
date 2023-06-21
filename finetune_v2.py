import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import transformers
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelAndDataArguments:
    model_name_or_path: str = field(default=None)
    dataset: str = field(default="")
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Resume training from checkpoint"}
    )


@dataclass
class TrainingArguments:
    output_dir: Optional[str] = field(default="output")
    logging_dir: Optional[str] = field(default="logs")
    load_bit: Optional[int] = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    use_lora: Optional[bool] = field(default=True)
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['query_key_value']
    )
    learning_rate: float = field(
        default=0.0002,
        metadata={"help": 'The learning rate'}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": 'The training batch size per GPU. Increase for better speed.'}
    )
    # 减少内存耗费，会增加训练时间
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'}
    )
    max_steps: int = field(
        default=10000,
        metadata={"help": 'How many optimizer update steps to take'}
    )
    save_step: Optional[int] = field(default=500)
    logging_step: Optional[int] = field(default=1)
    report_to: str = field(
        default=None,
        metadata={"help": "To use wandb or something else for reporting."}
    )


def get_last_checkpoint(resume_from_checkpoint):
    if resume_from_checkpoint is None:
        return None
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        return adapters_weights
    else:
        print(f"Checkpoint {checkpoint_name} not found")
        return None


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all_params: {all_param} || trainable % = {100 * trainable_parameters / all_param}"
    )


def generate_prompt(data_point):
    return f"""
<news>: {data_point['news_title']}
<sentiment>: {data_point['emotion']}
""".strip()


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )
    # 添加结束标记
    tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
    tokenized_full_prompt["attention_mask"].append(1)
    return tokenized_full_prompt


def get_model(args):
    # 多显卡配置
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # DistributedDataParallel
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        args.gradient_accumulation_steps = args.gradient_accumulation_steps // world_size

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_bit == 4,
        load_in_8bit=args.load_bit == 8,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    # 启用梯度检查点，不存储中间激活，用计算来换取内存
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA
    if args.use_lora:
        adapter_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, adapter_config)

    # 从检查点加载
    checkpoint = get_last_checkpoint(args.resume_from_checkpoint)
    if checkpoint is not None:
        model = set_peft_model_state_dict(model, checkpoint)

    print_trainable_parameters(model)

    return model


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    # 指定填充用结束标记
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_dataset(args, tokenizer):
    dataset = load_dataset("json", data_files=args.dataset)
    dataset = dataset["train"].shuffle().map(lambda x: generate_and_tokenize_prompt(x, tokenizer))
    return dataset


def train():
    hf_parser = transformers.HfArgumentParser((
        ModelAndDataArguments, TrainingArguments
    ))
    model_and_data_args, training_args = \
        hf_parser.parse_args_into_dataclasses()
    args = argparse.Namespace(
        **vars(model_and_data_args), **vars(training_args)
    )

    model = get_model(args)
    tokenizer = get_tokenizer(args)
    dataset = get_dataset(args, tokenizer)

    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=args.logging_step,
        save_steps=args.save_step,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to=args.report_to,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    train()
