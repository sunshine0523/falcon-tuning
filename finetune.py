import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def train(
        # model params
        model_name: str = "",
        data_path: str = "",
        output_dir: str = "./output",
        logging_dir: str = "./logs",
        load_4bit: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        save_step: int = 200,
        # train hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        max_steps: int = 80,
        learning_rate: float = 3e-4,
        report: List[str] = None,
        # adapter
        use_lora: bool = True,
        # lora params
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
):
    if lora_target_modules is None:
        lora_target_modules = ["query_key_value"]
    # 梯度累积，减少显存使用
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"

    # 4bit配置
    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=bnb_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=False,
            load_in_8bit=False,
            device_map=device_map,
            trust_remote_code=True
        )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    # adapter配置
    if use_lora:
        adapter_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, adapter_config)
    # 从检查点恢复
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # 打印参数
    def print_trainable_parameters():
        trainable_parameters = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_parameters += param.numel()

        print(
            f"trainable parames: {trainable_parameters} || all_params: {all_param} || trainable % = {100 * trainable_parameters / all_param}"
        )

    print_trainable_parameters()

    # 分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    # 指定填充用结束标记
    tokenizer.pad_token = tokenizer.eos_token

    def generate_prompt(data_point):
        return f"""
<news>: {data_point['news_title']}
<sentiment>: {data_point['emotion']}
""".strip()

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenizer(
            full_prompt,
            padding=True,
            truncation=True
        )
        tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
        tokenized_full_prompt["attention_mask"].append(1)
        return tokenized_full_prompt

    data = load_dataset("json", data_files=data_path)
    data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    # 开始训练
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=save_step,
        output_dir=output_dir,
        logging_dir=logging_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to=report,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=train_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)


if __name__ == '__main__':
    fire.Fire(train)
