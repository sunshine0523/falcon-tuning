import os

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer


def generate(
        # model params
        base_model: str = "",
        adapter_model: str = None,
        load_4bit: bool = False,
):
    device_map = {"": 0}
    # 使用deepspeed进行多显卡配置
    # # 多显卡配置
    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device_map,
            return_dict=True,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=False,
            load_in_8bit=False,
            device_map=device_map,
            trust_remote_code=True
        )
    if adapter_model is not None:
        model = PeftModel.from_pretrained(model, adapter_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 指定填充用结束标记
    tokenizer.pad_token = tokenizer.eos_token

    # 生成配置
    generation_config = model.generation_config
    generation_config.max_new_tokens = 10
    generation_config.temperature = 1
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    def generate_response(question: str) -> str:
        prompt = f"""
<news>: {question}
<sentiment>: 
""".strip()
        encoding = tokenizer(prompt, return_tensors="pt").to(device='cuda')
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding.input_ids,
                generation_config=generation_config,
                attention_mask=encoding.attention_mask,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_start = "<sentiment>:"
        response_start = response.find(assistant_start)
        return response[response_start + len(assistant_start):].strip()

    question = input('Input news. Input exit to exit:')
    while question != 'exit':
        print(generate_response(question))
        question = input("Input news. Input exit to exit:")


if __name__ == '__main__':
    fire.Fire(generate)
