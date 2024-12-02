import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_quantized_model(model_name: str = "NousResearch/Llama-2-7b-chat-hf"):
    # Set bitsandbytes config
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    # Load base model
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model


def load_tokenizer(model_name: str = "NousResearch/Llama-2-7b-chat-hf"):
    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    return tokenizer


def load_model_with_peft(
    base_model, peft_model_path: str = "models/Llama-2-7b-chat-finetune"
):
    model = PeftModel.from_pretrained(base_model, peft_model_path)

    return model
