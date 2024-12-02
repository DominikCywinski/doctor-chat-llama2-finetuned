import random
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import (
    TrainingArguments,
    pipeline,
    logging,
)
from dataset import load_train_dataset
from model import load_quantized_model, load_tokenizer

train_dataset = load_train_dataset()

model_name = "NousResearch/Llama-2-7b-chat-hf"

model = load_quantized_model(model_name)
tokenizer = load_tokenizer(model_name)

# Define parameter combinations

# lora_r_values = [4, 16, 64]
# lora_alpha_values = [8, 16, 32]
# learning_rate_values = [1e-4, 2e-4, 5e-5]
# num_train_epochs_values = [1, 3, 5]
# max_grad_norm_values = [0.3, 1.0, 5.0]

# Best params below
lora_r_values = [64]
lora_alpha_values = [16]
learning_rate_values = [2e-4]
num_train_epochs_values = [3]
max_grad_norm_values = [0.3]

random_trials_number = 1

# Random selection of parameter combinations (10 random trials)
for _ in range(random_trials_number):
    lora_r = random.choice(lora_r_values)
    lora_alpha = random.choice(lora_alpha_values)
    learning_rate = random.choice(learning_rate_values)
    num_train_epochs = random.choice(num_train_epochs_values)
    max_grad_norm = random.choice(max_grad_norm_values)

    print(
        f"Starting experiment with: learning_rate={learning_rate}, num_train_epochs={num_train_epochs}"
    )

    new_model_name = (
        f"Llama-2-7b-chat-finetune-lr={learning_rate}-epochs={num_train_epochs}"
    )

    # Set LoRA configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=f"results/train_results/{new_model_name}",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=max_grad_norm,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(f"models/{new_model_name}")
    logging.set_verbosity(logging.CRITICAL)

    #
    # # Run text generation pipeline with our next model
    # prompt = "What is a large language model?"
    # pipe = pipeline(
    #     task="text-generation", model=model, tokenizer=tokenizer, max_length=200
    # )
    # result = pipe(f"[INST] {prompt} [/INST]")
    # print(result[0]["generated_text"])
