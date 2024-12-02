# python src/evaluate.py <'base_model' or provide adapter_model_path> <'cv' or 'test' dataset>

import sys
import os
import torch
import re
from peft import PeftModel, PeftConfig
from trl import SFTTrainer
from rouge_score import rouge_scorer
from transformers import (
    TrainingArguments,
    pipeline,
    logging,
)
from dataset import load_cv_dataset, load_test_dataset
from model import load_quantized_model, load_tokenizer, load_model_with_peft


def extract_prompt_and_response(text):
    prompt_pattern = r"<s>(\[INST\].*?\[/INST\])"
    response_pattern = r"\[/INST\](.*?)</s>"

    prompt = re.search(prompt_pattern, text, re.DOTALL)
    response = re.search(response_pattern, text, re.DOTALL)

    if prompt:
        prompt = prompt.group(1).strip()
    if response:
        response = response.group(1).strip()

    return prompt, response


def calculate_rouge(scorer, generated, reference):
    return scorer.score(reference, generated)


def model_eval(model, tokenizer, dataset, model_name, peft_config=None):
    # Set evaluation parameters
    eval_arguments = TrainingArguments(
        output_dir=f"results/eval_results/{model_name}_{arguments[1]}_dataset",
        per_device_eval_batch_size=4,
        logging_steps=10,
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        max_steps=-1,
        do_train=False,
        do_eval=True,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        eval_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=eval_arguments,
        packing=False,
    )

    # Evaluate
    model.eval()
    torch.no_grad()

    evaluation_results = trainer.evaluate()
    print("Evaluation results:", evaluation_results)


def eval_rouge(model, tokenizer, dataset):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for i, example in enumerate(dataset.select(range(10))):
        print(f"Processing example {i}")
        prompt, reference_response = extract_prompt_and_response(example["text"])

        pipe = pipeline(
            task="text-generation", model=model, tokenizer=tokenizer, max_length=200
        )
        result = pipe(prompt)

        generated_text = result[0]["generated_text"]
        response = generated_text.split("[/INST]", 1)[-1].strip()

        scores = calculate_rouge(scorer, response, reference_response)

        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
        print(
            f"Prompt: {prompt}\nReference Response: {reference_response}\nGenerated Response: {response}"
        )
        print(
            f"ROUGE-1: {scores['rouge1'].fmeasure}\nROUGE-2: {scores['rouge2'].fmeasure}\nROUGE-L: {scores['rougeL'].fmeasure}"
        )

    mean_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
    mean_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
    mean_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])

    print(f"Mean ROUGE-1: {mean_rouge1}")
    print(f"Mean ROUGE-2: {mean_rouge2}")
    print(f"Mean ROUGE-L: {mean_rougeL}")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    dataset = load_cv_dataset() if arguments[1] == "cv" else load_test_dataset()
    base_model = load_quantized_model()

    if arguments[0] != "base":
        adapter_model_path = arguments[0]
        model_name = os.path.basename(adapter_model_path)
        model = load_model_with_peft(base_model, adapter_model_path)
        peft_config = PeftConfig.from_pretrained(adapter_model_path)
        tokenizer = load_tokenizer(peft_config.base_model_name_or_path)
        model_eval(model, tokenizer, dataset, model_name, peft_config)
    else:
        model_name = "base"
        model = base_model
        tokenizer = load_tokenizer()
        model_eval(model, tokenizer, dataset, model_name)

    eval_rouge(model, tokenizer, dataset)
