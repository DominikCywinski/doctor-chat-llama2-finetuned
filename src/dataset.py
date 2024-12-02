from datasets import load_dataset


dataset_name = "avaliev/chat_doctor"


def transform_to_llama_format(example):
    input_text = example["input"]
    response_text = example["output"]
    transformed_text = (
        f"<s>[INST] {input_text.strip()} [/INST] {response_text.strip()} </s>"
    )

    return {"text": transformed_text}


def load_train_dataset():
    train_dataset = (
        load_dataset(dataset_name, split="train").shuffle(seed=42).select(range(3500))
    )
    train_dataset_transformed = train_dataset.map(
        transform_to_llama_format, remove_columns=["input", "instruction", "output"]
    )

    return train_dataset_transformed


def load_cv_dataset():
    cv_dataset = (
        load_dataset(dataset_name, split="validation")
        .shuffle(seed=42)
        .select(range(750))
    )
    cv_dataset_transformed = cv_dataset.map(
        transform_to_llama_format, remove_columns=["input", "instruction", "output"]
    )

    return cv_dataset_transformed


def load_test_dataset():
    test_dataset = (
        load_dataset(dataset_name, split="test").shuffle(seed=42).select(range(750))
    )
    test_dataset_transformed = test_dataset.map(
        transform_to_llama_format, remove_columns=["input", "instruction", "output"]
    )

    return test_dataset_transformed


# dataset_name = "mlabonne/guanaco-llama2"

#     def load_train_dataset():
#     train_dataset = load_dataset(dataset_name, split="train")
#     train_cv_split = train_dataset.train_test_split(test_size=0.1, seed=42)
#     train_dataset = train_cv_split["train"]
#     cv_dataset = train_cv_split["test"]
#
#     return train_dataset, cv_dataset
#
#
# def load_test_dataset():
#     test_dataset = load_dataset(dataset_name, split="test")
#
#     return test_dataset
