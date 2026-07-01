from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    model_id: str
    dataset_id: str
    output_dir: str
    max_duration: float
    test_size: float
    seed: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    warmup_ratio: float
    push_to_hub: bool
    hub_model_id: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune DistilHuBERT on GTZAN.")
    parser.add_argument("--model-id", default="ntu-spml/distilhubert")
    parser.add_argument("--dataset-id", default="marsyas/gtzan")
    parser.add_argument("--output-dir", default="artifacts/distilhubert-gtzan")
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    return parser


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    args = build_parser().parse_args(argv)
    return TrainConfig(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        max_duration=args.max_duration,
        test_size=args.test_size,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


def compute_metrics(eval_pred):
    import evaluate
    import numpy as np

    metric = evaluate.load("accuracy")
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def preprocess_builder(feature_extractor, max_duration: float):
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        return feature_extractor(
            audio_arrays,
            sampling_rate=feature_extractor.sampling_rate,
            max_length=int(feature_extractor.sampling_rate * max_duration),
            truncation=True,
            return_attention_mask=True,
        )

    return preprocess_function


def main(argv: list[str] | None = None) -> None:
    import torch
    from datasets import Audio, DatasetDict, load_dataset
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        Trainer,
        TrainingArguments,
    )

    config = parse_args(argv)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_train = load_dataset(config.dataset_id, "all", trust_remote_code=True)["train"]
    indices = list(range(len(full_train)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        shuffle=True,
        stratify=full_train["genre"],
    )
    gtzan = DatasetDict(
        {
            "train": full_train.select(train_indices),
            "test": full_train.select(test_indices),
        }
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config.model_id,
        do_normalize=True,
        return_attention_mask=True,
    )
    sampling_rate = feature_extractor.sampling_rate
    gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

    preprocess_function = preprocess_builder(feature_extractor, config.max_duration)
    encoded = gtzan.map(
        preprocess_function,
        remove_columns=["audio", "file"],
        batched=True,
        batch_size=100,
        num_proc=1,
    )
    encoded = encoded.rename_column("genre", "label")

    labels = gtzan["train"].features["genre"].names
    id2label = {str(i): label for i, label in enumerate(labels)}
    label2id = {label: str(i) for i, label in enumerate(labels)}

    model = AutoModelForAudioClassification.from_pretrained(
        config.model_id,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        push_to_hub=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model()
    feature_extractor.save_pretrained(output_dir)

    if config.push_to_hub:
        kwargs = {
            "dataset_tags": config.dataset_id,
            "dataset": "GTZAN",
            "model_name": config.hub_model_id or output_dir.name,
            "finetuned_from": config.model_id,
            "tasks": "audio-classification",
        }
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
