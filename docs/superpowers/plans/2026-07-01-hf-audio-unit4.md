# Hugging Face Audio Course Unit 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Colab-first repository for Hugging Face Audio Course Unit 4 that fine-tunes a music genre classifier on GTZAN, provides a Gradio demo, and isolates user-presence steps to Colab execution and optional Hub upload.

**Architecture:** Keep the repository minimal. Put training logic in `src/train.py`, demo logic in `src/app.py`, dependency setup in `requirements-colab.txt`, and use `notebooks/unit4_colab.ipynb` only as a thin Colab runner that installs dependencies and calls the normal Python entry points. Support both local checkpoint paths and Hub model ids for inference so the same demo code works before and after upload.

**Tech Stack:** Python 3, Hugging Face Transformers, Datasets, Evaluate, Accelerate, Gradio, PyTorch, Google Colab, Git

---

## File Structure

- Create: `C:/Users/xcc2/Documents/Unit 4/README.md` - assignment quickstart and repository layout
- Create: `C:/Users/xcc2/Documents/Unit 4/requirements-colab.txt` - exact dependencies for Colab
- Create: `C:/Users/xcc2/Documents/Unit 4/src/train.py` - GTZAN fine-tuning CLI
- Create: `C:/Users/xcc2/Documents/Unit 4/src/app.py` - Gradio demo CLI
- Create: `C:/Users/xcc2/Documents/Unit 4/notebooks/unit4_colab.ipynb` - thin notebook runner for Colab
- Create: `C:/Users/xcc2/Documents/Unit 4/tests/test_smoke.py` - smoke tests for argument/config behavior
- Create: `C:/Users/xcc2/Documents/Unit 4/docs/manual-steps.md` - exact user-touch steps for Colab and Hub

### Task 1: Bootstrap the repository

**Files:**
- Create: `C:/Users/xcc2/Documents/Unit 4/README.md`
- Create: `C:/Users/xcc2/Documents/Unit 4/requirements-colab.txt`
- Create: `C:/Users/xcc2/Documents/Unit 4/docs/manual-steps.md`

- [ ] **Step 1: Write the failing documentation check**

```python
from pathlib import Path

def test_bootstrap_docs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "requirements-colab.txt").exists()
    assert (root / "docs" / "manual-steps.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_bootstrap_docs_exist -v`
Expected: FAIL with `AssertionError` because the files do not exist yet.

- [ ] **Step 3: Write minimal bootstrap files**

```text
# README.md
# Hugging Face Audio Course Unit 4

This repository contains a Colab-first solution for Hugging Face Audio Course Unit 4.

## Files

- `src/train.py` - fine-tuning script for GTZAN
- `src/app.py` - Gradio demo script
- `notebooks/unit4_colab.ipynb` - Colab runner
- `requirements-colab.txt` - Colab dependencies
- `docs/manual-steps.md` - user actions that require presence

## Quickstart

1. Open `notebooks/unit4_colab.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Run the cells to install dependencies and launch training.
4. Use `src/app.py` with the saved checkpoint or Hub model id to launch the demo.
```

```text
# requirements-colab.txt
git+https://github.com/huggingface/transformers
datasets[audio]
evaluate
accelerate
gradio
torch
torchaudio
scikit-learn
soundfile
librosa
ipywidgets
```

```text
# docs/manual-steps.md
# Manual Steps

## Required user actions

1. Open the notebook in Google Colab.
2. Switch the runtime to GPU.
3. If you want Hub upload, run `from huggingface_hub import notebook_login` and authenticate.
4. Run the training command cells.
5. Run the optional `--push-to-hub` command only after authentication.

## Agent-completed work

- repository files
- training script
- demo script
- notebook runner
- smoke tests
```

- [ ] **Step 4: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_bootstrap_docs_exist -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md requirements-colab.txt docs/manual-steps.md tests/test_smoke.py
git commit -m "docs: add unit 4 bootstrap files"
```

### Task 2: Add the training CLI

**Files:**
- Create: `C:/Users/xcc2/Documents/Unit 4/src/train.py`
- Modify: `C:/Users/xcc2/Documents/Unit 4/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
from src.train import build_parser

def test_train_parser_defaults():
    args = build_parser().parse_args([])
    assert args.model_id == "ntu-spml/distilhubert"
    assert args.dataset_id == "marsyas/gtzan"
    assert args.max_duration == 30.0
    assert args.push_to_hub is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_train_parser_defaults -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `src/train.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
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
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Prepared training config for {config.dataset_id} with model {config.model_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Expand `src/train.py` from parser-only to course workflow**

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import evaluate
import numpy as np
from datasets import Audio, load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
)


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
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


def compute_metrics(eval_pred):
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
    config = parse_args(argv)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gtzan = load_dataset(config.dataset_id, "all")
    gtzan = gtzan["train"].train_test_split(
        seed=config.seed,
        shuffle=True,
        test_size=config.test_size,
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
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["test"],
        tokenizer=feature_extractor,
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_train_parser_defaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/train.py tests/test_smoke.py
git commit -m "feat: add unit 4 training cli"
```

### Task 3: Add the Gradio demo CLI

**Files:**
- Create: `C:/Users/xcc2/Documents/Unit 4/src/app.py`
- Modify: `C:/Users/xcc2/Documents/Unit 4/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
from src.app import build_parser

def test_app_parser_defaults():
    args = build_parser().parse_args([])
    assert args.model == "artifacts/distilhubert-gtzan"
    assert args.share is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_app_parser_defaults -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` because `src/app.py` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the GTZAN demo.")
    parser.add_argument("--model", default="artifacts/distilhubert-gtzan")
    parser.add_argument("--share", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(f"Ready to launch demo with {args.model}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Expand `src/app.py` to a real Gradio demo**

```python
from __future__ import annotations

import argparse

import gradio as gr
from transformers import pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the GTZAN demo.")
    parser.add_argument("--model", default="artifacts/distilhubert-gtzan")
    parser.add_argument("--share", action="store_true")
    return parser


def build_classifier(model_name: str):
    return pipeline("audio-classification", model=model_name)


def classify_audio(audio_path: str, classifier):
    result = classifier(audio_path)
    top = result[0]
    return f"{top['label']} ({top['score']:.3f})", {item["label"]: float(item["score"]) for item in result}


def main() -> None:
    args = build_parser().parse_args()
    classifier = build_classifier(args.model)

    def predict(audio_path):
        return classify_audio(audio_path, classifier)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Audio(type="filepath", label="Upload a song clip"),
        outputs=[
            gr.Textbox(label="Top prediction"),
            gr.Label(label="All scores"),
        ],
        title="GTZAN Genre Classifier",
        description="DistilHuBERT fine-tuned on marsyas/gtzan.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_app_parser_defaults -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/app.py tests/test_smoke.py
git commit -m "feat: add gradio demo cli"
```

### Task 4: Add the Colab runner and smoke coverage

**Files:**
- Create: `C:/Users/xcc2/Documents/Unit 4/notebooks/unit4_colab.ipynb`
- Modify: `C:/Users/xcc2/Documents/Unit 4/tests/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
from pathlib import Path

def test_notebook_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "notebooks" / "unit4_colab.ipynb").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `py -3.11 -m pytest tests/test_smoke.py::test_notebook_exists -v`
Expected: FAIL with `AssertionError` because the notebook does not exist yet.

- [ ] **Step 3: Write the notebook and smoke test module**

```python
# tests/test_smoke.py
from pathlib import Path

from src.app import build_parser as build_app_parser
from src.train import build_parser as build_train_parser


def test_bootstrap_docs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "requirements-colab.txt").exists()
    assert (root / "docs" / "manual-steps.md").exists()


def test_train_parser_defaults():
    args = build_train_parser().parse_args([])
    assert args.model_id == "ntu-spml/distilhubert"
    assert args.dataset_id == "marsyas/gtzan"
    assert args.max_duration == 30.0
    assert args.push_to_hub is False


def test_app_parser_defaults():
    args = build_app_parser().parse_args([])
    assert args.model == "artifacts/distilhubert-gtzan"
    assert args.share is False


def test_notebook_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "notebooks" / "unit4_colab.ipynb").exists()
```

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Hugging Face Audio Course Unit 4\n",
        "\n",
        "Run this notebook in Google Colab with a GPU runtime.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!nvidia-smi\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!pip install -q -r requirements-colab.txt\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "!python src/train.py --output-dir artifacts/distilhubert-gtzan\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Optional after login:\n",
        "# !python src/train.py --output-dir artifacts/distilhubert-gtzan --push-to-hub --hub-model-id your-username/distilhubert-finetuned-gtzan\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Optional demo:\n",
        "# !python src/app.py --model artifacts/distilhubert-gtzan\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 4: Run the smoke suite**

Run: `py -3.11 -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Run syntax verification**

Run: `py -3.11 -m py_compile src/train.py src/app.py tests/test_smoke.py`
Expected: no output

- [ ] **Step 6: Commit**

```bash
git add notebooks/unit4_colab.ipynb tests/test_smoke.py
git commit -m "test: add smoke coverage and colab runner"
```

### Task 5: Final repo verification and handoff

**Files:**
- Modify: `C:/Users/xcc2/Documents/Unit 4/README.md`
- Modify: `C:/Users/xcc2/Documents/Unit 4/docs/manual-steps.md`

- [ ] **Step 1: Add the final quickstart details**

```text
# README.md
# Hugging Face Audio Course Unit 4

This repository contains a Colab-first solution for Hugging Face Audio Course Unit 4.

## Setup

1. Push this repository to a Git remote.
2. Open `notebooks/unit4_colab.ipynb` in Google Colab.
3. Enable a GPU runtime.
4. Install dependencies from `requirements-colab.txt`.
5. Run `python src/train.py --output-dir artifacts/distilhubert-gtzan`.

## Optional Hub upload

Authenticate in Colab and run:

`python src/train.py --output-dir artifacts/distilhubert-gtzan --push-to-hub --hub-model-id your-username/distilhubert-finetuned-gtzan`

## Demo

Run:

`python src/app.py --model artifacts/distilhubert-gtzan`
```

```text
# docs/manual-steps.md
# Manual Steps

## Steps that require the user

1. Publish the repository to a Git remote that Colab can access.
2. Open the notebook in Google Colab.
3. Switch the runtime to GPU.
4. If Hub upload is desired, authenticate with `notebook_login`.
5. Run the training cell.
6. Run the optional Hub upload command.

## Steps completed by the agent

- repository structure
- training CLI
- demo CLI
- Colab notebook
- smoke tests
- local verification
```

- [ ] **Step 2: Run the full local verification set**

Run: `py -3.11 -m pytest tests/test_smoke.py -v`
Expected: PASS

Run: `py -3.11 -m py_compile src/train.py src/app.py tests/test_smoke.py`
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add README.md docs/manual-steps.md
git commit -m "docs: finalize unit 4 handoff"
```

## Self-Review

- Spec coverage: Task 1 covers repository bootstrap and docs, Task 2 covers GTZAN fine-tuning, Task 3 covers the Gradio demo, Task 4 covers the Colab runner and smoke checks, and Task 5 covers the final user-touch handoff. No spec gaps remain.
- Placeholder scan: no `TODO`, `TBD`, or vague "add validation later" language remains in the executable steps.
- Type consistency: `build_parser`, `parse_args`, `TrainConfig`, `--push-to-hub`, `--hub-model-id`, and the artifact path `artifacts/distilhubert-gtzan` are named consistently across all tasks.
