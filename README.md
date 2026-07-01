# Hugging Face Audio Course Unit 4

This repository contains a Colab-first solution for Hugging Face Audio Course Unit 4.

## Files

- `src/train.py` - fine-tuning script for GTZAN
- `src/app.py` - Gradio demo script
- `notebooks/unit4_colab.ipynb` - Colab runner
- `requirements-colab.txt` - Colab dependencies
- `docs/manual-steps.md` - user actions that require presence

## Setup

1. Push this repository to a Git remote that Google Colab can access.
2. Open `notebooks/unit4_colab.ipynb` in Google Colab.
3. Replace the repository URL placeholder in the notebook.
4. Enable a GPU runtime.
5. Run the Unit 4 certificate-oriented recipe:

`python src/train.py --model-id MIT/ast-finetuned-audioset-10-10-0.4593 --output-dir artifacts/ast-gtzan --per-device-train-batch-size 4 --per-device-eval-batch-size 4 --learning-rate 5e-5 --num-train-epochs 20 --seed 42`

## Optional Hub upload

Authenticate in Colab and run:

`python src/train.py --model-id MIT/ast-finetuned-audioset-10-10-0.4593 --output-dir artifacts/ast-gtzan --per-device-train-batch-size 4 --per-device-eval-batch-size 4 --learning-rate 5e-5 --num-train-epochs 20 --seed 42 --push-to-hub --hub-model-id your-username/ast-finetuned-gtzan`

## Demo

Run:

`python src/app.py --model artifacts/ast-gtzan`
