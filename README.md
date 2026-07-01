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
5. Run `python src/train.py --output-dir artifacts/distilhubert-gtzan`.

## Optional Hub upload

Authenticate in Colab and run:

`python src/train.py --output-dir artifacts/distilhubert-gtzan --push-to-hub --hub-model-id your-username/distilhubert-finetuned-gtzan`

## Demo

Run:

`python src/app.py --model artifacts/distilhubert-gtzan`
