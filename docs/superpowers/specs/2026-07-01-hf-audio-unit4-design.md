# Hugging Face Audio Course Unit 4 Design

## Context

We need to complete Hugging Face Audio Course Unit 4, "Build a music genre classifier", in a repository that is currently empty. The user wants progress carried as far as possible without their presence, and wants the points that require them to be clearly isolated. The course flow covers fine-tuning a music genre classifier on GTZAN, then building a Gradio demo, with optional Hub upload at the end.

The local machine does not currently have a working training stack in the Python environments we inspected: `torch`, `datasets`, `evaluate`, `gradio`, and `accelerate` are missing. Because the course material explicitly targets a free Google Colab T4 GPU and the user agreed to it, the execution path should be Colab-first.

## Goals

- Reproduce the Unit 4 workflow in a way that matches the course requirements.
- Keep the project minimal: one training entry point, one inference/demo entry point, one Colab notebook, and lightweight docs.
- Use the local Git repository as the source of truth for code and notebook files.
- Carry the work all the way to the steps that require the user to be present, such as opening Colab and authenticating for Hugging Face Hub upload.

## Non-Goals

- Building a larger app, package, or reusable framework around the assignment.
- Supporting multiple training backends beyond the agreed Colab-first path.
- Adding custom dataset abstractions, experiment tracking systems, or deployment automation.

## Recommended Approach

Use a Colab-first repository layout where the notebook is a thin runner around normal Python scripts kept in Git. This keeps the training logic in regular files instead of burying it inside notebook cells, avoids duplicating logic across local and hosted environments, and makes it easy to run everything except the final manual steps from versioned code.

The repository will contain:

- a training script that fine-tunes `ntu-spml/distilhubert` on `marsyas/gtzan`
- a demo script that loads a saved or Hub-hosted checkpoint and exposes Gradio audio classification
- a Colab notebook that installs the needed packages, clones or opens the repo, and runs the training command
- short docs that explain the required manual steps for Colab launch and Hub upload

## Data And Model Flow

The training script will follow the course structure closely:

1. Load `marsyas/gtzan` from the Hugging Face Hub.
2. Create the course-style train/test split.
3. Load `AutoFeatureExtractor` for `ntu-spml/distilhubert`.
4. Resample audio to the extractor sampling rate.
5. Preprocess batches into `input_values` and `attention_mask`.
6. Rename `genre` to `label`.
7. Fine-tune with `Trainer` using accuracy as the main evaluation metric.
8. Save the best checkpoint locally.
9. Optionally push to the Hub only when the user is present and authenticated.

The demo script will load the final checkpoint and expose a single-file audio classification interface with Gradio. It will accept either a local checkpoint path or a Hub model id so the same script works before and after upload.

## Manual-Touch Boundaries

The following actions require the user:

- opening Google Colab and running the notebook in a GPU runtime
- authenticating in Colab if Hugging Face Hub upload is desired
- confirming any final repository push destination if remote Git publishing is needed

Everything else should be prepared by the agent: repository structure, scripts, notebook, dependency files, and instructions.

## Files

Planned minimal file set:

- `README.md` - quickstart for the assignment
- `requirements-colab.txt` - Colab dependency install list
- `src/train.py` - fine-tuning entry point
- `src/app.py` - Gradio demo entry point
- `notebooks/unit4_colab.ipynb` - thin Colab runner
- `tests/test_smoke.py` - one small smoke-style check for CLI argument wiring or config construction
- `docs/manual-steps.md` - explicit user-presence steps

## Error Handling And Validation

The code should fail early and plainly when:

- required dependencies are missing
- the model path is not available
- the user asks for Hub upload without authentication

The implementation should keep checks lightweight. We do not need a heavy validation layer; small argument validation and clear runtime errors are enough for this assignment.

## Testing Strategy

Follow the smallest useful check rule:

- add one smoke test that exercises non-trivial setup logic without requiring full model training
- run a syntax/import pass on the created scripts
- if practical, run a lightweight dry-run path that builds dataset and config objects without launching full training

We do not need a full test suite for a course assignment repository.

## Success Criteria

The assignment is considered ready when:

- the repository contains all code and notebook artifacts needed for Unit 4
- the training flow matches the course requirements for GTZAN fine-tuning
- the Gradio demo path is implemented
- the manual user steps are isolated to Colab execution and optional Hub upload
- we have verified the local artifacts as far as the current machine allows
