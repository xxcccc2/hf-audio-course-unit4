from pathlib import Path


def test_bootstrap_docs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "requirements-colab.txt").exists()
    assert (root / "docs" / "manual-steps.md").exists()


from src.train import build_parser


def test_train_parser_defaults():
    args = build_parser().parse_args([])
    assert args.model_id == "ntu-spml/distilhubert"
    assert args.dataset_id == "marsyas/gtzan"
    assert args.max_duration == 30.0
    assert args.push_to_hub is False


from src.app import build_parser as build_app_parser


def test_app_parser_defaults():
    args = build_app_parser().parse_args([])
    assert args.model == "artifacts/distilhubert-gtzan"
    assert args.share is False


def test_notebook_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "notebooks" / "unit4_colab.ipynb").exists()
