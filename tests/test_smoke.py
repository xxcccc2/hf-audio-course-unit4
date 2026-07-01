from pathlib import Path


def test_bootstrap_docs_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    assert (root / "requirements-colab.txt").exists()
    assert (root / "docs" / "manual-steps.md").exists()
