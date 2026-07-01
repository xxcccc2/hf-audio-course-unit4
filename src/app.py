from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the GTZAN demo.")
    parser.add_argument("--model", default="artifacts/distilhubert-gtzan")
    parser.add_argument("--share", action="store_true")
    return parser


def build_classifier(model_name: str):
    from transformers import pipeline

    return pipeline("audio-classification", model=model_name)


def classify_audio(audio_path: str, classifier):
    result = classifier(audio_path)
    top = result[0]
    return f"{top['label']} ({top['score']:.3f})", {
        item["label"]: float(item["score"]) for item in result
    }


def main() -> None:
    import gradio as gr

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
