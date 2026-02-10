import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "negative", 1: "positive"}


def load_model(model_dir: str):
    model_path = Path(model_dir)
    if not model_path.exists():
        raise SystemExit(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


def predict_one(tokenizer, model, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)

    probs = F.softmax(logits, dim=-1)
    pred_id = int(torch.argmax(probs).item())
    pred_label = LABELS.get(pred_id, str(pred_id))
    pred_prob = float(probs[pred_id].item())
    return pred_label, pred_prob, probs.tolist(), logits.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment prediction using a fine-tuned DistilBERT model.")
    parser.add_argument("text", type=str, help="Input text to classify (wrap in quotes).")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/distilbert-sst2",
        help="Path to saved model folder (default: models/distilbert-sst2).",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    label, prob, probs, logits = predict_one(tokenizer, model, args.text)

    print(f"text: {args.text}")
    print(f"prediction: {label} (prob={prob:.4f})")
    print(f"probs: {probs}")
    # Uncomment if you want raw logits too
    # print(f"logits: {logits}")


if __name__ == "__main__":
    main()
