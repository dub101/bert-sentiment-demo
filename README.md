# BERT Sentiment Demo (DistilBERT)

Beginner-friendly demo:
- Fine-tune a pretrained DistilBERT model on a sentiment dataset
- Save the model + tokenizer
- Run inference via a CLI script
- Dockerized for reproducibility

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -U torch --index-url https://download.pytorch.org/whl/cpu
pip install -U transformers datasets evaluate accelerate scikit-learn tqdm

## Train
```bash
python src/train.py

## Predict
```bash
python src/predict.py

