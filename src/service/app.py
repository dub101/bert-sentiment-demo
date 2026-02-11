import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "distilbert-sst2"
DEFAULT_CLASSIFIER_CKPT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
CLASSIFIER_CKPT = os.getenv("CLASSIFIER_CKPT", DEFAULT_CLASSIFIER_CKPT)
CLASSIFIER_MODEL_DIR = os.getenv("CLASSIFIER_MODEL_DIR", "")
LABELS = {0: "negative", 1: "positive"}
CORPUS_PATH = REPO_ROOT / "data" / "corpus.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ClassifyRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_classifier(app)
    init_search(app)
    yield


app = FastAPI(title="BERT Demo Service", version="0.1.0", lifespan=lifespan)


def init_classifier(app: FastAPI) -> None:
    if CLASSIFIER_MODEL_DIR:
        src = Path(CLASSIFIER_MODEL_DIR)
        tokenizer_src = src
        model_src = src
    elif MODEL_DIR.exists():
        tokenizer_src = MODEL_DIR
        model_src = MODEL_DIR
    else:
        tokenizer_src = CLASSIFIER_CKPT
        model_src = CLASSIFIER_CKPT

    app.state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(model_src)
    app.state.model.eval()


def init_search(app: FastAPI) -> None:
    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Corpus file not found: {CORPUS_PATH}")
    corpus = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    app.state.corpus = corpus
    app.state.embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [d["text"] for d in corpus]
    app.state.corpus_emb = app.state.embedder.encode(texts, normalize_embeddings=True)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/classify")
def classify(req: ClassifyRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    return {
        "text": req.text,
        "label": LABELS.get(pred_id, str(pred_id)),
        "prob": float(probs[pred_id].item()),
        "probs": [float(x) for x in probs.tolist()],
    }


@app.post("/search")
def search(req: SearchRequest):
    corpus = app.state.corpus
    embedder = app.state.embedder
    corpus_emb = app.state.corpus_emb
    q = req.query.strip()
    if not q:
        return {"query": req.query, "results": []}

    q_emb = embedder.encode([q], normalize_embeddings=True)[0]  # shape [D]
    scores = corpus_emb @ q_emb  # cosine similarity because both normalized

    top_k = max(1, min(req.top_k, len(corpus)))
    idx = np.argsort(-scores)[:top_k]

    results = []
    for i in idx:
        d = corpus[int(i)]
        results.append({"id": d["id"], "text": d["text"], "score": float(scores[int(i)])})

    return {"query": req.query, "top_k": top_k, "results": results}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}
