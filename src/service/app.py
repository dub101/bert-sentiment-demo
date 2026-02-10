import json
import numpy as np
from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


MODEL_DIR = Path("models/distilbert-sst2")
LABELS = {0: "negative", 1: "positive"}
tokenizer = None
model = None
CORPUS_PATH = Path(__file__).resolve().parents[2]  / "data" / "corpus.jsonl"
corpus = []
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = None
corpus_emb = None


class ClassifyRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

app = FastAPI(title="BERT Demo Service", version="0.1.0")


@app.on_event("startup")
def load_classifier():
    global tokenizer, model
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

@app.on_event("startup")
def load_corpus():
    global corpus
    if not CORPUS_PATH.exists():
        raise RuntimeError(f"Corpus file not found: {CORPUS_PATH}")
    corpus = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    global embedder, corpus_emb
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [d["text"] for d in corpus]
    corpus_emb = embedder.encode(texts, normalize_embeddings=True)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/classify")
def classify(req: ClassifyRequest):
    global tokenizer, model
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    return  {
        "text": req.text,
        "label": LABELS.get(pred_id, str(pred_id)),
        "prob": float(probs[pred_id].item()),
        "probs": [float(x) for x in probs.tolist()],
    }

@app.post("/search")
def search(req: SearchRequest):
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
