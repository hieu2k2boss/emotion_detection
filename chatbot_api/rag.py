import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb, json
from pathlib import Path

# ── Load models ──────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[RAG] device={device}")

embed_model = SentenceTransformer("BAAI/bge-m3", device=device)

chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection("emotion_kb")
    print(f"[RAG] Loaded existing collection: {collection.count()} docs")
except:
    collection = chroma_client.create_collection(
        "emotion_kb", metadata={"hnsw:space": "cosine"}
    )
    print("[RAG] Created new collection")

# BM25 state
_docs:      list = []
_metas:     list = []
_bm25:      BM25Okapi | None = None

def build_doc_text(sample: dict) -> str:
    parts = [sample["last_utterance"]]
    clues = sample.get("context_clues", [])
    if clues:
        parts.append(" ".join(clues))
    turns     = sample.get("turns", [])
    cust_msgs = [t["text"] for t in turns if t["role"] == "customer"]
    if len(cust_msgs) >= 2:
        parts.append(cust_msgs[-2])
    return " | ".join(parts)

def load_kb(data_dir: str = "data"):
    """Load data/*.json → embed → index ChromaDB + BM25"""
    global _docs, _metas, _bm25

    all_data = []
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name == "all.json":
            continue
        with open(path, encoding="utf-8") as f:
            all_data.extend(json.load(f))

    if not all_data:
        print("[RAG] ⚠️  Không có data, dùng KB mặc định")
        all_data = _default_kb()

    _docs, _metas, ids = [], [], []
    for i, sample in enumerate(all_data):
        doc_text = build_doc_text(sample)
        conv_text = " → ".join([f"{t['role']}: {t['text']}" for t in sample["turns"]])
        _docs.append(doc_text)
        _metas.append({
            "label":          sample["label"],
            "last_utterance": sample["last_utterance"],
            "region":         sample.get("region", "chung"),
            "difficulty":     sample.get("difficulty", "medium"),
            "context_clues":  ", ".join(sample.get("context_clues", [])),
            "full_conv":      conv_text[:400],
        })
        ids.append(f"doc_{i:04d}")

    # Embed + index ChromaDB
    embeddings = embed_model.encode(
        _docs, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    # Reset collection
    global collection
    try: chroma_client.delete_collection("emotion_kb")
    except: pass
    collection = chroma_client.create_collection(
        "emotion_kb", metadata={"hnsw:space": "cosine"}
    )
    collection.add(embeddings=embeddings, documents=_docs, metadatas=_metas, ids=ids)

    # BM25
    _bm25 = BM25Okapi([d.lower().split() for d in _docs])

    print(f"[RAG] ✅ Indexed {len(_docs)} docs")

def hybrid_search(query: str, top_k: int = 3) -> list:
    if not _docs or _bm25 is None:
        return []

    RRF_K = 60
    q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
    v_res  = collection.query(query_embeddings=q_emb, n_results=top_k * 2)
    v_ids  = v_res["ids"][0]
    v_dist = v_res["distances"][0]

    bm25_scores = _bm25.get_scores(query.lower().split())
    bm25_top    = sorted(range(len(_docs)), key=lambda i: bm25_scores[i], reverse=True)[:top_k*2]

    rrf = {}
    for rank, doc_id in enumerate(v_ids):
        idx = int(doc_id.split("_")[1])
        rrf[idx] = rrf.get(idx, 0) + 1 / (RRF_K + rank)
    for rank, idx in enumerate(bm25_top):
        rrf[idx] = rrf.get(idx, 0) + 1 / (RRF_K + rank)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
    results = []
    for idx in top_ids:
        v_score = None
        if f"doc_{idx:04d}" in v_ids:
            pos = v_ids.index(f"doc_{idx:04d}")
            v_score = round(1 - v_dist[pos], 3)
        results.append({
            "doc_id":         f"doc_{idx:04d}",
            "label":          _metas[idx]["label"],
            "last_utterance": _metas[idx]["last_utterance"],
            "context_clues":  _metas[idx]["context_clues"],
            "rrf_score":      round(rrf[idx], 4),
            "vector_score":   v_score,
        })
    return results

def _default_kb() -> list:
    """KB tối thiểu khi chưa có data folder"""
    return [
        {"turns":[{"role":"customer","text":"ừ thôi kệ đi"}],
         "label":"disappointed","last_utterance":"ừ thôi kệ đi",
         "context_clues":["bỏ cuộc ngầm"],"region":"chung","difficulty":"hard"},
        {"turns":[{"role":"customer","text":"ship lâu vl 5 ngày rồi"}],
         "label":"frustrated","last_utterance":"ship lâu vl 5 ngày rồi",
         "context_clues":["chờ lâu","teencode vl"],"region":"chung","difficulty":"easy"},
        {"turns":[{"role":"customer","text":"tiền bị trừ mà không thấy đơn"}],
         "label":"anxious","last_utterance":"tiền bị trừ mà không thấy đơn",
         "context_clues":["lo mất tiền","cần gấp"],"region":"chung","difficulty":"medium"},
        {"turns":[{"role":"customer","text":"ship nhanh vcl cảm ơn shop"}],
         "label":"happy","last_utterance":"ship nhanh vcl cảm ơn shop",
         "context_clues":["khen ngợi","hài lòng"],"region":"chung","difficulty":"easy"},
        {"turns":[{"role":"customer","text":"tôi sẽ post lên mạng cho mọi người biết"}],
         "label":"angry","last_utterance":"tôi sẽ post lên mạng",
         "context_clues":["đe dọa","khiếu nại"],"region":"chung","difficulty":"easy"},
        {"turns":[{"role":"customer","text":"lần đầu mà cũng lần cuối luôn"}],
         "label":"angry","last_utterance":"lần đầu mà cũng lần cuối luôn",
         "context_clues":["tuyên bố rời đi","tức giận"],"region":"chung","difficulty":"medium"},
        {"turns":[{"role":"customer","text":"thôi kệ đi lần sau chắc không mua nữa"}],
         "label":"disappointed","last_utterance":"thôi kệ đi lần sau chắc không mua nữa",
         "context_clues":["phương ngữ Nam","bỏ cuộc"],"region":"nam","difficulty":"hard"},
    ]

def compress_retrieved(query: str, docs: list) -> str:
    """
    Với mỗi doc, chỉ giữ phần thực sự liên quan đến query
    """
    compressed = []
    for doc in docs:
        # Chỉ lấy context_clues liên quan
        clues = doc["context_clues"].split(", ")
        relevant_clues = [
            c for c in clues
            if any(word in query.lower() for word in c.lower().split())
        ]
        compressed.append(
            f"- \"{doc['last_utterance']}\" → {doc['label']}"
            + (f" [{', '.join(relevant_clues)}]" if relevant_clues else "")
        )
    return "\n".join(compressed)