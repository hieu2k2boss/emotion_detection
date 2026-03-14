# CSKH EmotionAI — Bối cảnh & Mục tiêu

> Tài liệu cập nhật trạng thái dự án, những gì đã có và những gì cần làm tiếp

---

## 1. Tổng quan dự án

Xây dựng hệ thống **nhận diện cảm xúc khách hàng** trong hội thoại CSKH tiếng Việt sử dụng **Agentic RAG**. Hệ thống không chỉ phân loại cảm xúc mà còn hiểu ngữ cảnh đa lượt, phương ngữ, teencode và tự động cảnh báo khi cần escalate.

---

## 2. Những gì đã có (✅ Done)

### 2.1 Data Pipeline
- **Generate data** bằng DeepSeek API (`deepseek-chat`)
- **7 nhãn cảm xúc**: `neutral` | `happy` | `confused` | `anxious` | `frustrated` | `disappointed` | `angry`
- **~140 samples** (20 samples × 7 nhãn) lưu tại `data/*.json`
- Mỗi sample có: `turns`, `label`, `last_utterance`, `context_clues`, `region`, `difficulty`
- Format JSON chuẩn, đã validate

### 2.2 Vector Database
- **Embedding model**: `BAAI/bge-m3` (1024 dims, normalize_embeddings=True)
- **Vector DB**: ChromaDB (cosine similarity, HNSW index)
- **Keyword search**: BM25Okapi (rank_bm25)
- **Hybrid Search**: Vector + BM25 + RRF Fusion (k=60)
- `build_doc_text()`: ghép `last_utterance` + `context_clues` + turn áp cuối

### 2.3 Agentic RAG Pipeline
```
Input → Orchestrator → Tools → Memory → Generation → Output
```

**Tools đã viết:**
- `tool_context_analyzer()` — đếm neg_signals, phát hiện escalation
- `tool_slang_lookup()` — tra teencode + phương ngữ Nam bộ
- `tool_emotion_retriever()` — hybrid search KB

**Orchestrator:**
- Complexity check: `simple` vs `complex`
- Self-reflect: retry nếu `confidence < 0.75`
- Switch model: `deepseek-chat` (simple) → `deepseek-reasoner` (complex)

**WorkingMemory dataclass:**
```python
query, complexity, context_summary, slang_found,
retrieved_docs, prompt, result, attempts
```

### 2.4 SQLite Database
**8 bảng:**
| Bảng | Mô tả |
|---|---|
| `customers` | Khách hàng, tier (normal/silver/gold/vip) |
| `products` | Sản phẩm, giá, stock |
| `orders` | Đơn hàng, status, shipper, ETA |
| `order_items` | Chi tiết sản phẩm trong đơn |
| `payments` | Thanh toán, hoàn tiền, mã giao dịch |
| `tickets` | Mỗi cuộc hội thoại CSKH = 1 ticket |
| `messages` | Từng tin nhắn trong ticket |
| `emotion_logs` | Kết quả Agentic RAG lưu lại |

### 2.5 OrderLookup Tool
- Extract `order_id` từ câu hỏi tự nhiên (regex patterns)
- Extract `phone` từ câu hỏi
- Query SQLite → format response tự nhiên
- Phân biệt: `by_id` / `by_phone` / `not_found` / `need_info`

### 2.6 FastAPI Backend
```
POST /chat          ← Gửi tin nhắn → nhận reply + emotion
GET  /history/{id}  ← Lịch sử hội thoại
GET  /orders/{id}   ← Tra cứu đơn hàng
```
- Tích hợp OrderLookup + Orchestrator RAG
- Lưu toàn bộ messages + emotion_logs vào SQLite
- CORS enabled

### 2.7 Frontend (Mock)
- `client/index.html` — Landing page + chat widget nhúng
- `admin/index.html` — Dashboard 3 tab: Dashboard / Tickets / Agents
- `shared/data.js` — Mock data, EMOTIONS config, localStorage helpers
- Dark theme, Chart.js, Bootstrap 5

---

## 3. Thang cảm xúc

```
😐 neutral       0  — Hỏi thông tin, không cảm xúc
😊 happy         0  — Hài lòng, khen ngợi
🤔 confused      1  — Bối rối, không hiểu
😰 anxious       2  — Lo lắng về tiền/đơn, cần gấp
😤 frustrated    3  — Bực bội vì chờ lâu
😢 disappointed  4  — Thất vọng ngầm, bỏ cuộc  ← NGUY HIỂM
😠 angry         5  — Tức giận, đe dọa          ← CẦN XỬ LÝ NGAY
```

> **Lưu ý:** `disappointed` nguy hiểm hơn `frustrated` vì khách âm thầm rời đi không phàn nàn.

---

## 4. Tech Stack hiện tại

| Layer | Technology | Ghi chú |
|---|---|---|
| LLM API | DeepSeek-chat / DeepSeek-R1 | `https://api.deepseek.com` |
| Embedding | `BAAI/bge-m3` | 1024 dims, ~2GB VRAM |
| Vector DB | ChromaDB | cosine similarity |
| Keyword | BM25Okapi | rank_bm25 |
| Database | SQLite | `cskh.db` |
| Backend | FastAPI + uvicorn | port 8000 |
| Frontend | Bootstrap 5 + Chart.js | HTML/JS thuần |
| Runtime | Local machine | Không dùng Colab |

---

## 5. Những vấn đề đã giải quyết

| Vấn đề | Giải pháp |
|---|---|
| JSON parse lỗi từ DeepSeek | `parse_json_safe()` — thử nhiều cách |
| Model generate tiếng Trung | System prompt + regex filter `[\u4e00-\u9fff]` |
| VRAM OOM khi load model | `del model; gc.collect(); torch.cuda.empty_cache()` |
| Retrieve 0 samples | Fix `build_doc_text()` + kiểm tra raw output |
| IndexError EXPECTED list | Thêm đủ nhãn khi thêm kịch bản |
| PyTorch no CUDA local | Auto-detect: `"cuda" if torch.cuda.is_available() else "cpu"` |
| LLMLingua chưa cài | Dùng sliding_window + summarize_middle thay thế |

---

## 6. Context Engineering đã biết

| Kỹ thuật | Trạng thái | Mô tả |
|---|---|---|
| System Prompt Design | ⚠️ Cơ bản | Cần thêm ràng buộc rõ hơn |
| Few-shot Prompting | ❌ Chưa có | Thêm 5-10 hard cases |
| Chain of Thought | ❌ Chưa có | Cho complex cases |
| Sliding Window | ✅ Có | `turns[-6:]` |
| Summarize Middle | ✅ Có | Dùng DeepSeek tóm tắt |
| Smart Truncate | ✅ Có | Token counting + cắt dần |
| Retrieval Compression | ⚠️ Cơ bản | Chỉ lấy clues liên quan |
| Persona Calibration | ❌ Chưa có | Tune tone theo emotion |
| Instruction Hierarchy | ❌ Chưa có | Priority 1→4 |
| Negative Prompting | ⚠️ Cơ bản | Cần bổ sung |
| Dynamic Context Injection | ❌ Chưa có | VIP, trend, order info |
| LLMLingua | ❌ Chưa cài | Nén token level |
| RRF Fusion | ✅ Có | k=60, Vector + BM25 |
| HyDE | ❌ Chưa có | Hypothetical Document Embedding |
| Query Rewriting | ❌ Chưa có | Viết lại query trước khi search |

---

## 7. Kết quả test hiện tại

```
Test 5 kịch bản cơ bản:   3/5  = 60%
Test 15 kịch bản mở rộng: chưa đo
```

**Kịch bản hay sai nhất:**
- Lịch sự giả: "thôi được rồi cảm ơn 😊" → thường bị đoán `neutral`
- Phương ngữ Nam: "bể hết trơn", "thôi kệ đi quá"
- Leo thang ngầm: không có từ tức giận nhưng nhiều turns tiêu cực

---

## 8. Mong muốn làm tiếp

### 8.1 Tăng accuracy lên 80%+
- [ ] Thêm few-shot examples vào System Prompt (5-10 hard cases)
- [ ] Thêm Chain of Thought cho complex cases
- [ ] Generate thêm data cho nhãn `disappointed` (hay sai nhất)
- [ ] Fine-tune PhoBERT-base-v2 làm classifier nhanh
- [ ] Thêm HyDE + Query Rewriting vào RAG pipeline

### 8.2 Hoàn thiện Context Engineering
- [ ] Instruction Hierarchy (Priority 1→4)
- [ ] Negative Prompting đầy đủ
- [ ] Persona Calibration theo emotion
- [ ] Dynamic Context Injection (VIP, emotion trend, order info)
- [ ] Cài LLMLingua khi hội thoại > 30 turns

### 8.3 Backend / API
- [ ] Authentication (API key hoặc JWT)
- [ ] Rate limiting
- [ ] Async endpoints (FastAPI async)
- [ ] WebSocket cho realtime chat
- [ ] Migrate SQLite → PostgreSQL khi scale

### 8.4 Evaluation
- [ ] F1 / Accuracy per class
- [ ] Confusion matrix 7×7
- [ ] Latency benchmark (p50, p95, p99)
- [ ] So sánh: Naive RAG vs Agentic RAG
- [ ] A/B test: deepseek-chat vs deepseek-reasoner

### 8.5 Frontend
- [ ] Kết nối frontend thật với FastAPI (bỏ mock data)
- [ ] Realtime emotion badge trong chat widget
- [ ] Alert notification khi angry/disappointed
- [ ] Emotion trend chart theo thời gian

---

## 9. Cấu trúc file hiện tại

```
project/
├── chatbot_api/
│   ├── __init__.py
│   ├── main.py          ← FastAPI app
│   ├── db.py            ← SQLite helper
│   ├── rag.py           ← ChromaDB + BM25 + bge-m3
│   ├── orchestrator.py  ← Agentic RAG pipeline
│   ├── chatbot.py       ← Chat handler
│   ├── tools.py         ← OrderLookup tool
│   └── cskh.db          ← SQLite database
├── data/
│   ├── neutral.json
│   ├── happy.json
│   ├── confused.json
│   ├── anxious.json
│   ├── frustrated.json
│   ├── disappointed.json
│   ├── angry.json
│   └── all.json
├── frontend/
│   ├── client/index.html
│   ├── admin/index.html
│   └── shared/data.js
└── docs/
    ├── pipeline_flow.md
    ├── uml_architecture.html
    └── project_context.md   ← file này
```

---

## 10. Câu hỏi / Quyết định còn pending

```
1. Scale lên PostgreSQL hay giữ SQLite?
   → SQLite đủ dùng cho prototype, migrate sau

2. Deploy ở đâu?
   → Local trước, sau xét Railway / Render / VPS

3. Authentication?
   → Chưa cần cho prototype

4. Fine-tune PhoBERT hay dùng LLM API mãi?
   → Fine-tune khi có đủ data (~5K samples)

5. Có cần speculative decoding (EAGLE3)?
   → Khi latency > 3s mới xét
```

---

## 11. Lệnh chạy nhanh

```bash
# Cài dependencies
pip install fastapi uvicorn sentence-transformers chromadb rank_bm25 openai

# Chạy server
uvicorn chatbot_api.main:app --reload --port 8000

# Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "ship lâu vl 5 ngày rồi"}'

# Xem docs
open http://localhost:8000/docs
```

---

*Cập nhật lần cuối: 2025 — Phiên bản 1.0*