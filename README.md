# EmotionAI CSKH — Thiết kế hệ thống Lambda Architecture

> **Phiên bản:** 2.0 — Lambda Architecture  
> **Giai đoạn:** Prototype / Research  
> **Cập nhật:** 2026

---

## 1. Tổng quan hệ thống

Hệ thống nhận diện cảm xúc khách hàng trong hội thoại CSKH tiếng Việt, được thiết kế theo **Lambda Architecture** với 3 layer riêng biệt. Mục tiêu là vừa xử lý **realtime** từng tin nhắn, vừa cung cấp **thống kê chính xác** từ batch processing, và **phục vụ API** bằng cách merge cả hai nguồn.

```
Vấn đề giải quyết:
  ├── Phân tích cảm xúc đa lượt tiếng Việt (teencode, phương ngữ)
  ├── Tra cứu đơn hàng tức thì qua hội thoại tự nhiên
  ├── Cảnh báo realtime khi khách có nguy cơ rời đi
  └── Dashboard thống kê chính xác cho admin
```

---

## 2. Lambda Architecture — 3 Layer

```
                     ┌──────────────────────────────────┐
  Raw Events         │         LAMBDA ARCHITECTURE       │
  (messages)  ──▶   │                                   │
                     │  ┌────────────┐  ┌─────────────┐ │
                     │  │BATCH LAYER │  │SPEED LAYER  │ │
                     │  │(nightly)   │  │(realtime)   │ │
                     │  └─────┬──────┘  └──────┬──────┘ │
                     │        │                │        │
                     │        ▼                ▼        │
                     │  ┌──────────────────────────┐    │
                     │  │      SERVING LAYER        │    │
                     │  │  (merge batch + speed)    │    │
                     │  └──────────────────────────┘    │
                     └──────────────────────────────────┘
```

### 2.1 Batch Layer

**Vai trò:** Xử lý toàn bộ historical data, tính toán chính xác các view thống kê, re-index RAG Knowledge Base.

**Khi chạy:** Tự động lúc `02:00 AM` hàng đêm (dùng `schedule` library).

```
Input:  Toàn bộ emotion_logs + messages trong SQLite
Output: batch_views.json (emotion_dist, alert_by_hour, resolution_time)
        ChromaDB được re-index với data mới nhất
```
1. Re-index RAG Knowledge Base
   └── Nếu ban ngày có thêm data/*.json mới
       → Re-embed bằng bge-m3 → cập nhật ChromaDB
       → Accuracy của Orchestrator tốt hơn ngày hôm sau

2. Tính thống kê chính xác 100%
   └── Đọc toàn bộ emotion_logs trong SQLite
       → emotion_dist: hôm nay có bao nhiêu angry, disappointed...
       → alert_by_hour: giờ nào nhiều khách bức xúc nhất
       → avg_resolution_time: trung bình bao lâu giải quyết 1 ticket

3. Phát hiện accuracy giảm
   └── So sánh emotion predicted vs feedback thật
       → Nếu accuracy < 75% → flag cần generate thêm data
       → Nếu nhãn "disappointed" sai nhiều → alert để xử lý

**Các job:**
```
| Job | Mô tả | Thời gian |
|-----|-------|-----------|
| `batch_aggregate()` | Tính emotion distribution, alert rate, resolution time | ~1 phút |
| `batch_reindex()` | Re-embed data/*.json → ChromaDB (có cache MD5) | ~15 phút |
| `batch_accuracy()` | So sánh predicted vs feedback, flag nếu < 75% | ~2 phút |
```


### 2.2 Speed Layer

**Vai trò:** Xử lý từng tin nhắn ngay khi nó đến, cập nhật in-memory counters, quản lý alert queue.

**Đặc điểm:** Dữ liệu mất khi restart — nhưng Batch Layer đã có toàn bộ historical data trong SQLite.

```python
SpeedLayer (in-memory):
  ├── emotion_counts: defaultdict(int)   # đếm theo nhãn
  ├── msg_count: int                     # tổng tin nhắn từ lúc start
  ├── alert_count: int                   # tổng alerts
  ├── alert_queue: deque(maxlen=100)     # 100 alerts gần nhất
  └── recent_feed: deque(maxlen=50)      # 50 tin nhắn gần nhất
```

### 2.3 Serving Layer

**Vai trò:** Nhận query từ Admin Dashboard, merge Batch View (đến 2AM) + Speed Delta (từ 2AM đến hiện tại) → trả kết quả chính xác nhất.

```
GET /dashboard
  │
  ├── Batch: emotion_dist (đến 2AM hôm nay)
  └── Speed: emotion_counts (2AM → now)
        │
        ▼
      Merge → kết quả đầy đủ
```

---

## 3. Agentic RAG Pipeline

Nằm trong **Speed Layer**, chịu trách nhiệm phân tích cảm xúc mỗi tin nhắn.

```
turns[]
  │
  ▼
Orchestrator
  ├── Step 1: complexity_check()
  │     └── simple / complex
  │
  ├── Step 2: Tools
  │     ├── ContextAnalyzer   → neg_signals, escalation_risk
  │     ├── SlangLookup       → teencode, phương ngữ Nam
  │     └── HybridSearch      → ChromaDB (vector) + BM25 (keyword) + RRF
  │
  ├── Step 3: build_prompt()
  │     └── conv + rag_block + slang_block + few-shot + CoT
  │
  ├── Step 4: LLM
  │     ├── simple  → deepseek-chat
  │     └── complex → deepseek-reasoner
  │
  └── Step 5: self_reflect()
        └── conf < 0.75 → retry (max 2 lần)
```

### Hybrid Search — RRF Fusion

```
Query
  ├── bge-m3 encode → ChromaDB vector search → top-K với score
  └── BM25Okapi     → keyword search         → top-K với rank

RRF Fusion: score(doc) = Σ 1/(k + rank_i)   k=60
  └── Cộng score từ cả 2 nguồn → re-rank → top-3 results
```

---

## 4. Context Engineering

Các kỹ thuật prompt engineering áp dụng trong Agentic RAG:

| Kỹ thuật | Trạng thái | Mô tả |
|---|---|---|
| System Prompt Design | ✅ | Role + ràng buộc + output format |
| Few-shot Prompting | ✅ | 5 hard cases (lịch sự giả, phương ngữ) |
| Chain of Thought | ✅ | 5 bước suy luận cho complex cases |
| Instruction Hierarchy | ✅ | Priority 1→4 (Safety > Task > Style > Format) |
| Negative Prompting | ✅ | Liệt kê rõ KHÔNG làm gì |
| Sliding Window | ✅ | Giữ 6 turns cuối |
| Summarize Middle | ✅ | DeepSeek tóm tắt turns giữa |
| Smart Truncate | ✅ | Token counting + cắt dần |
| Retrieval Compression | ✅ | Chỉ lấy context_clues liên quan |
| Persona Calibration | ⬜ | Tune tone theo emotion (TODO) |
| Dynamic Context Injection | ⬜ | VIP, emotion trend (TODO) |
| LLMLingua | ⬜ | Khi turns > 30 (TODO) |
| HyDE | ⬜ | Hypothetical Document Embedding (TODO) |
| Query Rewriting | ⬜ | Viết lại query trước khi search (TODO) |

---

## 5. Data Flow

```
User nhắn tin
      │
      ▼
POST /chat
      │
      ├─── is_order_query? ──YES──▶ OrderLookupTool ──▶ SQL query ──▶ reply
      │
      └─── NO
            │
            ├── DeepSeek API ──────────────────────────────────▶ reply text
            │
            └── Orchestrator (Agentic RAG)
                  ├── ContextAnalyzer
                  ├── SlangLookup
                  ├── HybridSearch (ChromaDB + BM25 + RRF)
                  ├── PromptBuilder (few-shot + CoT + RAG block)
                  ├── LLM (chat / reasoner)
                  └── SelfReflect (retry nếu conf < 0.75)
                        │
                        ▼
                  {emotion, confidence, reason, alert}
      │
      ├── save_message(DB)
      ├── save_emotion_log(DB)
      └── speed_layer.ingest()
            ├── update counters
            └── alert? → alert_queue.append()
      │
      ▼
Response: {reply, emotion, confidence, alert, ticket_id}
```

---

## 6. OrderLookup Tool

Ưu tiên chạy **trước** Agentic RAG. Dùng SQL cố định, không Text-to-SQL.

```
Nhận diện ý định:
  ├── Có mã đơn (DH001)  → get_order(id)
  ├── Có SĐT (0901...)   → get_orders_by_phone(phone)
  └── Có keyword đơn hàng → hỏi thêm mã / SĐT

Không trigger OrderLookup:
  └── "ship lâu vl" → phàn nàn → Emotion pipeline
```

---

## 7. Database Schema

**8 bảng SQLite** (`cskh.db`):

```
customers    ──┐
               ├──▶ orders ──▶ order_items ──▶ products
               │         └──▶ payments
               │
               └──▶ tickets ──▶ messages ──▶ emotion_logs
                         └──▶ ticket_agents ──▶ agents
```

| Bảng | Vai trò |
|---|---|
| `customers` | Thông tin khách, tier (normal/silver/gold/vip) |
| `products` | Danh mục sản phẩm |
| `orders` | Đơn hàng, status, shipper, ETA |
| `order_items` | Chi tiết sản phẩm trong đơn |
| `payments` | Thanh toán, hoàn tiền |
| `tickets` | Mỗi cuộc hội thoại = 1 ticket |
| `messages` | Từng tin nhắn trong ticket |
| `emotion_logs` | Kết quả phân tích cảm xúc (nguồn truth cho Batch) |

---

## 8. Thang cảm xúc — 7 nhãn

| Nhãn | Severity | Ghi chú |
|---|---|---|
| `neutral` | 0 | Hỏi thông tin, không cảm xúc |
| `happy` | 0 | Hài lòng, khen ngợi |
| `confused` | 1 | Bối rối, không hiểu |
| `anxious` | 2 | Lo lắng tiền / đơn / gấp |
| `frustrated` | 3 | Bực bội vì chờ lâu |
| `disappointed` | 4 | ⚠️ Thất vọng ngầm, âm thầm rời đi |
| `angry` | 5 | 🚨 Tức giận, đe dọa → escalate ngay |

> `disappointed` nguy hiểm hơn `angry` vì không có cảnh báo rõ ràng.

---

## 9. Tech Stack

| Layer | Technology | Lý do chọn |
|---|---|---|
| LLM API | DeepSeek-chat / DeepSeek-R1 | Chi phí thấp, tốt với tiếng Việt |
| Embedding | `BAAI/bge-m3` (1024 dims) | State-of-the-art multilingual |
| Vector DB | ChromaDB (cosine, HNSW) | Nhẹ, không cần server riêng |
| Keyword | BM25Okapi (`rank_bm25`) | Bổ sung cho vector search |
| Database | SQLite → PostgreSQL | SQLite cho prototype, migrate sau |
| Backend | FastAPI + uvicorn | Async, type hints, auto docs |
| Batch | `schedule` library | Đơn giản, đủ dùng cho prototype |
| Speed | In-memory (`deque`, `defaultdict`) | Nhanh, không cần Redis ở giai đoạn này |
| Frontend | Bootstrap 5 + Chart.js | Nhanh, không cần build step |

> **Kafka:** Chưa cần ở giai đoạn prototype (< 1K user/ngày). Sẽ xem xét khi scale > 10K msg/ngày.

---

## 10. API Endpoints

| Method | Endpoint | Layer | Mô tả |
|---|---|---|---|
| `POST` | `/chat` | Speed | Gửi tin nhắn → reply + emotion |
| `GET` | `/history/{id}` | Storage | Lịch sử hội thoại |
| `GET` | `/orders/{id}` | Storage | Chi tiết đơn hàng |
| `GET` | `/dashboard` | Serving | Merge Batch + Speed stats |
| `GET` | `/feed` | Speed | 50 tin nhắn gần nhất |
| `GET` | `/alerts` | Speed | Alert queue realtime |

---

## 11. Cấu trúc thư mục

```
NLP/
├── chatbot_api/
│   ├── __init__.py
│   ├── main.py          ← FastAPI + endpoints
│   ├── db.py            ← SQLite helper
│   ├── rag.py           ← ChromaDB + BM25 + bge-m3
│   ├── orchestrator.py  ← Agentic RAG pipeline
│   ├── chatbot.py       ← Chat handler
│   ├── tools.py         ← OrderLookup tool
│   └── .cache/          ← Embedding cache (MD5)
├── speed/
│   └── realtime_state.py  ← SpeedLayer class
├── serving/
│   └── merge.py           ← Serving Layer merge logic
├── batch/
│   └── nightly_job.py     ← Batch jobs + scheduler
├── frontend/
│   ├── index.html         ← Chat widget
│   └── admin.html         ← Admin dashboard
├── docs/
│   ├── uml_lambda.html    ← UML diagrams
│   ├── system_design.md   ← File này
│   └── project_context.md
├── data/
│   ├── neutral.json
│   ├── happy.json
│   ├── confused.json
│   ├── anxious.json
│   ├── frustrated.json
│   ├── disappointed.json
│   ├── angry.json
│   └── all.json
├── cskh.db
├── batch_views.json       ← Output của Batch Layer
└── test_api.py
```

---

## 12. Kết quả hiện tại & Roadmap

### Hiện tại
- Accuracy: **~60%** (3/5 test cases)
- Hay sai: `disappointed` (lịch sự giả), phương ngữ Nam

### Roadmap

**Phase 1 — Cải thiện accuracy (ngắn hạn)**
- [ ] Thêm few-shot 10 hard cases vào orchestrator
- [ ] Chain of Thought cho complex cases
- [ ] Generate thêm data cho nhãn `disappointed`
- [ ] Đo F1 / confusion matrix 7×7

**Phase 2 — Hoàn thiện Lambda (trung hạn)**
- [ ] Viết `batch/nightly_job.py` hoàn chỉnh
- [ ] `GET /dashboard` dùng Serving Layer thật
- [ ] `GET /feed` + `GET /alerts` từ SpeedLayer
- [ ] Kết nối frontend thật với API (bỏ mock data)

**Phase 3 — Scale (dài hạn)**
- [ ] Migrate SQLite → PostgreSQL
- [ ] Fine-tune PhoBERT-base-v2 classifier
- [ ] Thêm Kafka khi > 10K msg/ngày
- [ ] Authentication (JWT)
- [ ] Deploy lên VPS / Railway

---

*Tài liệu thiết kế hệ thống — EmotionAI CSKH v2.0*
