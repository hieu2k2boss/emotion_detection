# Lambda Architecture — Implementation Guide

## 3 phần đã implement:

### ✅ 1. Speed Layer Tracking
**File:** [batch/speed_layer.py](batch/speed_layer.py)

**Features:**
- In-memory counters: `emotion_counts`, `message_count`, `alert_count`
- Realtime feed queue (100 messages gần nhất)
- Alert queue (200 alerts gần nhất)
- Per-ticket emotion tracking

**API Endpoints:**
```
GET /admin/speed/stats     → Thống kê realtime
GET /admin/speed/feed      → 50 tin nhắn gần nhất
GET /admin/speed/alerts    → 50 alerts gần nhất
```

---

### ✅ 2. Serving Layer Merge Logic
**File:** [batch/serving_layer.py](batch/serving_layer.py)

**Merge Functions:**
- `merge_emotion_stats()` — Batch + Speed emotion distribution
- `merge_high_risk_customers()` — Batch risk score + Speed recent emotion
- `merge_alert_report()` — Batch daily summary + Speed hôm nay
- `merge_live_feed()` — Speed Layer with emotion enrichment
- `get_dashboard_overview()` — Tổng hợp tất cả

**API Endpoints:**
```
GET /admin/serving/emotion-stats       → Merge Batch + Speed stats
GET /admin/serving/high-risk          → Merge high risk customers
GET /admin/serving/alert-report/{date} → Merge alerts
GET /admin/serving/live-feed          → Live feed (enriched)
GET /admin/serving/dashboard          → Full overview
```

---

### ✅ 3. Dashboard Admin Updated
**File:** [Frontend/dashboard_admin.html](Frontend/dashboard_admin.html)

**Changes:**
- Gọi `/admin/serving/dashboard` thay vì chỉ `/admin/tickets`
- Stats lấy từ Serving Layer (merge Batch + Speed)
- Emotion chart từ merged data
- Live feed tự động refresh mỗi 10s từ API
- Alert list dùng real data từ Speed Layer

---

## Cách test:

### 1. Start server:
```bash
cd /home/hieupt/Bản tải về/emotion_detection-software_architecture
python main.py
```

### 2. Test Speed Layer tracking:
```bash
# Gửi vài tin nhắn
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "ship lâu vl 5 ngày rồi"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "tôi rất thất vọng với dịch vụ"}'

# Check Speed Layer stats
curl http://localhost:8000/admin/speed/stats

# Check live feed
curl http://localhost:8000/admin/speed/feed

# Check alerts
curl http://localhost:8000/admin/speed/alerts
```

### 3. Test Serving Layer merge:
```bash
# Emotion stats (merge Batch + Speed)
curl http://localhost:8000/admin/serving/emotion-stats?hours=24

# Live feed (enriched)
curl http://localhost:8000/admin/serving/live-feed?limit=20

# Full dashboard overview
curl http://localhost:8000/admin/serving/dashboard
```

### 4. Test Dashboard Frontend:
```bash
# Mở file trong browser
file:///home/hieupt/Bản%20t%E1%BA%A3i%20v%E1%BB%81/emotion_detection-software_architecture/Frontend/dashboard_admin.html
```

**Kiểm tra:**
- Stats có hiển thị real data?
- Live Feed tab có cập nhật tự động?
- Alert count có đúng?
- Emotion chart có từ merged data?

---

## Lambda Architecture Flow:

```
User Message
    ↓
POST /chat
    ↓
┌──────────────────┐
│  Speed Layer     │  ← Track realtime counters
│  (in-memory)     │
└──────────────────┘
    ↓
┌──────────────────┐
│  Batch Layer     │  ← chạy scheduler (APScheduler)
│  (nightly jobs)  │     - emotion_hourly_job
└──────────────────┘     - customer_segment_job
    ↓                       - alert_report_job
┌──────────────────┐
│ Serving Layer    │  ← merge_batch_speed()
│ (API endpoints)  │     + Dashboard calls this
└──────────────────┘
    ↓
Dashboard (Frontend)
```

---

## Troubleshooting:

### Speed Layer not tracking?
- Check if `init_speed_layer()` was called in `lifespan()`
- Check logs: `[SpeedLayer] Initialized`

### Serving Layer returns empty?
- Make sure Batch Layer jobs đã chạy ít nhất 1 lần
- Manual trigger: `POST /admin/batch/run-now`

### Dashboard shows no data?
- Open browser console (F12)
- Check API calls to `/admin/serving/dashboard`
- Verify API returns 200

---

## Files changed/created:

✅ **Created:**
- `batch/speed_layer.py` — Speed Layer implementation
- `batch/serving_layer.py` — Serving Layer merge logic
- `LAMBDA_IMPLEMENTATION.md` — (file này)

✅ **Modified:**
- `main.py` — Init Speed Layer, add serving endpoints
- `Frontend/dashboard_admin.html` — Use Serving APIs

---

**Done! Lambda Architecture hoàn chỉnh.** 🎉