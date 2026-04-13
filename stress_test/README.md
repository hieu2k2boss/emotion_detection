# Stress Test — Lambda Architecture

## User Scenarios

### 1. ChatUser (Normal Customer)
- **Purpose:** Simulate real customer conversations
- **Weight:** 6 (highest)
- **Actions:**
  - POST /chat (6x) — Send messages with various emotions
  - GET /history/{id} (2x) — View ticket history
  - GET /orders/{id} (1x) — Track orders

**Test coverage:**
- All 7 emotions (neutral, happy, confused, anxious, frustrated, disappointed, angry)
- Multi-turn conversations (3-5 messages per session)
- Response validation (all required fields present)

---

### 2. SpeedLayerUser (Realtime Monitoring)
- **Purpose:** Test Speed Layer in-memory tracking
- **Weight:** Fast polling (0.5-2s intervals)
- **Actions:**
  - GET /admin/speed/stats (3x) — Realtime stats
  - GET /admin/speed/feed (3x) — Live feed
  - GET /admin/speed/alerts (2x) — Alert queue

**Metrics:**
- Response time < 100ms (should be very fast — in-memory)
- No data loss under load

---

### 3. ServingLayerUser (Admin Dashboard)
- **Purpose:** Test Batch + Speed merge logic
- **Weight:** Medium load (3-8s intervals)
- **Actions:**
  - GET /admin/serving/dashboard (5x) — Full dashboard
  - GET /admin/serving/emotion-stats (2x) — Merged stats
  - GET /admin/serving/high-risk (2x) — Risk customers
  - GET /admin/serving/live-feed (2x) — Enriched feed
  - GET /admin/batch/* (1x) — Batch-only (baseline)

**Test:**
- Merge correctness (Batch + Speed = correct totals)
- Performance impact of merge vs batch-only

---

### 4. MixedScenarioUser (Concurrent)
- **Purpose:** Test mixed workload (chat + dashboard same session)
- **Weight:** 1 (low)
- **Actions:**
  - 80% chat + 20% dashboard queries
  - Simulates power users / supervisors

**Test:**
- Connection pooling
- Resource contention

---

## How to Run

### 1. Install Locust
```bash
pip install locust
```

### 2. Start API Server
```bash
cd /home/hieupt/Bản\ tải\ về/emotion_detection-software_architecture
python main.py
```

### 3. Run Stress Test

**GUI Mode (recommended):**
```bash
cd /home/hieupt/Bản\ tải\ về/emotion_detection-software_architecture
locust -f stress_test/locustfile.py --host=http://localhost:8000
```
Then open http://localhost:8089

**Headless Mode (CI/CD):**
```bash
locust -f stress_test/locustfile.py --host=http://localhost:8000 \
  --headless -u 100 -r 10 -t 5m --html report.html
```

**Test specific scenarios:**
```bash
# Only Speed Layer
locust -f stress_test/locustfile.py --host=http://localhost:8000 \
  ChatUser SpeedLayerUser

# Only Serving Layer
locust -f stress_test/locustfile.py --host=http://localhost:8000 \
  ServingLayerUser
```

---

## Recommended Test Settings

### Load Test (Normal Load)
```
Number of users: 50
Spawn rate: 5 users/s
Run time: 5 minutes
```

### Stress Test (Peak Load)
```
Number of users: 200
Spawn rate: 20 users/s
Run time: 10 minutes
```

### Endurance Test (Stability)
```
Number of users: 20
Spawn rate: 1 user/s
Run time: 1 hour
```

---

## Metrics to Watch

### Response Times (p95)
| Endpoint | Target | Warning |
|---|---|---|
| POST /chat | < 3s | > 5s |
| GET /admin/speed/* | < 100ms | > 500ms |
| GET /admin/serving/dashboard | < 500ms | > 2s |

### Error Rate
- **Target:** < 1%
- **Acceptable:** < 5%
- **Critical:** > 10%

### Throughput
- **Target:** 50+ requests/second
- **Baseline:** Compare batch vs serving performance

---

## Performance Bottlenecks to Check

1. **Speed Layer**
   - In-memory operations should be < 50ms
   - If slow → check deque operations

2. **Serving Layer**
   - Merge logic should not regress > 2x vs batch-only
   - If slow → optimize merge queries

3. **Chat Endpoint**
   - LLM calls dominate latency (not data layer)
   - If p95 > 5s → consider streaming responses

---

## Troubleshooting

### Locust won't start
```bash
# Check port 8089 available
lsof -i :8089
```

### API server crashes
```bash
# Check logs for OOM or DB locks
# Reduce user count or increase DB timeout
```

### All requests failing
```bash
# Verify API is running
curl http://localhost:8000/
curl http://localhost:8000/admin/serving/dashboard
```

---

## Example Output

```
Name                                                              # reqs      # fails|    Avg     Min     Max    Median  | req/s failures/s
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
POST /chat                                                            500     0(0.00%)|   1250     800    2500      1200  |   10.0    0.00
GET /admin/serving/dashboard                                          200     0(0.00%)|    300     200     800       280  |    4.0    0.00
GET /admin/speed/feed                                                 600     0(0.00%)|     50      30     150       45  |   12.0    0.00
GET /admin/serving/emotion-stats                                       100     0(0.00%)|    200     150     400       190  |    2.0    0.00
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Total                                                                1400     0(0.00)|                                            28.0    0.00

✅ Success: 100.0% requests passed
Response times:
  50%: 280ms
  95%: 1200ms
  99%: 2400ms
```

---

**Happy testing! 🚀**