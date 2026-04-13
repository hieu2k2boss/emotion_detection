# locust -f stress_test/locustfile.py --host=http://localhost:8000
# Sau đó mở http://localhost:8089 để xem dashboard

import random
import time
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# ── Test data ─────────────────────────────────
CHAT_MESSAGES = [
    # Frustrated (ship delay)
    "ship lâu vl 5 ngày rồi mà chưa thấy",
    "đơn hàng mãi không tới, giao hàng chậm quá",
    "cam kết 2 ngày nay đã 5 ngày rồi",

    # Anxious (order tracking)
    "đơn DH001 đang ở đâu rồi ạ",
    "tiền bị trừ mà không thấy đơn",
    "bao giờ giao hàng vậy",
    "0901234567 cho tôi xem đơn",

    # Disappointed (silent quit)
    "thôi kệ đi lần sau không mua nữa",
    "ừ thôi kệ đi quá",
    "mãi không thấy hàng",

    # Angry (escalate)
    "tôi sẽ post lên mạng cho mọi người biết",
    "hoàn tiền ngay cho tôi",
    "làm ăn scam thế",

    # Happy
    "cảm ơn shop giao nhanh lắm",
    "hàng đẹp lắm, đúng mô tả",
    "tốt lắm, ủng hộ shop",

    # Confused
    "sao size L mà mặc chật vậy",
    "không hiểu cách dùng",
    "hướng dẫn lại đi",
]

# FIX 1: Each value is now a list of keywords (was a single comma-joined string
#         that never matched any message with `keyword in m`).
EMOTIONS_MAP = {
    "frustrated":   ["ship lâu", "chậm trễ", "giao hàng", "chậm"],
    "anxious":      ["đơn", "bao giờ", "090", "sđt", "ở đâu"],
    "disappointed": ["thôi kệ", "không mua nữa", "mãi không"],
    "angry":        ["post lên mạng", "hoàn tiền", "scam"],
    "happy":        ["cảm ơn", "tốt lắm", "ủng hộ"],
    "confused":     ["không hiểu", "sao", "hướng dẫn"],
}


def pick_message_for_emotion(emotion_type: str) -> str:
    """Return a chat message matching one of the emotion's keywords.
    Falls back to the full list so random.choice never gets an empty sequence.
    """
    keywords = EMOTIONS_MAP[emotion_type]
    filtered = [m for m in CHAT_MESSAGES if any(kw in m for kw in keywords)]
    return random.choice(filtered if filtered else CHAT_MESSAGES)


# ── User behaviors ────────────────────────────

class ChatUser(HttpUser):
    """
    Normal customer user — chat nhiều, thỉnh thoảng xem order.
    Giả lập 100 customers truy cập cùng lúc.
    """
    wait_time = between(1, 4)  # 1-4s giữa các request

    def on_start(self):
        """Tạo ticket mới cho mỗi user session."""
        self.ticket_id = None
        self.message_count = 0

    @task(6)  # weight cao nhất = chat hay nhất
    def chat_conversation(self):
        """Simulate hội thoại thật — 3-5 tin nhắn liên tục."""
        emotion_type = random.choice(list(EMOTIONS_MAP.keys()))
        msg = pick_message_for_emotion(emotion_type)  # FIX 1 applied here

        payload = {"message": msg}
        if self.ticket_id:
            payload["ticket_id"] = self.ticket_id

        with self.client.post("/chat", json=payload, catch_response=True, name="/chat") as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.ticket_id = data.get("ticket_id")
                self.message_count += 1

                # Validate response
                required_fields = ["ticket_id", "reply", "emotion", "confidence", "alert", "reason"]
                missing = [f for f in required_fields if f not in data]
                if missing:
                    resp.failure(f"Missing fields: {missing}")
                else:
                    resp.success()
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:100]}")

    @task(2)
    def track_order(self):
        """Thỉnh thoảng tra cứu đơn hàng."""
        if self.ticket_id:
            with self.client.get(f"/history/{self.ticket_id}", catch_response=True, name="/history/{id}") as resp:
                if resp.status_code in (200, 404):
                    resp.success()
                else:
                    resp.failure(f"Status {resp.status_code}")

    @task(1)
    def check_random_order(self):
        """Check đơn hàng ngẫu nhiên."""
        order_id = random.choice(["DH001", "DH002", "DH003", "DH004", "DH005"])
        with self.client.get(f"/orders/{order_id}", catch_response=True, name="/orders/{id}") as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")


class SpeedLayerUser(HttpUser):
    """
    Test Speed Layer endpoints — realtime tracking.
    Giả lập ai đó liên tục poll live feed.
    """
    wait_time = between(0.5, 2)  # Poll nhanh

    @task(3)
    def get_speed_stats(self):
        """Check Speed Layer stats."""
        self.client.get("/admin/speed/stats", name="/admin/speed/stats")

    @task(3)
    def get_speed_feed(self):
        """Poll live feed."""
        self.client.get("/admin/speed/feed?limit=30", name="/admin/speed/feed")

    @task(2)
    def get_speed_alerts(self):
        """Check alerts."""
        self.client.get("/admin/speed/alerts?limit=20", name="/admin/speed/alerts")


class ServingLayerUser(HttpUser):
    """
    Test Serving Layer — merge Batch + Speed.
    Giả lập admin dashboard queries.
    """
    wait_time = between(3, 8)

    @task(5)
    def view_full_dashboard(self):
        """Load complete dashboard — test merge logic."""
        with self.client.get("/admin/serving/dashboard", catch_response=True, name="/admin/serving/dashboard") as resp:
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    required = ["emotion_stats_24h", "high_risk_customers", "live_feed", "speed_realtime"]
                    missing = [r for r in required if r not in data]
                    if missing:
                        resp.failure(f"Missing dashboard sections: {missing}")
                    else:
                        resp.success()
                except Exception as e:
                    resp.failure(f"JSON parse error: {e}")
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def get_emotion_stats(self):
        """Test merge emotion stats."""
        hours = random.choice([1, 6, 12, 24, 48])
        self.client.get(f"/admin/serving/emotion-stats?hours={hours}", name="/admin/serving/emotion-stats")

    @task(2)
    def get_high_risk_customers(self):
        """Test merge high risk customers."""
        limit = random.choice([10, 20, 50])
        self.client.get(f"/admin/serving/high-risk?limit={limit}", name="/admin/serving/high-risk")

    @task(2)
    def get_live_feed(self):
        """Test live feed from Serving Layer."""
        limit = random.choice([20, 50, 100])
        self.client.get(f"/admin/serving/live-feed?limit={limit}", name="/admin/serving/live-feed")

    @task(1)
    def get_batch_only_endpoints(self):
        """Test Batch Layer only (so sánh performance)."""
        self.client.get("/admin/batch/emotion-trend?hours=24", name="/admin/batch/emotion-trend")
        self.client.get("/admin/batch/high-risk-customers", name="/admin/batch/high-risk-customers")


class MixedScenarioUser(HttpUser):
    """
    User hỗn hợp — chat + xem dashboard.
    Giả lập customer + admin cùng 1 người (test concurrency).
    """
    wait_time = between(2, 5)
    weight = 1  # Ít hơn ChatUser

    def on_start(self):
        self.ticket_id = None
        self.is_admin_mode = random.choice([True, False])

    @task
    def mixed_usage(self):
        """Random: 80% chat, 20% xem dashboard."""
        if not self.is_admin_mode or random.random() < 0.8:
            msg = random.choice(CHAT_MESSAGES)
            payload = {"message": msg}
            if self.ticket_id:
                payload["ticket_id"] = self.ticket_id

            with self.client.post("/chat", json=payload, catch_response=True, name="/chat (mixed)") as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    self.ticket_id = data.get("ticket_id")
                    resp.success()
                else:
                    resp.failure(f"Status {resp.status_code}")
        else:
            self.client.get("/admin/serving/dashboard", name="/admin/serving/dashboard (mixed)")


# ── Test statistics ────────────────────────────

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """In ra summary khi test kết thúc."""
    total = environment.stats.total
    num_req = total.num_requests

    # FIX 2: success_ratio was removed in newer Locust — compute manually.
    fail_ratio = (total.num_failures / num_req) if num_req > 0 else 0.0
    success_rate = (1 - fail_ratio) * 100

    if fail_ratio > 0.1:
        print(f"\n⚠️ WARNING: {fail_ratio * 100:.1f}% requests failed!")
    else:
        print(f"\n✅ Success: {success_rate:.1f}% requests passed")

    print(f"Total requests: {num_req}")
    print(f"Response times:")
    print(f"  50%: {total.median_response_time}ms")
    print(f"  95%: {total.get_response_time_percentile(0.95)}ms")
    print(f"  99%: {total.get_response_time_percentile(0.99)}ms")


# ── Usage ─────────────────────────────────────

# Chạy test:
#   locust -f stress_test/locustfile.py --host=http://localhost:8000
#
# Mở http://localhost:8089
#
# Recommended settings:
#   - Number of users: 100
#   - Spawn rate: 10 users/s
#   - Run time: 5-10 minutes
#
# Headless mode (CLI):
#   locust -f stress_test/locustfile.py --host=http://localhost:8000 --headless \
#     -u 100 -r 10 -t 5m
#
# Test specific user class:
#   locust -f stress_test/locustfile.py --host=http://localhost:8000 \
#     --class-picker