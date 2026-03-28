"""
load_sim.py — Giả lập hàng nghìn / triệu user hỏi đáp
=====================================================
Cài đặt:
    pip install aiohttp rich faker tqdm asyncio

Chạy:
    python load_sim.py --users 1000 --rps 50
    python load_sim.py --users 100000 --rps 500 --mode stress
    python load_sim.py --mode spike
"""

import asyncio
import aiohttp
import argparse
import random
import time
import json
import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[!] pip install rich  để xem dashboard đẹp hơn")

# ══════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════
API_BASE = "http://127.0.0.1:8005"

SAMPLE_MESSAGES = [
    # Order queries
    "cho tôi thông tin đơn hàng DH001",
    "đơn DH002 giao đến đâu rồi?",
    "SĐT 0901234567 có đơn nào không?",
    "đơn hàng của tôi khi nào giao?",
    "mã đơn DH003 đang ở đâu vậy?",
    # Neutral
    "shop bán những mặt hàng gì?",
    "giờ làm việc của shop là mấy giờ?",
    "cho tôi xem bảng giá",
    "shop có chi nhánh ở Hà Nội không?",
    # Happy
    "ship nhanh vcl cảm ơn shop nhiều nha!",
    "hàng đẹp lắm, đúng mô tả, mua lần nữa",
    "dịch vụ tốt quá, 5 sao cho shop",
    # Confused
    "tôi không hiểu cách đặt hàng như thế nào?",
    "app bị lỗi, tôi không thấy giỏ hàng",
    "sao không thanh toán được vậy?",
    # Anxious
    "tiền đã bị trừ rồi mà không thấy đơn",
    "thanh toán 2 lần rồi, sợ bị trừ tiền thêm",
    "đơn ghi giao ngày mai mà giờ chưa thấy",
    # Frustrated
    "ship lâu vl 5 ngày rồi mà chưa thấy",
    "gọi điện mấy lần không ai bắt máy",
    "chờ hoài mà không có ai hỗ trợ",
    "đây là lần thứ 3 tôi hỏi rồi đó",
    # Disappointed
    "thôi kệ đi, lần sau chắc không mua nữa",
    "ừ thôi được rồi, cảm ơn bạn vậy",
    "thôi bỏ qua đi, tôi hiểu rồi",
    # Angry
    "tôi sẽ post lên mạng cho mọi người biết",
    "quản lý đâu, tôi cần gặp quản lý ngay",
    "lần này mà không giải quyết tôi kiện đó",
]

# Phân phối cảm xúc thực tế (%) — neutral nhiều nhất
EMOTION_WEIGHTS = [15, 15, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2]

# ══════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════
@dataclass
class RequestResult:
    user_id:      int
    message:      str
    status:       int        # HTTP status
    latency_ms:   float
    emotion:      str
    confidence:   float
    alert:        bool
    error:        Optional[str] = None
    timestamp:    float = field(default_factory=time.time)

@dataclass
class SimStats:
    total:        int   = 0
    success:      int   = 0
    failed:       int   = 0
    alerts:       int   = 0
    latencies:    list  = field(default_factory=list)
    emotions:     dict  = field(default_factory=lambda: defaultdict(int))
    errors:       list  = field(default_factory=list)
    start_time:   float = field(default_factory=time.time)
    rps_window:   deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def elapsed(self):
        return time.time() - self.start_time

    @property
    def avg_latency(self):
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0

    @property
    def p95_latency(self):
        if not self.latencies: return 0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.95)]

    @property
    def p99_latency(self):
        if not self.latencies: return 0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.99)]

    @property
    def current_rps(self):
        now = time.time()
        recent = [t for t in self.rps_window if now - t < 1.0]
        return len(recent)

    @property
    def success_rate(self):
        return self.success / self.total * 100 if self.total > 0 else 0

# ══════════════════════════════════════════════════
#  VIRTUAL USER
# ══════════════════════════════════════════════════
async def virtual_user(
    session:   aiohttp.ClientSession,
    user_id:   int,
    stats:     SimStats,
    n_turns:   int = 3,
    delay:     float = 0.1,
):
    """
    1 virtual user: gửi n_turns tin nhắn trong 1 session
    Giữ ticket_id để giả lập hội thoại liên tục
    """
    ticket_id = None
    messages  = random.choices(SAMPLE_MESSAGES, weights=EMOTION_WEIGHTS, k=n_turns)

    for msg in messages:
        t_start = time.time()
        try:
            async with session.post(
                f"{API_BASE}/chat",
                json    = {"message": msg, "ticket_id": ticket_id},
                timeout = aiohttp.ClientTimeout(total=10),
            ) as resp:
                latency = (time.time() - t_start) * 1000
                status  = resp.status

                if status == 200:
                    data      = await resp.json()
                    ticket_id = data.get("ticket_id")
                    emotion   = data.get("emotion", "unknown")
                    conf      = data.get("confidence", 0.0)
                    alert     = data.get("alert", False)

                    stats.total   += 1
                    stats.success += 1
                    if alert: stats.alerts += 1
                    stats.latencies.append(latency)
                    stats.emotions[emotion] += 1
                    stats.rps_window.append(time.time())
                else:
                    stats.total  += 1
                    stats.failed += 1
                    stats.errors.append(f"HTTP {status}")

        except asyncio.TimeoutError:
            stats.total  += 1
            stats.failed += 1
            stats.errors.append("Timeout")
        except aiohttp.ClientConnectorError:
            stats.total  += 1
            stats.failed += 1
            stats.errors.append("Connection refused — server chưa chạy?")
        except Exception as e:
            stats.total  += 1
            stats.failed += 1
            stats.errors.append(str(e)[:60])

        await asyncio.sleep(delay)

# ══════════════════════════════════════════════════
#  SIMULATION MODES
# ══════════════════════════════════════════════════
async def run_constant_load(
    n_users:     int,
    target_rps:  int,
    duration_s:  int,
    stats:       SimStats,
    n_turns:     int = 2,
):
    """Tải đều — target_rps request/giây trong duration_s giây"""
    print(f"\n🚀 Constant Load: {target_rps} RPS × {duration_s}s = ~{target_rps*duration_s} requests")

    connector = aiohttp.TCPConnector(limit=min(n_users, 1000), ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks    = []
        interval = 1.0 / target_rps
        end_time = time.time() + duration_s
        uid      = 0

        while time.time() < end_time:
            uid += 1
            task = asyncio.create_task(
                virtual_user(session, uid, stats, n_turns=n_turns, delay=0)
            )
            tasks.append(task)
            await asyncio.sleep(interval)

            # Progress
            if uid % 100 == 0:
                _print_progress(stats, uid)

        await asyncio.gather(*tasks, return_exceptions=True)


async def run_ramp_up(
    max_users:   int,
    ramp_s:      int,
    hold_s:      int,
    stats:       SimStats,
):
    """Tăng dần — ramp từ 1 đến max_users trong ramp_s giây, giữ hold_s giây"""
    print(f"\n📈 Ramp Up: 0 → {max_users} users trong {ramp_s}s, giữ {hold_s}s")

    connector = aiohttp.TCPConnector(limit=min(max_users, 2000))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        # Ramp phase
        for i in range(max_users):
            uid  = i + 1
            task = asyncio.create_task(
                virtual_user(session, uid, stats, n_turns=3, delay=0.05)
            )
            tasks.append(task)
            await asyncio.sleep(ramp_s / max_users)

            if uid % 50 == 0:
                _print_progress(stats, uid)

        # Hold phase
        print(f"\n⏸️  Holding {max_users} concurrent users for {hold_s}s...")
        hold_tasks = [
            asyncio.create_task(virtual_user(session, max_users+i, stats, n_turns=5))
            for i in range(min(max_users, 500))
        ]
        await asyncio.sleep(hold_s)

        await asyncio.gather(*tasks, *hold_tasks, return_exceptions=True)


async def run_spike(
    baseline_rps: int,
    spike_rps:    int,
    spike_s:      int,
    stats:         SimStats,
):
    """Spike — đột ngột tăng gấp 10x rồi giảm về"""
    print(f"\n⚡ Spike Test: {baseline_rps} RPS → spike {spike_rps} RPS ({spike_s}s) → {baseline_rps} RPS")

    connector = aiohttp.TCPConnector(limit=2000)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        uid   = 0

        # Phase 1: Baseline 10s
        print("Phase 1: Baseline...")
        for _ in range(baseline_rps * 10):
            uid += 1
            tasks.append(asyncio.create_task(
                virtual_user(session, uid, stats, n_turns=1)
            ))
            await asyncio.sleep(1.0 / baseline_rps)

        # Phase 2: Spike
        print(f"Phase 2: SPIKE {spike_rps} RPS!")
        for _ in range(spike_rps * spike_s):
            uid += 1
            tasks.append(asyncio.create_task(
                virtual_user(session, uid, stats, n_turns=1)
            ))
            await asyncio.sleep(1.0 / spike_rps)
            if uid % 200 == 0:
                _print_progress(stats, uid)

        # Phase 3: Recovery
        print("Phase 3: Recovery...")
        for _ in range(baseline_rps * 10):
            uid += 1
            tasks.append(asyncio.create_task(
                virtual_user(session, uid, stats, n_turns=1)
            ))
            await asyncio.sleep(1.0 / baseline_rps)

        await asyncio.gather(*tasks, return_exceptions=True)


async def run_stress(
    start_rps: int,
    step_rps:  int,
    step_s:    int,
    max_rps:   int,
    stats:     SimStats,
):
    """Stress test — tăng dần cho đến khi hệ thống chịu không nổi"""
    print(f"\n💥 Stress Test: {start_rps} → {max_rps} RPS, step={step_rps} mỗi {step_s}s")

    connector = aiohttp.TCPConnector(limit=5000)
    async with aiohttp.ClientSession(connector=connector) as session:
        rps  = start_rps
        uid  = 0
        prev_success_rate = 100.0

        while rps <= max_rps:
            print(f"\n  🔧 Testing {rps} RPS for {step_s}s...")
            tasks     = []
            t_start   = time.time()
            step_stat = SimStats()

            while time.time() - t_start < step_s:
                uid += 1
                task = asyncio.create_task(
                    virtual_user(session, uid, step_stat, n_turns=1)
                )
                tasks.append(task)
                await asyncio.sleep(1.0 / rps)

            await asyncio.gather(*tasks, return_exceptions=True)

            # Merge step stats
            stats.total   += step_stat.total
            stats.success += step_stat.success
            stats.failed  += step_stat.failed
            stats.latencies.extend(step_stat.latencies)
            for k, v in step_stat.emotions.items():
                stats.emotions[k] += v

            sr = step_stat.success_rate
            al = step_stat.avg_latency
            print(f"  → success={sr:.1f}% | avg_latency={al:.0f}ms | total={step_stat.total}")

            # Breaking point
            if sr < 80 or al > 5000:
                print(f"\n  🔴 BREAKING POINT tại {rps} RPS!")
                print(f"     success_rate={sr:.1f}% | avg_latency={al:.0f}ms")
                break

            prev_success_rate = sr
            rps += step_rps


# ══════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════
def _print_progress(stats: SimStats, uid: int):
    print(
        f"  [{stats.elapsed:5.1f}s] "
        f"users={uid:6d} | "
        f"req={stats.total:7d} | "
        f"ok={stats.success:7d} ({stats.success_rate:5.1f}%) | "
        f"fail={stats.failed:5d} | "
        f"avg={stats.avg_latency:6.0f}ms | "
        f"p95={stats.p95_latency:6.0f}ms | "
        f"alerts={stats.alerts:4d} | "
        f"rps={stats.current_rps:4d}"
    )


def print_final_report(stats: SimStats, mode: str):
    print("\n" + "═"*70)
    print(f"  📊 FINAL REPORT — Mode: {mode.upper()}")
    print("═"*70)
    print(f"  Tổng thời gian    : {stats.elapsed:.2f}s")
    print(f"  Tổng requests     : {stats.total:,}")
    print(f"  Thành công        : {stats.success:,} ({stats.success_rate:.2f}%)")
    print(f"  Thất bại          : {stats.failed:,}")
    print(f"  Alerts triggered  : {stats.alerts:,}")
    print(f"  Throughput (avg)  : {stats.total/stats.elapsed:.1f} req/s")
    print()
    print(f"  Latency:")
    print(f"    avg  = {stats.avg_latency:.0f} ms")
    print(f"    p95  = {stats.p95_latency:.0f} ms")
    print(f"    p99  = {stats.p99_latency:.0f} ms")
    if stats.latencies:
        print(f"    min  = {min(stats.latencies):.0f} ms")
        print(f"    max  = {max(stats.latencies):.0f} ms")
    print()
    print("  Emotion Distribution:")
    total_em = sum(stats.emotions.values())
    ICONS = {"neutral":"😐","happy":"😊","confused":"🤔","anxious":"😰",
             "frustrated":"😤","disappointed":"😢","angry":"😠"}
    for em, cnt in sorted(stats.emotions.items(), key=lambda x: -x[1]):
        pct  = cnt / total_em * 100 if total_em else 0
        icon = ICONS.get(em, "❓")
        bar  = "█" * int(pct / 2)
        print(f"    {icon} {em:<14} {bar:<25} {cnt:5d} ({pct:.1f}%)")
    if stats.errors:
        from collections import Counter
        ec = Counter(stats.errors).most_common(5)
        print("\n  Top Errors:")
        for err, cnt in ec:
            print(f"    [{cnt:4d}×] {err}")
    print("═"*70)


def save_csv(stats: SimStats, filename: str = "load_test_results.csv"):
    """Lưu latency data ra CSV để phân tích sau"""
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "latency_ms"])
        for i, lat in enumerate(stats.latencies):
            w.writerow([i, f"{lat:.2f}"])
    print(f"\n  💾 Saved latency data → {filename}")


# ══════════════════════════════════════════════════
#  QUICK HEALTH CHECK
# ══════════════════════════════════════════════════
async def health_check() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_BASE}/", timeout=aiohttp.ClientTimeout(total=3)) as r:
                return r.status == 200
    except:
        return False


# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════
async def main():
    parser = argparse.ArgumentParser(description="EmotionAI Load Simulator")
    parser.add_argument("--mode",     default="constant",
                        choices=["constant","ramp","spike","stress"],
                        help="Chế độ giả lập")
    parser.add_argument("--users",    type=int, default=100,
                        help="Số virtual users (constant/ramp mode)")
    parser.add_argument("--rps",      type=int, default=20,
                        help="Target requests/second")
    parser.add_argument("--duration", type=int, default=30,
                        help="Thời gian chạy (giây) - constant mode")
    parser.add_argument("--turns",    type=int, default=2,
                        help="Số tin nhắn mỗi user")
    parser.add_argument("--save",     action="store_true",
                        help="Lưu kết quả ra CSV")
    parser.add_argument("--url",      default=API_BASE,
                        help=f"API base URL (default: {API_BASE})")
    args = parser.parse_args()

    global API_BASE
    API_BASE = args.url

    print("╔══════════════════════════════════════════════╗")
    print("║    EmotionAI CSKH — Load Simulator           ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  API    : {API_BASE}")
    print(f"  Mode   : {args.mode}")
    print(f"  Users  : {args.users:,}")
    print(f"  RPS    : {args.rps}")

    # Health check
    print("\n⏳ Kiểm tra server...")
    ok = await health_check()
    if not ok:
        print("❌ Server không phản hồi. Chạy uvicorn trước:\n")
        print("   cd NLP/")
        print("   python -m uvicorn chatbot_api.main:app --port 8005")
        return
    print(" Server OK\n")

    stats = SimStats()

    try:
        if args.mode == "constant":
            await run_constant_load(
                n_users    = args.users,
                target_rps = args.rps,
                duration_s = args.duration,
                stats      = stats,
                n_turns    = args.turns,
            )
        elif args.mode == "ramp":
            await run_ramp_up(
                max_users = args.users,
                ramp_s    = min(args.duration, 60),
                hold_s    = 20,
                stats     = stats,
            )
        elif args.mode == "spike":
            await run_spike(
                baseline_rps = max(args.rps // 10, 5),
                spike_rps    = args.rps,
                spike_s      = 15,
                stats        = stats,
            )
        elif args.mode == "stress":
            await run_stress(
                start_rps = max(args.rps // 10, 5),
                step_rps  = max(args.rps // 10, 5),
                step_s    = 10,
                max_rps   = args.rps,
                stats     = stats,
            )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted — generating report...")

    print_final_report(stats, args.mode)

    if args.save:
        save_csv(stats)


if __name__ == "__main__":
    asyncio.run(main())