# ════════════════════════════════════════════════
# chatbot_api/orchestrator.py
# ════════════════════════════════════════════════
import json, re, uuid
from dataclasses import dataclass, field
from chatbot_api.rag import hybrid_search
from chatbot_api.api_client import call_api

SLANG_DICT = {
    "vl":       "rất tệ",
    "vcl":      "cực kỳ tệ",
    "ừ thôi":   "bỏ cuộc",
    "thôi kệ":  "từ bỏ",
    "bể":       "hỏng (Nam)",
    "hết trơn": "hoàn toàn (Nam)",
    "ko":       "không",
    "k":        "không",
    "ship":     "giao hàng",
    "mn":       "mọi người",
}
NEG_KEYWORDS = [
    "lâu", "mãi", "hoài", "sao chưa", "không thấy",
    "thôi", "kệ", "vl", "vcl", "tức", "bực", "chán",
    "lần cuối", "không mua", "bể", "post lên",
]
SYSTEM_PROMPT = """You must respond ONLY in Vietnamese.
Bạn là chuyên gia phân tích cảm xúc CSKH tiếng Việt.
Nhãn: neutral | happy | confused | anxious | frustrated | disappointed | angry

Quy tắc:
1. "ừ thôi", "thôi kệ" SAU phàn nàn = disappointed
2. Đọc TOÀN BỘ lịch sử hội thoại
3. Phương ngữ Nam: "bể hết trơn", "thôi kệ đi quá"
4. alert=true khi: angry HOẶC disappointed + dấu hiệu bỏ đi

Chỉ trả về JSON: {"emotion":"...","confidence":0.0,"reason":"...","alert":true/false}"""

@dataclass
class WorkingMemory:
    query:           str  = ""
    complexity:      str  = "simple"
    context_summary: dict = field(default_factory=dict)
    slang_found:     dict = field(default_factory=dict)
    retrieved_docs:  list = field(default_factory=list)
    result:          dict = field(default_factory=dict)
    attempts:        int  = 0

# ── Context helpers ───────────────────────────

def sliding_window(turns: list, max_turns: int = 6) -> list:
    return turns[-max_turns:]

def compress_context(turns: list, keep: int = 4) -> list:
    """Giữ 2 turns đầu + tóm tắt giữa + 4 turns cuối"""
    if len(turns) <= keep + 2:
        return turns

    head   = turns[:2]
    tail   = turns[-keep:]
    middle = turns[2:-keep]

    mid_text = "\n".join([
        f"{'Khách' if t['role']=='customer' else 'Agent'}: {t['text']}"
        for t in middle
    ])
    prompt = (
        f"Tóm tắt đoạn hội thoại sau thành 1-2 câu, giữ lại cảm xúc chính:\n{mid_text}"
    )
    try:
        summary = call_api(prompt, session_id=f"compress-{uuid.uuid4().hex[:8]}")
    except Exception:
        summary = "(không tóm tắt được)"

    return head + [{"role": "system", "text": f"[Tóm tắt: {summary}]"}] + tail

# ── Tools ─────────────────────────────────────

def tool_context_analyzer(turns: list) -> dict:
    recent    = turns[-5:]
    cust_msgs = [t["text"] for t in recent if t["role"] == "customer"]
    neg_count = sum(1 for msg in cust_msgs for kw in NEG_KEYWORDS if kw in msg.lower())
    return {
        "n_turns":         len(cust_msgs),
        "neg_signals":     neg_count,
        "escalation_risk": neg_count >= 2 or len(cust_msgs) >= 3,
        "recent_msgs":     cust_msgs,
        "summary":         f"{len(cust_msgs)} turns | {neg_count} tín hiệu tiêu cực",
    }

def tool_slang_lookup(text: str) -> dict:
    found = {k: v for k, v in SLANG_DICT.items() if k in text.lower()}
    return {"found": found}

def build_prompt(mem: WorkingMemory, turns: list) -> str:
    conv = "\n".join([
        f"{'Agent' if t['role']=='agent' else 'Khách'}: {t['text']}"
        for t in turns
    ])
    rag = "\n".join([
        f"- \"{d['last_utterance']}\" → {d['label']} | {d['context_clues']}"
        for d in mem.retrieved_docs
    ]) or "Không có"
    slang   = "\n".join([f"- '{k}' = {v}" for k, v in mem.slang_found.items()]) or "Không có"
    warning = ""
    if mem.context_summary.get("escalation_risk"):
        warning = f"\n⚠️  {mem.context_summary['neg_signals']} tín hiệu tiêu cực tích lũy!"

    return f"""{SYSTEM_PROMPT}

Hội thoại ({len(turns)} turns):
{conv}

Ví dụ tương tự từ KB:
{rag}

Từ lóng / phương ngữ:
{slang}

Ngữ cảnh: {mem.context_summary.get('summary', '')}
{warning}

Câu cần phân tích: "{mem.query}"
Chỉ trả về JSON."""

def call_llm(prompt: str, session_id: str = "orchestrator") -> str:
    """Gọi API phân tích cảm xúc — mỗi lần dùng session riêng biệt để tránh nhiễu ngữ cảnh"""
    return call_api(prompt, session_id=session_id)

def parse_result(raw: str) -> dict:
    raw = re.sub(r'```json|```', '', raw).strip()
    raw = re.sub(r'[\u4e00-\u9fff]+', '', raw)
    m   = re.search(r'\{.*?\}', raw, re.DOTALL)
    try:    return json.loads(m.group()) if m else {}
    except: return {}

from chatbot_api.strategies import StrategyFactory
from chatbot_api.config import settings

# ── Cấu hình Strategy ─────────────────────────
strategy = StrategyFactory.get_strategy(use_mock=settings.USE_MOCK_MODE)

def real_llm_orchestrator(turns: list) -> dict:
    """Logic phân tích cảm xúc gốc sử dụng LLM."""
    mem       = WorkingMemory()
    last_cust = next(
        (t["text"] for t in reversed(turns) if t["role"] == "customer"), ""
    )
    mem.query = last_cust

    # Step 1: Complexity check
    mem.context_summary = tool_context_analyzer(turns)
    mem.complexity = (
        "complex"
        if mem.context_summary["escalation_risk"] or len(turns) > 4
        else "simple"
    )

    # Step 2: Tools
    mem.slang_found    = tool_slang_lookup(last_cust)["found"]
    mem.retrieved_docs = hybrid_search(last_cust, top_k=3)

    if mem.complexity == "complex":
        all_cust = " ".join(t["text"] for t in turns if t["role"] == "customer")
        extra    = hybrid_search(all_cust, top_k=2)
        seen, merged = set(), []
        for d in mem.retrieved_docs + extra:
            if d["doc_id"] not in seen:
                seen.add(d["doc_id"])
                merged.append(d)
        mem.retrieved_docs = merged[:4]

    # Step 3: Generate + Self-reflect
    for attempt in range(1, 3):
        mem.attempts = attempt
        session_id   = f"emotion-{uuid.uuid4().hex[:12]}"
        prompt       = build_prompt(mem, turns)
        raw          = call_llm(prompt, session_id=session_id)
        mem.result   = parse_result(raw)
        conf         = mem.result.get("confidence", 0)

        if conf >= 0.75:
            break
        if mem.complexity == "complex" and attempt == 1:
            mem.query = f"[Phân tích kỹ hơn] {mem.query}"

    return mem.result

# ── Orchestrator chính (Wrapper) ──────────────

def orchestrator(turns: list) -> dict:
    """
    Hàm entry point phân tích cảm xúc, gọi thông qua Strategy.
    """
    return strategy.analyze_emotion(turns)