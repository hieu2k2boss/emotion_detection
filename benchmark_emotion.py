#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  CSKH EmotionAI — Evaluation Framework v4.0             ║
║  So sánh 4 pipeline phân tích cảm xúc tiếng Việt        ║
║                                                          ║
║  v3.0: KB expansion, char_wb TF-IDF, no warning bias,   ║
║        smart self-reflect, conditional RAG               ║
║                                                          ║
║  v4.0 Additions (target: Agentic ≥ 90%):                ║
║  [Add 1] SYSTEM_PROMPT_AGENTIC — disambiguation rules   ║
║           cho 4 confusion pair khó nhất                  ║
║  [Add 2] Structured CoT — "thought" field buộc model    ║
║           reason trước khi kết luận                      ║
║  [Add 3] Dynamic few-shot — format RAG hits thành       ║
║           proper input→output examples                   ║
║  [Add 4] Thought-aware self-reflect — dùng "thought"    ║
║           từ attempt 1 để build targeted clarification   ║
╚══════════════════════════════════════════════════════════╝

Pipeline:
  1. Baseline   — LLM + fixed prompt (không RAG, không tool)
  2. Vector RAG — TF-IDF cosine similarity only
  3. Hybrid RAG — TF-IDF + BM25 + RRF fusion
  4. Agentic    — Hybrid + tool_context_analyzer + tool_slang_lookup + self-reflect

Cách dùng:
  python eval_framework.py                         # Full run (50 scenarios)
  python eval_framework.py --skip-gen              # Dùng cache scenarios
  python eval_framework.py --pipelines 1,4         # Chỉ chạy pipeline 1 & 4
  python eval_framework.py --max-scenarios 10      # Quick smoke test

Output:
  .cache/eval_scenarios.json   — Scenarios đã generate
  .cache/eval_results.json     — Kết quả chi tiết
  eval_report.html             — Report trực quan
"""

import argparse
import json
import math
import re
import time
import os
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG — chỉnh tại đây
# ─────────────────────────────────────────────────────────────

LLM_API_URL   = os.getenv("LLM_API_URL", "https://2f2e0c7c166331.lhr.life")
LLM_API_KEY   = os.getenv("LLM_API_KEY", "67ca894cfcf256bc6855b33d97666180")
LLM_MODEL     = os.getenv("LLM_MODEL",   "Meta-Llama-3-70B-Instruct")
N_PER_LABEL   = 7      # 7 × 7 labels = 49 ≈ 50 scenarios
CACHE_DIR     = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

LABELS = ["neutral", "happy", "confused", "anxious", "frustrated", "disappointed", "angry"]

SLANG_DICT = {
    # Bỏ "vl"/"vcl" — ambiguous: "ship nhanh vcl" = happy, "ship lâu vl" = frustrated
    # Phát hiện qua context, không dùng lookup tĩnh
    "ừ thôi":   "bỏ cuộc (disappointed)",
    "thôi kệ":  "từ bỏ (disappointed)",
    "bể":        "hỏng (phương ngữ Nam)",
    "hết trơn":  "hoàn toàn (phương ngữ Nam)",
    "ko":        "không",
    "k":         "không",
    "ship":      "giao hàng",
    "mn":        "mọi người",
    "oke":       "đồng ý",
    "okê":       "đồng ý",
    "br":        "bây giờ",
}

# FIX Bug #1: Tách 2 tier — strong (1 là đủ) vs weak (cần tích lũy)
NEG_STRONG = [
    "lần cuối", "không mua nữa", "chỗ khác",
    "post lên", "đăng lên", "khiếu nại", "báo công an",
    "không để yên", "lừa dối", "hoàn tiền ngay",
]
NEG_WEAK = [
    "lâu", "mãi", "hoài", "sao chưa", "chưa thấy",
    "tức", "bực", "chán", "trả hàng",
]

SYSTEM_PROMPT = """You must respond ONLY in Vietnamese.
Bạn là chuyên gia phân tích cảm xúc CSKH tiếng Việt.
Nhãn: neutral | happy | confused | anxious | frustrated | disappointed | angry

Quy tắc QUAN TRỌNG:
1. "ừ thôi", "thôi kệ" SAU phàn nàn = disappointed (KHÔNG phải neutral)
2. Đọc TOÀN BỘ lịch sử hội thoại, không chỉ câu cuối
3. Phương ngữ Nam: "bể hết trơn", "thôi kệ đi quá" = disappointed/frustrated
4. Lịch sự giả ("thôi oke rồi") sau phàn nàn = disappointed
5. alert=true khi: angry HOẶC (disappointed + có dấu hiệu rời đi)

Chỉ trả về JSON (không text khác):
{"emotion":"...","confidence":0.0,"reason":"...","alert":true/false}"""

# Prompt riêng cho pipeline Agentic — có CoT + disambiguation chi tiết hơn
SYSTEM_PROMPT_AGENTIC = """You must respond ONLY in Vietnamese.
Bạn là chuyên gia phân tích cảm xúc CSKH tiếng Việt.
Nhãn: neutral | happy | confused | anxious | frustrated | disappointed | angry

═══ PHÂN BIỆT CÁC LABEL DỄ NHẦM ═══
confused vs anxious:
  • confused  = CHỈ bối rối, không hiểu sự mâu thuẫn — KHÔNG lo lắng, KHÔNG cần gấp
  • anxious   = lo lắng thực sự về mất tiền / deadline / đơn hàng mất tích

frustrated vs angry:
  • frustrated = bực bội vì chờ lâu / không được hỗ trợ, KHÔNG có đe dọa cụ thể
  • angry      = đe dọa hành động (post MXH, kiện, báo công an, "không để yên")

disappointed vs neutral:
  • disappointed = BẮT BUỘC có context phàn nàn TRƯỚC ĐÓ + câu kết là bỏ cuộc
  • neutral      = hỏi thông tin đơn thuần, không cảm xúc

═══ QUY TẮC ĐẶC BIỆT ═══
1. "ừ thôi" / "thôi kệ" / "không cần nữa" SAU phàn nàn → disappointed
2. Lịch sự giả ("thôi oke rồi") sau phàn nàn → disappointed, KHÔNG phải happy/neutral
3. Phương ngữ Nam: "bể hết trơn", "thôi kệ đi quá" → disappointed/frustrated
4. alert=true khi: angry HOẶC (disappointed + dấu hiệu rời đi rõ)

═══ QUY TRÌNH BẮT BUỘC ═══
Trước khi kết luận, suy nghĩ ngắn gọn trong trường "thought":
  a) Cảm xúc chủ đạo của TOÀN BỘ hội thoại là gì?
  b) Câu cuối cùng xác nhận hay thay đổi cảm xúc đó?
  c) Có thể nhầm với label nào gần giống?

Trả về JSON — bắt buộc có "thought" TRƯỚC "emotion":
{"thought":"...","emotion":"...","confidence":0.0,"reason":"...","alert":true/false}"""


# ─────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────

def call_llm(messages: list, temperature: float = 0, max_tokens: int = 1500) -> str:
    """Gọi LLM API, retry 3 lần nếu lỗi."""
    import requests as _req
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    payload = {
        "model":       LLM_MODEL,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "ignore_eos":  False,
    }
    for attempt in range(3):
        try:
            r = _req.post(LLM_API_URL, headers=headers, json=payload, timeout=90, verify=False)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"LLM API error after 3 attempts: {e}")
            time.sleep(2 ** attempt)
    return ""


def parse_json_safe(text: str) -> Any:
    """Parse JSON từ LLM output, xử lý markdown fences và junk."""
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)      # lọc tiếng Trung
    m = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — 35 entries cứng (5 × 7 labels)
# Tách biệt hoàn toàn với test scenarios
# ─────────────────────────────────────────────────────────────

HARDCODED_KB = [
    # neutral
    {"label":"neutral","last_utterance":"Cho hỏi sản phẩm này còn hàng không?",
     "context_clues":"hỏi thông tin","difficulty":"easy"},
    {"label":"neutral","last_utterance":"Mã giảm giá SALE50 áp dụng được không ạ?",
     "context_clues":"hỏi voucher","difficulty":"easy"},
    {"label":"neutral","last_utterance":"Cho em hỏi phí ship tới Hà Nội là bao nhiêu?",
     "context_clues":"hỏi phí vận chuyển","difficulty":"easy"},
    {"label":"neutral","last_utterance":"Sản phẩm này có màu đen không ạ?",
     "context_clues":"hỏi màu sắc","difficulty":"easy"},
    {"label":"neutral","last_utterance":"Thời gian giao hàng mất bao lâu vậy?",
     "context_clues":"hỏi thời gian giao","difficulty":"easy"},
    # happy
    {"label":"happy","last_utterance":"Shop giao nhanh lắm cảm ơn nhiều nha!",
     "context_clues":"khen ngợi hài lòng","difficulty":"easy"},
    {"label":"happy","last_utterance":"Hàng y hình, chất lượng tốt, sẽ mua lại",
     "context_clues":"hài lòng chất lượng","difficulty":"easy"},
    {"label":"happy","last_utterance":"ship nhanh vcl cảm ơn shop nha",
     "context_clues":"khen ngợi teencode","difficulty":"easy"},
    {"label":"happy","last_utterance":"Đóng gói kỹ lắm, rất hài lòng ạ",
     "context_clues":"hài lòng đóng gói","difficulty":"easy"},
    {"label":"happy","last_utterance":"Nhân viên tư vấn nhiệt tình quá trời, thích!",
     "context_clues":"khen nhân viên","difficulty":"easy"},
    # confused
    {"label":"confused","last_utterance":"Ủa sao đơn báo đã giao mà em chưa nhận được vậy?",
     "context_clues":"bối rối trạng thái đơn","difficulty":"medium"},
    {"label":"confused","last_utterance":"App hiện đang giao nhưng ngoài kia báo đã hủy là sao?",
     "context_clues":"mâu thuẫn thông tin","difficulty":"medium"},
    {"label":"confused","last_utterance":"Ơ sao trừ tiền 2 lần vậy shop ơi?",
     "context_clues":"bối rối thanh toán","difficulty":"medium"},
    {"label":"confused","last_utterance":"Hệ thống báo đặt thành công nhưng không có email xác nhận?",
     "context_clues":"không có xác nhận","difficulty":"medium"},
    {"label":"confused","last_utterance":"Em đặt màu đen nhưng sao giao màu xanh vậy?",
     "context_clues":"nhầm sản phẩm","difficulty":"easy"},
    # anxious
    {"label":"anxious","last_utterance":"Tiền bị trừ rồi mà không thấy đơn đâu hết, làm sao giờ?",
     "context_clues":"lo mất tiền cần gấp","difficulty":"medium"},
    {"label":"anxious","last_utterance":"Em cần hàng gấp trước 5 giờ chiều có kịp không ạ?",
     "context_clues":"cần gấp deadline","difficulty":"medium"},
    {"label":"anxious","last_utterance":"Thanh toán rồi mà app cứ báo lỗi, tiền có bị mất không?",
     "context_clues":"lo lắng mất tiền","difficulty":"medium"},
    {"label":"anxious","last_utterance":"Đơn hàng không thấy cập nhật 2 ngày rồi, ổn không ạ?",
     "context_clues":"lo lắng đơn chờ lâu","difficulty":"medium"},
    {"label":"anxious","last_utterance":"Sắp đến hạn rồi mà hàng chưa về, chị ơi làm sao?",
     "context_clues":"deadline lo lắng","difficulty":"hard"},
    # frustrated
    {"label":"frustrated","last_utterance":"Ship lâu vl, 5 ngày rồi chưa thấy hàng đâu",
     "context_clues":"chờ lâu bực bội teencode","difficulty":"easy"},
    {"label":"frustrated","last_utterance":"Gọi điện mãi không ai bắt máy, chán ghê",
     "context_clues":"không liên lạc được","difficulty":"easy"},
    {"label":"frustrated","last_utterance":"Đây là lần thứ 3 tôi nhắn mà chưa ai trả lời",
     "context_clues":"nhiều lần không phản hồi","difficulty":"medium"},
    {"label":"frustrated","last_utterance":"Hàng bể mà ship hoài chưa chịu đổi cho tôi",
     "context_clues":"bể hàng chờ đổi","difficulty":"medium"},
    {"label":"frustrated","last_utterance":"Sao cứ hẹn rồi lại hẹn hoài vậy shop?",
     "context_clues":"hẹn liên tục bực bội","difficulty":"medium"},
    # disappointed
    {"label":"disappointed","last_utterance":"ừ thôi kệ đi, lần sau chắc không mua nữa",
     "context_clues":"bỏ cuộc phương ngữ Nam","difficulty":"hard"},
    {"label":"disappointed","last_utterance":"Thôi được rồi cảm ơn, không cần nữa đâu",
     "context_clues":"lịch sự giả bỏ cuộc","difficulty":"hard"},
    {"label":"disappointed","last_utterance":"Thôi kệ đi quá, mua chỗ khác vậy",
     "context_clues":"bỏ sang chỗ khác","difficulty":"hard"},
    {"label":"disappointed","last_utterance":"Đơn hàng bể hết trơn, thôi kệ luôn",
     "context_clues":"phương ngữ Nam thất vọng","difficulty":"hard"},
    {"label":"disappointed","last_utterance":"oke thôi, em hiểu rồi, không sao đâu",
     "context_clues":"lịch sự giả bỏ cuộc","difficulty":"hard"},
    # angry
    {"label":"angry","last_utterance":"Tôi sẽ post lên mạng xã hội cho mọi người biết!",
     "context_clues":"đe dọa đăng mạng","difficulty":"easy"},
    {"label":"angry","last_utterance":"Lần đầu mà cũng lần cuối luôn, tệ hết chỗ nói",
     "context_clues":"tuyên bố rời đi tức giận","difficulty":"medium"},
    {"label":"angry","last_utterance":"Tôi sẽ khiếu nại lên cơ quan chức năng!",
     "context_clues":"đe dọa khiếu nại pháp lý","difficulty":"easy"},
    {"label":"angry","last_utterance":"Các anh lừa dối khách hàng, tôi sẽ không để yên đâu",
     "context_clues":"tức giận đe dọa","difficulty":"easy"},
    {"label":"angry","last_utterance":"Trả lại tiền cho tôi ngay hoặc tôi báo công an",
     "context_clues":"yêu cầu mạnh đe dọa pháp lý","difficulty":"medium"},
]


# ─────────────────────────────────────────────────────────────
# SCENARIO GENERATOR
# ─────────────────────────────────────────────────────────────

SCENARIO_CACHE = CACHE_DIR / "eval_scenarios.json"

_LABEL_HINTS = {
    "neutral":      "Câu hỏi thông thường, không biểu lộ cảm xúc, chỉ muốn thông tin.",
    "happy":        "Hài lòng, vui, khen ngợi shop hoặc sản phẩm/dịch vụ.",
    "confused":     "Bối rối, không hiểu sự mâu thuẫn, cần giải thích rõ hơn.",
    "anxious":      "Lo lắng về tiền, đơn hàng, deadline — cần giải quyết gấp.",
    "frustrated":   "Bực bội vì chờ lâu, không được phản hồi, vấn đề tái diễn.",
    "disappointed": "NGUY HIỂM — Thất vọng ngầm, bỏ cuộc, dùng 'thôi kệ/ừ thôi', KHÔNG tức giận thẳng mặt. Phải có context phàn nàn trước đó.",
    "angry":        "Tức giận thẳng mặt, đe dọa post lên MXH, kiện tụng, đòi hoàn tiền ngay.",
}

_SCENARIO_EXAMPLE = """[
  {
    "turns": [
      {"role": "customer", "text": "ship lâu vl 5 ngày rồi"},
      {"role": "agent",    "text": "Dạ em xin lỗi ạ, em kiểm tra lại ngay"},
      {"role": "customer", "text": "ừ thôi kệ đi mua chỗ khác cho xong"}
    ],
    "label": "disappointed",
    "last_utterance": "ừ thôi kệ đi mua chỗ khác cho xong",
    "context_clues": ["bỏ cuộc", "phương ngữ Nam", "thôi kệ"],
    "difficulty": "hard"
  }
]"""


def generate_scenarios(skip_gen: bool = False) -> List[dict]:
    """Generate 49 test scenarios (7 × 7). Cache xuống disk."""
    if skip_gen and SCENARIO_CACHE.exists():
        data = json.loads(SCENARIO_CACHE.read_text(encoding="utf-8"))
        print(f"[GEN] ✓ Loaded {len(data)} cached scenarios")
        return data

    if SCENARIO_CACHE.exists() and not skip_gen:
        print("[GEN] Cache exists — delete .cache/eval_scenarios.json to re-generate")
        data = json.loads(SCENARIO_CACHE.read_text(encoding="utf-8"))
        if len(data) >= 40:
            print(f"[GEN] ✓ Using {len(data)} cached scenarios")
            return data

    all_scenarios: List[dict] = []
    print(f"[GEN] Generating {N_PER_LABEL * len(LABELS)} scenarios via LLM...")

    for label in LABELS:
        n = N_PER_LABEL + (1 if label == "disappointed" else 0)
        print(f"  → {label} ({n} scenarios)...", end=" ", flush=True)

        prompt = f"""Tạo {n} kịch bản hội thoại CSKH tiếng Việt, cảm xúc khách hàng là: "{label}"

MÔ TẢ: {_LABEL_HINTS[label]}

Yêu cầu:
- turns: 2-5 lượt (customer/agent xen kẽ), bắt đầu bằng customer
- last_utterance: câu CUỐI CÙNG của khách (phải là phần tử cuối trong turns)
- context_clues: 2-4 từ/cụm từ nhận biết cảm xúc
- difficulty: đa dạng easy/medium/hard
- Dùng đa dạng: có teencode (vl, vcl, ko, k), phương ngữ Nam, hoặc ngôn ngữ trang trọng

Ví dụ format đúng:
{_SCENARIO_EXAMPLE}

Trả về JSON ARRAY gồm {n} phần tử. KHÔNG thêm bất kỳ text nào ngoài JSON."""

        try:
            raw = call_llm([{"role": "user", "content": prompt}], temperature=0.8, max_tokens=3000)
            parsed = parse_json_safe(raw)

            if isinstance(parsed, list) and len(parsed) > 0:
                valid = []
                for s in parsed[:n]:
                    s["label"] = label  # enforce ground truth
                    if "turns" in s and "last_utterance" in s:
                        valid.append(s)
                all_scenarios.extend(valid)
                print(f"✓ {len(valid)}")
            else:
                print(f"✗ parse failed → fallback")
                all_scenarios.extend(_fallback_scenarios(label, n))
        except Exception as e:
            print(f"✗ error: {e} → fallback")
            all_scenarios.extend(_fallback_scenarios(label, n))

        time.sleep(1.2)  # tránh rate limit

    import random
    random.seed(42)
    random.shuffle(all_scenarios)

    SCENARIO_CACHE.write_text(json.dumps(all_scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[GEN] ✓ {len(all_scenarios)} scenarios saved → {SCENARIO_CACHE}")
    return all_scenarios


def _fallback_scenarios(label: str, n: int) -> list:
    """Fallback hardcoded scenarios nếu LLM generate thất bại."""
    templates = {
        "neutral":      [{"turns":[{"role":"customer","text":"Cho hỏi giá sản phẩm này bao nhiêu?"}],"last_utterance":"Cho hỏi giá sản phẩm này bao nhiêu?","context_clues":["hỏi giá"],"difficulty":"easy"}],
        "happy":        [{"turns":[{"role":"customer","text":"Shop ok lắm, hàng đẹp, giao nhanh!"}],"last_utterance":"Shop ok lắm, hàng đẹp, giao nhanh!","context_clues":["khen ngợi"],"difficulty":"easy"}],
        "confused":     [{"turns":[{"role":"customer","text":"Sao đơn báo giao rồi mà tôi không nhận?"}],"last_utterance":"Sao đơn báo giao rồi mà tôi không nhận?","context_clues":["bối rối"],"difficulty":"medium"}],
        "anxious":      [{"turns":[{"role":"customer","text":"Tiền trừ rồi mà không thấy đơn, sao vậy?"}],"last_utterance":"Tiền trừ rồi mà không thấy đơn, sao vậy?","context_clues":["lo mất tiền"],"difficulty":"medium"}],
        "frustrated":   [{"turns":[{"role":"customer","text":"Ship lâu vl mấy ngày rồi mà chưa có gì"}],"last_utterance":"Ship lâu vl mấy ngày rồi mà chưa có gì","context_clues":["chờ lâu","vl"],"difficulty":"easy"}],
        "disappointed": [{"turns":[{"role":"customer","text":"Thôi kệ, không cần nữa"},{"role":"customer","text":"Ừ thôi kệ đi, lần sau không mua nữa"}],"last_utterance":"Ừ thôi kệ đi, lần sau không mua nữa","context_clues":["bỏ cuộc"],"difficulty":"hard"}],
        "angry":        [{"turns":[{"role":"customer","text":"Tôi sẽ post lên mạng ngay bây giờ!"}],"last_utterance":"Tôi sẽ post lên mạng ngay bây giờ!","context_clues":["đe dọa"],"difficulty":"easy"}],
    }
    base = templates.get(label, templates["neutral"])
    result = []
    for i in range(n):
        s = dict(base[i % len(base)])
        s["label"] = label
        result.append(s)
    return result


# ─────────────────────────────────────────────────────────────
# SEARCH ENGINES
# ─────────────────────────────────────────────────────────────

class TFIDFSearch:
    """TF-IDF cosine similarity (mô phỏng vector search thuần túy)."""

    def __init__(self, docs: List[str], metas: List[dict]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        self.docs   = docs
        self.metas  = metas
        self._cos   = cos_sim
        self.vect   = TfidfVectorizer(
            analyzer="char_wb",       # character n-gram, phù hợp tiếng Việt không cần tokenize
            ngram_range=(2, 4),       # bigram → 4-gram char, bắt được dấu + từ ghép
            min_df=1,
            sublinear_tf=True,        # log TF để giảm dominance của từ lặp nhiều
        )
        self.matrix = self.vect.fit_transform(docs)

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        q_vec = self.vect.transform([query])
        sims  = self._cos(q_vec, self.matrix)[0]
        top   = np.argsort(sims)[::-1][:top_k]
        return [
            {**self.metas[i],
             "vector_score": round(float(sims[i]), 4),
             "doc_id": f"doc_{i:04d}"}
            for i in top if sims[i] > 0.01
        ]

    def get_sims(self, query: str) -> np.ndarray:
        q_vec = self.vect.transform([query])
        return self._cos(q_vec, self.matrix)[0]


class BM25Engine:
    """BM25Okapi keyword search."""

    def __init__(self, docs: List[str], metas: List[dict]):
        from rank_bm25 import BM25Okapi
        self.docs  = docs
        self.metas = metas
        self.bm25  = BM25Okapi([d.lower().split() for d in docs])

    def get_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.lower().split()))

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        scores = self.get_scores(query)
        top = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.metas[i],
             "bm25_score": round(float(scores[i]), 4),
             "doc_id": f"doc_{i:04d}"}
            for i in top if scores[i] > 0
        ]


def hybrid_rrf_search(
    query: str,
    tfidf: TFIDFSearch,
    bm25:  BM25Engine,
    top_k: int = 3,
    rrf_k: int = 60,
) -> List[dict]:
    """Hybrid: TF-IDF + BM25 + RRF fusion."""
    v_sims = tfidf.get_sims(query)
    b_sims = bm25.get_scores(query)

    v_top = np.argsort(v_sims)[::-1][: top_k * 2]
    b_top = np.argsort(b_sims)[::-1][: top_k * 2]

    rrf: Dict[int, float] = {}
    for rank, idx in enumerate(v_top):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)
    for rank, idx in enumerate(b_top):
        rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank)

    top_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
    return [
        {**tfidf.metas[i],
         "vector_score": round(float(v_sims[i]), 4),
         "bm25_score":   round(float(b_sims[i]), 4),
         "rrf_score":    round(rrf[i], 5),
         "doc_id":       f"doc_{i:04d}"}
        for i in top_ids
    ]


def load_kb() -> List[dict]:
    """
    Load knowledge base: merge HARDCODED_KB với all_vn.json nếu có.
    Ưu tiên tìm file theo thứ tự:
      1. all_vn.json (cùng thư mục script)
      2. data/all_vn.json
      3. Windows path từ OneDrive nếu chạy local
    """
    kb = list(HARDCODED_KB)
    existing_utts = {e["last_utterance"] for e in kb}

    search_paths = [
        Path(__file__).parent / "all_vn.json",
        Path("all_vn.json"),
        Path("data/all_vn.json"),
        Path(r"C:\Users\hieu2\OneDrive\Máy tính\NLP\Github\emotion_detection\data\all_vn.json"),
    ]
    for p in search_paths:
        try:
            if p.exists():
                ext_data = json.loads(p.read_text(encoding="utf-8"))
                added = [
                    e for e in ext_data
                    if isinstance(e, dict)
                    and e.get("last_utterance") not in existing_utts
                    and e.get("label") in set(LABELS)
                ]
                kb.extend(added)
                existing_utts.update(e["last_utterance"] for e in added)
                print(f"[KB] ✓ Loaded {len(added)} entries from {p} → total KB={len(kb)}")
                break
        except Exception as e:
            continue
    else:
        print(f"[KB] ⚠ all_vn.json not found, using {len(kb)} hardcoded entries only")
    return kb


def build_search_engines(kb: List[dict]) -> Tuple[TFIDFSearch, BM25Engine]:
    # Nối last_utterance + context_clues (xử lý cả list lẫn string)
    def _doc_text(d: dict) -> str:
        utt = d.get("last_utterance", "")
        clues = d.get("context_clues", "")
        if isinstance(clues, list):
            clues = " ".join(clues)
        return f"{utt} {clues}"

    docs  = [_doc_text(d) for d in kb]
    tfidf = TFIDFSearch(docs, kb)
    bm25  = BM25Engine(docs, kb)
    return tfidf, bm25


# ─────────────────────────────────────────────────────────────
# TOOLS (Agentic pipeline)
# ─────────────────────────────────────────────────────────────

def tool_context_analyzer(turns: list) -> dict:
    """
    Đếm neg_signals với 2 tier:
    - STRONG: 1 hit là đủ escalation (đe dọa, bỏ đi hẳn)
    - WEAK:   cần >= 3 hit mới là escalation (chờ lâu, bực nhẹ)

    FIX Bug #1: Bỏ điều kiện len(cust_msgs) >= 3 vì hội thoại dài
    không đồng nghĩa với escalation.
    """
    recent    = turns[-6:]
    cust_msgs = [t["text"] for t in recent if t.get("role") == "customer"]

    strong_hits = sum(1 for msg in cust_msgs for kw in NEG_STRONG if kw in msg.lower())
    weak_hits   = sum(1 for msg in cust_msgs for kw in NEG_WEAK   if kw in msg.lower())

    escalation_risk = strong_hits >= 1 or weak_hits >= 3

    return {
        "n_turns":         len(cust_msgs),
        "neg_signals":     strong_hits * 2 + weak_hits,   # weighted count
        "strong_signals":  strong_hits,
        "weak_signals":    weak_hits,
        "escalation_risk": escalation_risk,
        "recent_msgs":     cust_msgs,
        "summary":         (
            f"{len(cust_msgs)} customer turns | "
            f"strong={strong_hits} weak={weak_hits}"
            + (" | ⚠️ escalation" if escalation_risk else "")
        ),
    }


def tool_slang_lookup(text: str) -> dict:
    """Tra teencode / phương ngữ Nam."""
    found = {k: v for k, v in SLANG_DICT.items() if k in text.lower()}
    return {"found": found}


# ─────────────────────────────────────────────────────────────
# EMOTION PREDICTOR
# ─────────────────────────────────────────────────────────────

def predict_emotion_raw(prompt: str) -> dict:
    raw = call_llm([{"role": "user", "content": prompt}], temperature=0)
    raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
    raw = re.sub(r"[\u4e00-\u9fff]+", "", raw)
    m   = re.search(r"\{.*?\}", raw, re.DOTALL)
    try:
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


def format_conv(turns: list) -> str:
    lines = []
    for t in turns:
        role = "Khách" if t.get("role") == "customer" else "Agent"
        lines.append(f"{role}: {t['text']}")
    return "\n".join(lines)


def format_rag_context(docs: list, source: str = "") -> str:
    if not docs:
        return "Không có"
    lines = []
    for d in docs:
        label  = d.get("label", "?")
        utt    = d.get("last_utterance", "")
        clues  = d.get("context_clues", "")
        score  = d.get("rrf_score") or d.get("vector_score") or d.get("bm25_score") or 0
        lines.append(f'- "{utt}" → {label} | clues: {clues} | score={score:.4f}')
    prefix = f"[{source}] " if source else ""
    return prefix + "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 4 PIPELINES
# ─────────────────────────────────────────────────────────────

def pipeline_1_baseline(scenario: dict) -> dict:
    """Pipeline 1: LLM + fixed prompt, không RAG, không tool."""
    conv   = format_conv(scenario.get("turns", []))
    prompt = f"""{SYSTEM_PROMPT}

Hội thoại:
{conv}

Câu cần phân tích: "{scenario['last_utterance']}"
Chỉ trả về JSON."""
    result = predict_emotion_raw(prompt)
    result["_pipeline"] = "baseline"
    return result


def pipeline_2_vector_rag(scenario: dict, tfidf: TFIDFSearch) -> dict:
    """Pipeline 2: LLM + Vector-only RAG (TF-IDF cosine)."""
    query = scenario["last_utterance"]
    docs  = tfidf.search(query, top_k=3)
    conv  = format_conv(scenario.get("turns", []))

    prompt = f"""{SYSTEM_PROMPT}

Hội thoại:
{conv}

Ví dụ tương tự từ Knowledge Base (Vector Search):
{format_rag_context(docs, "Vector")}

Câu cần phân tích: "{query}"
Chỉ trả về JSON."""

    result = predict_emotion_raw(prompt)
    result["_pipeline"]  = "vector_rag"
    result["_retrieved"] = len(docs)
    return result


def pipeline_3_hybrid_rag(scenario: dict, tfidf: TFIDFSearch, bm25: BM25Engine) -> dict:
    """Pipeline 3: LLM + Hybrid RAG (TF-IDF + BM25 + RRF)."""
    query = scenario["last_utterance"]
    docs  = hybrid_rrf_search(query, tfidf, bm25, top_k=3)
    conv  = format_conv(scenario.get("turns", []))

    prompt = f"""{SYSTEM_PROMPT}

Hội thoại:
{conv}

Ví dụ tương tự từ Knowledge Base (Hybrid RAG = Vector + BM25 + RRF):
{format_rag_context(docs, "Hybrid")}

Câu cần phân tích: "{query}"
Chỉ trả về JSON."""

    result = predict_emotion_raw(prompt)
    result["_pipeline"]  = "hybrid_rag"
    result["_retrieved"] = len(docs)
    return result


def format_few_shot(docs: list) -> str:
    """
    Format top RAG hits thành few-shot examples thực sự (input → output),
    không chỉ là "tham khảo". Mỗi example có hội thoại rút gọn + label + lý do.
    """
    if not docs:
        return ""
    lines = ["Ví dụ thực tế đã được xác nhận (dùng để học pattern, KHÔNG copy label mù quáng):"]
    for i, d in enumerate(docs[:2], 1):
        utt   = d.get("last_utterance", "")
        label = d.get("label", "?")
        clues = d.get("context_clues", "")
        if isinstance(clues, list):
            clues = ", ".join(clues)
        score = d.get("vector_score") or d.get("rrf_score") or 0
        lines.append(
            f'  [{i}] Khách: "{utt}"\n'
            f'      → emotion: {label} | pattern: {clues} | sim={score:.2f}'
        )
    return "\n".join(lines)


def pipeline_4_agentic(scenario: dict, tfidf: TFIDFSearch, bm25: BM25Engine) -> dict:
    """
    Pipeline 4: Agentic v4 — CoT + Dynamic few-shot + Smart self-reflect.

    v4 Additions (trên v3):
    - Add 1: SYSTEM_PROMPT_AGENTIC với disambiguation rules + CoT "thought" field
    - Add 2: format_few_shot() — RAG hits thành proper few-shot examples
    - Add 3: _parse_cot() — extract emotion từ response có/không có "thought"
    - Add 4: Improved self-reflect dùng "thought" từ attempt 1 để build targeted prompt
    """
    turns = scenario.get("turns", [])
    query = scenario["last_utterance"]

    # ── Tool 1: Context Analyzer ───────────────────────────────
    ctx = tool_context_analyzer(turns)

    # ── Tool 2: Slang Lookup (context-aware) ──────────────────
    slang_raw = {k: v for k, v in SLANG_DICT.items() if k in query.lower()}
    praise_words = ["nhanh", "đẹp", "tốt", "ngon", "cảm ơn", "thích", "oke", "ok"]
    has_praise = any(w in query.lower() for w in praise_words)
    if has_praise:
        slang = {k: v for k, v in slang_raw.items() if k not in ("ừ thôi", "thôi kệ")}
    else:
        slang = slang_raw

    # ── Tool 3: Hybrid RAG — threshold 0.40, top-2 ────────────
    RAG_THRESHOLD = 0.40
    docs_raw = hybrid_rrf_search(query, tfidf, bm25, top_k=3)
    docs = [d for d in docs_raw if (d.get("vector_score") or 0) >= RAG_THRESHOLD]
    if not docs and docs_raw:
        docs = docs_raw[:1]

    if ctx["escalation_risk"]:
        cust_msgs = ctx["recent_msgs"]
        neg_msg = next(
            (m for m in reversed(cust_msgs)
             if any(kw in m.lower() for kw in NEG_STRONG + NEG_WEAK)),
            cust_msgs[-1] if cust_msgs else query
        )
        extra = hybrid_rrf_search(neg_msg, tfidf, bm25, top_k=2)
        extra = [d for d in extra if (d.get("vector_score") or 0) >= RAG_THRESHOLD]
        seen, merged = set(), []
        for d in docs + extra:
            key = d.get("doc_id", d.get("last_utterance", ""))
            if key not in seen:
                seen.add(key)
                merged.append(d)
        docs = merged[:2]

    # ── Build prompt ───────────────────────────────────────────
    conv      = format_conv(turns)
    slang_txt = "\n".join(f"  '{k}' = {v}" for k, v in slang.items()) or "  Không có"

    has_good_rag = bool(docs) and (docs[0].get("vector_score") or 0) >= RAG_THRESHOLD
    few_shot_section = ""
    if has_good_rag:
        few_shot_section = f"\n{format_few_shot(docs)}\n"

    prompt = f"""{SYSTEM_PROMPT_AGENTIC}

Hội thoại ({len(turns)} turns):
{conv}
{few_shot_section}
Từ lóng / phương ngữ: {slang_txt}
Ngữ cảnh: {ctx['summary']}

Câu cần phân tích: "{query}"
Trả về JSON với "thought" đầu tiên."""

    # ── Parse helper — xử lý cả response có/không có "thought" ─
    def _parse_cot(raw: str) -> dict:
        raw = re.sub(r"```json\s*|```\s*", "", raw).strip()
        raw = re.sub(r"[\u4e00-\u9fff]+", "", raw)
        m   = re.search(r"\{.*?\}", raw, re.DOTALL)
        try:
            obj = json.loads(m.group()) if m else {}
            # Đảm bảo có đủ fields dù model không trả "thought"
            obj.setdefault("thought", "")
            obj.setdefault("emotion", "unknown")
            obj.setdefault("confidence", 0.0)
            obj.setdefault("reason", "")
            obj.setdefault("alert", False)
            return obj
        except Exception:
            return {}

    # ── Attempt 1: temperature=0 với CoT ──────────────────────
    result   = _parse_cot(call_llm([{"role": "user", "content": prompt}], temperature=0))
    conf     = result.get("confidence", 0.0)
    emotion1 = result.get("emotion", "unknown")
    thought1 = result.get("thought", "")

    # ── Smart self-reflect — chỉ khi conf < 0.70 ──────────────
    CONFUSION_PAIRS: Dict[str, List[str]] = {
        "anxious":      ["confused",     "frustrated"],
        "frustrated":   ["anxious",      "angry"],
        "disappointed": ["neutral",      "frustrated"],
        "angry":        ["frustrated",   "disappointed"],
        "confused":     ["anxious",      "neutral"],
        "neutral":      ["confused",     "happy"],
        "happy":        ["neutral",      "confused"],
    }

    if conf < 0.70 and emotion1 in CONFUSION_PAIRS:
        alternatives = CONFUSION_PAIRS[emotion1]
        # Dùng "thought" từ attempt 1 làm context cho reflection
        thought_ctx = f'\nLần trước bạn đã nhận xét: "{thought1}"' if thought1 else ""
        clarify_prompt = f"""Hội thoại:
{conv}

Câu cần phân tích: "{query}"{thought_ctx}

Kết quả lần trước: {emotion1} (confidence={conf:.2f})

Kiểm tra lại: Tại sao KHÔNG phải "{alternatives[0]}" hay "{alternatives[1]}"?
Hãy phân biệt rõ rồi đưa ra kết luận cuối cùng.

Trả về JSON: {{"thought":"...","emotion":"...","confidence":0.0,"reason":"...","alert":true/false}}"""

        result2 = _parse_cot(
            call_llm([{"role": "user", "content": clarify_prompt}], temperature=0)
        )
        conf2 = result2.get("confidence", 0.0)
        if result2 and conf2 > conf + 0.05:
            result = result2

    if not result:
        result = {"thought": "", "emotion": "neutral", "confidence": 0.0,
                  "reason": "parse error", "alert": False}

    result["_pipeline"]    = "agentic"
    result["_retrieved"]   = len(docs)
    result["_neg_signals"] = ctx["neg_signals"]
    result["_slang_found"] = list(slang.keys())
    return result


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(ground_truth: list, predictions: list, latencies: list) -> dict:
    n = len(ground_truth)
    correct = sum(1 for g, p in zip(ground_truth, predictions) if g == p)

    # Per-class precision / recall / F1
    per_class = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(ground_truth, predictions) if g == label and p == label)
        fp = sum(1 for g, p in zip(ground_truth, predictions) if g != label and p == label)
        fn = sum(1 for g, p in zip(ground_truth, predictions) if g == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {
            "precision": round(prec, 3),
            "recall":    round(rec,  3),
            "f1":        round(f1,   3),
            "support":   sum(1 for g in ground_truth if g == label),
            "correct":   tp,
        }

    macro_f1      = float(np.mean([v["f1"]        for v in per_class.values()]))
    macro_prec    = float(np.mean([v["precision"]  for v in per_class.values()]))
    macro_recall  = float(np.mean([v["recall"]     for v in per_class.values()]))
    weighted_f1   = float(np.average(
        [v["f1"] for v in per_class.values()],
        weights=[v["support"] for v in per_class.values()]
    ))

    # Confusion matrix [actual × predicted]
    label_idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0] * len(LABELS) for _ in range(len(LABELS))]
    for g, p in zip(ground_truth, predictions):
        gi = label_idx.get(g, -1)
        pi = label_idx.get(p, -1)
        if gi >= 0 and pi >= 0:
            cm[gi][pi] += 1

    return {
        "accuracy":      round(correct / n, 4) if n > 0 else 0,
        "macro_f1":      round(macro_f1,     4),
        "macro_prec":    round(macro_prec,   4),
        "macro_recall":  round(macro_recall, 4),
        "weighted_f1":   round(weighted_f1,  4),
        "per_class":     per_class,
        "confusion_matrix": cm,
        "n_correct":     correct,
        "n_total":       n,
        "avg_latency_s": round(float(np.mean(latencies)), 2) if latencies else 0,
        "p50_latency_s": round(float(np.percentile(latencies, 50)), 2) if latencies else 0,
        "p95_latency_s": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
    }


# ─────────────────────────────────────────────────────────────
# HTML REPORT GENERATOR
# ─────────────────────────────────────────────────────────────

def generate_html_report(data: dict, output_path: Path = Path("eval_report.html")):
    """Tạo report HTML trực quan với Chart.js."""

    metrics  = data["metrics"]
    gt       = data["ground_truth"]
    preds    = data["pipeline_predictions"]
    now_str  = data.get("timestamp", "")
    n_total  = len(gt)

    pipe_names = {
        "baseline":   "1. Baseline (LLM only)",
        "vector_rag": "2. Vector RAG (TF-IDF)",
        "hybrid_rag": "3. Hybrid RAG (TF-IDF+BM25+RRF)",
        "agentic":    "4. Agentic (Hybrid+Tools+Self-reflect)",
    }
    pipe_colors = {
        "baseline":   "#94a3b8",
        "vector_rag": "#60a5fa",
        "hybrid_rag": "#34d399",
        "agentic":    "#f97316",
    }
    emotion_colors = {
        "neutral": "#64748b", "happy": "#22c55e", "confused": "#a855f7",
        "anxious": "#eab308", "frustrated": "#f97316",
        "disappointed": "#ef4444", "angry": "#dc2626",
    }
    emotion_icons = {
        "neutral":"😐","happy":"😊","confused":"🤔",
        "anxious":"😰","frustrated":"😤","disappointed":"😢","angry":"😠",
    }

    # Accuracy bars data
    pipe_keys = list(metrics.keys())
    accuracies = [round(metrics[k]["accuracy"] * 100, 1) for k in pipe_keys]
    macro_f1s  = [round(metrics[k]["macro_f1"]  * 100, 1) for k in pipe_keys]
    avg_lat    = [metrics[k]["avg_latency_s"] for k in pipe_keys]

    # Per-class F1 per pipeline
    per_class_js = {}
    for pipe in pipe_keys:
        per_class_js[pipe] = {lbl: metrics[pipe]["per_class"][lbl]["f1"] for lbl in LABELS}

    # Confusion matrix for agentic (best pipeline)
    best_pipe   = max(metrics, key=lambda k: metrics[k]["accuracy"])
    cm          = metrics[best_pipe]["confusion_matrix"]
    cm_max      = max(max(row) for row in cm) or 1

    # Scenario details table
    scenarios_json = json.dumps(data.get("scenarios", []), ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CSKH EmotionAI — Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --surface2: #334155;
    --border: #475569; --text: #f1f5f9; --muted: #94a3b8;
    --accent: #f97316; --green: #22c55e; --red: #ef4444;
    --blue: #60a5fa; --purple: #a855f7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: var(--muted); margin-bottom: 20px; }}
  h3 {{ font-size: 1rem; font-weight: 600; color: var(--text); margin-bottom: 12px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
  .grid-2 {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }}
  .grid-3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .metric-card {{ text-align: center; }}
  .metric-card .value {{ font-size: 2.5rem; font-weight: 800; line-height: 1; }}
  .metric-card .label {{ font-size: 0.8rem; color: var(--muted); margin-top: 6px; }}
  .metric-card .sub {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; }}
  .badge-best {{ background: #16a34a22; color: var(--green); border: 1px solid #16a34a44; }}
  .badge-worst {{ background: #dc262622; color: var(--red); border: 1px solid #dc262644; }}
  canvas {{ max-width: 100%; }}
  .cm-grid {{ display: grid; gap: 2px; }}
  .cm-cell {{ width: 100%; aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
              font-size: 0.7rem; font-weight: 700; border-radius: 4px; }}
  .cm-label {{ font-size: 0.65rem; color: var(--muted); text-align: center; overflow: hidden; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  th {{ background: var(--surface2); padding: 8px 12px; text-align: left; color: var(--muted); font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:hover td {{ background: #ffffff08; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }}
  .correct {{ background: #16a34a22; color: var(--green); }}
  .wrong   {{ background: #dc262622; color: var(--red); }}
  select, input {{ background: var(--surface2); color: var(--text); border: 1px solid var(--border);
                   border-radius: 6px; padding: 6px 10px; font-size: 0.8rem; }}
  .section-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }}
  .divider {{ border: none; border-top: 1px solid var(--border); margin: 24px 0; }}
  .highlight {{ color: var(--accent); font-weight: 700; }}
  .tag {{ display: inline-block; background: var(--surface2); border: 1px solid var(--border);
          border-radius: 4px; padding: 1px 6px; font-size: 0.68rem; color: var(--muted); margin: 1px; }}
  @media (max-width: 900px) {{
    .grid-4 {{ grid-template-columns: repeat(2, 1fr); }}
    .grid-2 {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <div style="margin-bottom: 32px;">
    <h1>🤖 CSKH EmotionAI — Evaluation Report</h1>
    <h2>So sánh 4 pipeline phân tích cảm xúc khách hàng tiếng Việt</h2>
    <div style="color: var(--muted); font-size: 0.8rem;">{now_str} &nbsp;·&nbsp; {n_total} test scenarios &nbsp;·&nbsp; Model: {LLM_MODEL}</div>
  </div>

  <!-- SUMMARY METRICS -->
  <div class="grid-4">
"""
    # Best pipeline highlight cards
    best_acc = max(accuracies)
    for i, pipe in enumerate(pipe_keys):
        acc   = accuracies[i]
        f1    = macro_f1s[i]
        lat   = avg_lat[i]
        color = pipe_colors.get(pipe, "#fff")
        is_b  = acc == best_acc
        html += f"""
    <div class="card metric-card" style="border-color: {color}44;">
      <div class="value" style="color:{color}">{acc}%</div>
      <div class="label">{pipe_names.get(pipe, pipe)}</div>
      <div class="sub">F1={f1}% &nbsp; Lat={lat}s</div>
      {"<span class='badge badge-best'>🏆 Best</span>" if is_b else ""}
    </div>"""

    html += """
  </div>

  <!-- MAIN CHARTS -->
  <div class="grid-2">
    <div class="card">
      <h3>📊 Accuracy & Macro F1 so sánh</h3>
      <canvas id="barChart" height="220"></canvas>
    </div>
    <div class="card">
      <h3>⏱ Latency (avg per scenario)</h3>
      <canvas id="latChart" height="220"></canvas>
    </div>
  </div>

  <!-- PER CLASS F1 -->
  <div class="card" style="margin-bottom: 24px;">
    <div class="section-header">
      <h3>🎯 F1 score theo nhãn cảm xúc</h3>
      <select id="pipeSelect" onchange="updateF1Chart()">
"""
    for p in pipe_keys:
        html += f'        <option value="{p}">{pipe_names.get(p, p)}</option>\n'

    html += f"""      </select>
    </div>
    <canvas id="f1Chart" height="120"></canvas>
  </div>

  <!-- CONFUSION MATRIX -->
  <div class="card" style="margin-bottom: 24px;">
    <h3>🔲 Confusion Matrix — {pipe_names.get(best_pipe, best_pipe)} (Best Pipeline)</h3>
    <div id="cmWrap" style="margin-top: 12px; overflow-x: auto;"></div>
  </div>

  <hr class="divider">

  <!-- SCENARIO TABLE -->
  <div class="card">
    <div class="section-header">
      <h3>📋 Chi tiết từng kịch bản</h3>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <select id="filterLabel" onchange="filterTable()">
          <option value="">Tất cả nhãn</option>
"""
    for lbl in LABELS:
        html += f'          <option value="{lbl}">{emotion_icons.get(lbl,"")} {lbl}</option>\n'

    html += """        </select>
        <select id="filterPipe" onchange="filterTable()">
          <option value="">Tất cả pipeline</option>
"""
    for p in pipe_keys:
        html += f'          <option value="{p}">{pipe_names.get(p,p)}</option>\n'

    html += """        </select>
        <select id="filterResult" onchange="filterTable()">
          <option value="">Tất cả kết quả</option>
          <option value="correct">✓ Đúng</option>
          <option value="wrong">✗ Sai</option>
        </select>
        <input type="text" id="searchText" placeholder="🔍 Tìm kiếm..." oninput="filterTable()" style="width:180px">
      </div>
    </div>
    <div id="tableCount" style="color:var(--muted);font-size:0.78rem;margin-bottom:8px;"></div>
    <div style="overflow-x:auto;">
      <table id="scenarioTable">
        <thead>
          <tr>
            <th>#</th>
            <th>Kịch bản (last_utterance)</th>
            <th>Nhãn đúng</th>
            <th>P1: Baseline</th>
            <th>P2: Vector RAG</th>
            <th>P3: Hybrid RAG</th>
            <th>P4: Agentic</th>
            <th>Khó</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </div>

  <!-- PER CLASS DETAIL TABLE -->
  <div class="card" style="margin-top: 24px;">
    <h3>📈 Chi tiết per-class metrics</h3>
    <div style="overflow-x:auto; margin-top:12px;">
      <table>
        <thead>
          <tr>
            <th>Emotion</th>
            <th>Support</th>
"""
    for p in pipe_keys:
        html += f'            <th colspan="3">{pipe_names.get(p,p)}</th>\n'
    html += """          </tr>
          <tr>
            <th></th><th></th>
"""
    for _ in pipe_keys:
        html += "            <th>Prec</th><th>Rec</th><th>F1</th>\n"
    html += "          </tr>\n        </thead>\n        <tbody>\n"

    for lbl in LABELS:
        icon = emotion_icons.get(lbl, "")
        sup  = metrics[pipe_keys[0]]["per_class"][lbl]["support"]
        html += f"          <tr>\n            <td>{icon} {lbl}</td>\n            <td>{sup}</td>\n"
        for p in pipe_keys:
            pc   = metrics[p]["per_class"][lbl]
            prec = pc["precision"]
            rec  = pc["recall"]
            f1   = pc["f1"]
            html += f"            <td>{prec:.2f}</td><td>{rec:.2f}</td><td><b>{f1:.2f}</b></td>\n"
        html += "          </tr>\n"

    html += """        </tbody>
      </table>
    </div>
  </div>

</div><!-- /container -->

<script>
// ── DATA ──────────────────────────────────────
const LABELS     = """ + json.dumps(LABELS) + """;
const PIPE_KEYS  = """ + json.dumps(pipe_keys) + """;
const PIPE_NAMES = """ + json.dumps({p: pipe_names.get(p,p) for p in pipe_keys}) + """;
const PIPE_COLORS= """ + json.dumps({p: pipe_colors.get(p,"#fff") for p in pipe_keys}) + """;
const EM_ICONS   = """ + json.dumps(emotion_icons) + """;
const EM_COLORS  = """ + json.dumps(emotion_colors) + """;

const ACCURACIES   = """ + json.dumps(accuracies) + """;
const MACRO_F1S    = """ + json.dumps(macro_f1s)  + """;
const AVG_LATENCY  = """ + json.dumps(avg_lat)    + """;
const PER_CLASS_JS = """ + json.dumps(per_class_js) + """;

const GT_LABELS    = """ + json.dumps(gt) + """;
const PREDICTIONS  = """ + json.dumps({k: preds.get(k, []) for k in pipe_keys}) + """;
const SCENARIOS    = """ + scenarios_json + """;

const CM_LABELS  = LABELS;
const CM_MATRIX  = """ + json.dumps(cm) + """;
const CM_MAX     = """ + str(cm_max) + """;
const BEST_PIPE  = """ + json.dumps(best_pipe) + """;

// ── BAR CHART ─────────────────────────────────
new Chart(document.getElementById('barChart'), {
  type: 'bar',
  data: {
    labels: PIPE_KEYS.map(p => PIPE_NAMES[p]),
    datasets: [
      { label: 'Accuracy %', data: ACCURACIES, backgroundColor: PIPE_KEYS.map(p=>PIPE_COLORS[p]+'cc'), borderRadius: 6 },
      { label: 'Macro F1 %', data: MACRO_F1S,  backgroundColor: PIPE_KEYS.map(p=>PIPE_COLORS[p]+'55'), borderRadius: 6 },
    ]
  },
  options: {
    responsive: true, plugins: { legend: { labels: { color:'#94a3b8' } } },
    scales: { x: { ticks:{color:'#94a3b8'}, grid:{color:'#334155'} }, y: { ticks:{color:'#94a3b8'}, grid:{color:'#334155'}, max:100, min:0 } }
  }
});

// ── LATENCY CHART ─────────────────────────────
new Chart(document.getElementById('latChart'), {
  type: 'bar',
  data: {
    labels: PIPE_KEYS.map(p => PIPE_NAMES[p]),
    datasets: [{ label: 'Avg latency (s)', data: AVG_LATENCY, backgroundColor: PIPE_KEYS.map(p=>PIPE_COLORS[p]+'cc'), borderRadius: 6 }]
  },
  options: {
    responsive: true, plugins: { legend: { labels: { color:'#94a3b8' } } },
    scales: { x: { ticks:{color:'#94a3b8'}, grid:{color:'#334155'} }, y: { ticks:{color:'#94a3b8'}, grid:{color:'#334155'} } }
  }
});

// ── F1 CHART ──────────────────────────────────
let f1Chart = null;
function updateF1Chart() {
  const pipe = document.getElementById('pipeSelect').value;
  const vals = LABELS.map(l => Math.round(PER_CLASS_JS[pipe][l] * 100));
  if (f1Chart) f1Chart.destroy();
  f1Chart = new Chart(document.getElementById('f1Chart'), {
    type: 'bar',
    data: {
      labels: LABELS.map(l => EM_ICONS[l] + ' ' + l),
      datasets: [{ label: 'F1 %', data: vals, backgroundColor: LABELS.map(l=>EM_COLORS[l]+'cc'), borderRadius: 6 }]
    },
    options: {
      responsive: true, plugins: { legend: { display: false } },
      scales: { x:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}, y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'},max:100,min:0} }
    }
  });
}
updateF1Chart();

// ── CONFUSION MATRIX ──────────────────────────
function renderCM() {
  const wrap = document.getElementById('cmWrap');
  const short = ['neu','hap','conf','anx','fru','dis','ang'];
  const n = LABELS.length;
  const cellSize = 56;
  let html = `<div style="display:grid;grid-template-columns:60px ${Array(n).fill(cellSize+'px').join(' ')};gap:2px;">`;
  // top labels
  html += '<div></div>';
  for (let j = 0; j < n; j++) {
    html += `<div style="font-size:0.65rem;color:#94a3b8;text-align:center;padding-bottom:4px;">${EM_ICONS[LABELS[j]]}<br>${short[j]}</div>`;
  }
  for (let i = 0; i < n; i++) {
    html += `<div style="font-size:0.65rem;color:#94a3b8;display:flex;align-items:center;justify-content:flex-end;padding-right:6px;">${EM_ICONS[LABELS[i]]} ${short[i]}</div>`;
    for (let j = 0; j < n; j++) {
      const val   = CM_MATRIX[i][j];
      const alpha = val / CM_MAX;
      const bg    = i === j ? `rgba(34,197,94,${0.15 + alpha * 0.7})` : val > 0 ? `rgba(239,68,68,${0.1 + alpha * 0.6})` : 'rgba(255,255,255,0.04)';
      html += `<div style="background:${bg};width:${cellSize}px;height:${cellSize}px;display:flex;align-items:center;justify-content:center;border-radius:4px;font-size:0.8rem;font-weight:700;">${val}</div>`;
    }
  }
  html += '</div>';
  html += '<div style="margin-top:8px;font-size:0.72rem;color:#64748b;">Hàng = nhãn thực (actual) &nbsp;·&nbsp; Cột = nhãn dự đoán (predicted) &nbsp;·&nbsp; Ô xanh = đúng &nbsp;·&nbsp; Ô đỏ = sai</div>';
  wrap.innerHTML = html;
}
renderCM();

// ── SCENARIO TABLE ────────────────────────────
function renderTable(rows) {
  const tbody = document.getElementById('tableBody');
  const count = document.getElementById('tableCount');
  count.textContent = `Hiển thị ${rows.length} hàng`;
  tbody.innerHTML = rows.map(r => {
    const preds = PIPE_KEYS.map(p => {
      const pred = PREDICTIONS[p][r.idx] || '?';
      const ok   = pred === GT_LABELS[r.idx];
      return `<td><span class="pill ${ok?'correct':'wrong'}">${ok?'✓':'✗'} ${pred}</span></td>`;
    }).join('');
    const diff = SCENARIOS[r.idx]?.difficulty || '';
    return `<tr>
      <td style="color:#64748b">${r.idx+1}</td>
      <td style="max-width:280px;">${escHtml(SCENARIOS[r.idx]?.last_utterance || '')}</td>
      <td><span class="pill" style="background:${EM_COLORS[GT_LABELS[r.idx]]+'33'};color:${EM_COLORS[GT_LABELS[r.idx]]}">${EM_ICONS[GT_LABELS[r.idx]]} ${GT_LABELS[r.idx]}</span></td>
      ${preds}
      <td><span class="tag">${diff}</span></td>
    </tr>`;
  }).join('');
}

function filterTable() {
  const fl = document.getElementById('filterLabel').value;
  const fp = document.getElementById('filterPipe').value;
  const fr = document.getElementById('filterResult').value;
  const fs = document.getElementById('searchText').value.toLowerCase();
  const rows = GT_LABELS.map((lbl, idx) => ({idx, lbl})).filter(r => {
    if (fl && r.lbl !== fl) return false;
    const utt = (SCENARIOS[r.idx]?.last_utterance || '').toLowerCase();
    if (fs && !utt.includes(fs)) return false;
    if (fp) {
      const pred = PREDICTIONS[fp][r.idx] || '?';
      const ok   = pred === r.lbl;
      if (fr === 'correct' && !ok) return false;
      if (fr === 'wrong'   &&  ok) return false;
    }
    if (!fp && fr) {
      const anyWrong  = PIPE_KEYS.some(p => (PREDICTIONS[p][r.idx] || '?') !== r.lbl);
      const allCorrect = PIPE_KEYS.every(p => (PREDICTIONS[p][r.idx] || '?') === r.lbl);
      if (fr === 'correct' && !allCorrect) return false;
      if (fr === 'wrong'   && !anyWrong)   return false;
    }
    return true;
  });
  renderTable(rows);
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

filterTable(); // initial render
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"[RPT] ✓ HTML report → {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def _progress_bar(done: int, total: int, width: int = 30) -> str:
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


def _print_comparison(results: dict, ground_truth: list):
    """In bảng so sánh baseline vs agentic sạch, không log thừa."""
    b = results.get("baseline")
    a = results.get("agentic")
    if not b or not a:
        return

    bm = b["metrics"]
    am = a["metrics"]

    LABELS_ORDER = ["neutral", "happy", "confused", "anxious", "frustrated", "disappointed", "angry"]
    ICONS = {"neutral":"😐","happy":"😊","confused":"🤔","anxious":"😰",
             "frustrated":"😤","disappointed":"😢","angry":"😠"}

    # ── Header ──────────────────────────────────
    print()
    print("=" * 62)
    print("  📊  BASELINE  vs  AGENTIC  —  50 scenarios")
    print("=" * 62)

    # ── Overall ─────────────────────────────────
    def delta(v_a, v_b, fmt=".1%"):
        d = v_a - v_b
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:{fmt}}"

    print(f"\n  {'Metric':<18} {'Baseline':>10}  {'Agentic':>10}  {'Δ':>8}")
    print(f"  {'-'*50}")
    rows = [
        ("Accuracy",    bm["accuracy"],    am["accuracy"],    ".1%"),
        ("Macro F1",    bm["macro_f1"],    am["macro_f1"],    ".3f"),
        ("Weighted F1", bm["weighted_f1"], am["weighted_f1"], ".3f"),
        ("Avg latency", bm["avg_latency_s"], am["avg_latency_s"], ".1f"),
    ]
    for name, bv, av, fmt in rows:
        unit = "s" if "lat" in name.lower() else ""
        better = av > bv if "lat" not in name.lower() else av < bv
        marker = "▲" if better else ("▼" if av != bv else "=")
        print(f"  {name:<18} {bv:>9{fmt}}{unit}  {av:>9{fmt}}{unit}  {marker} {delta(av, bv, fmt)}{unit}")

    # ── Per-class F1 ────────────────────────────
    print(f"\n  {'Emotion':<16} {'Base-F1':>8}  {'Agen-F1':>8}  {'Δ':>7}  {'Winner':>8}")
    print(f"  {'-'*54}")
    for lbl in LABELS_ORDER:
        bf1 = bm["per_class"].get(lbl, {}).get("f1", 0)
        af1 = am["per_class"].get(lbl, {}).get("f1", 0)
        d   = af1 - bf1
        sign = "+" if d >= 0 else ""
        winner = "Agentic" if af1 > bf1 else ("Baseline" if bf1 > af1 else "Tie")
        icon = ICONS.get(lbl, "")
        print(f"  {icon} {lbl:<14} {bf1:>8.3f}  {af1:>8.3f}  {sign}{d:>+6.3f}  {winner:>8}")

    # ── Error analysis: cases where they disagree ──
    b_preds = b["predictions"]
    a_preds = a["predictions"]
    disagree = [(i, ground_truth[i], b_preds[i], a_preds[i])
                for i in range(len(ground_truth))
                if b_preds[i] != a_preds[i]]

    b_wins = [(i, gt, bp, ap) for i, gt, bp, ap in disagree if bp == gt and ap != gt]
    a_wins = [(i, gt, bp, ap) for i, gt, bp, ap in disagree if ap == gt and bp != gt]
    both_wrong = [(i, gt, bp, ap) for i, gt, bp, ap in disagree if bp != gt and ap != gt]

    print(f"\n  ── Disagreement analysis ({len(disagree)} cases) ──────────────")
    print(f"  Agentic đúng, Baseline sai : {len(a_wins):>3}  ← Agentic gains")
    print(f"  Baseline đúng, Agentic sai : {len(b_wins):>3}  ← Agentic regressions")
    print(f"  Cả hai đều sai             : {len(both_wrong):>3}")

    if a_wins:
        print(f"\n  Top Agentic gains (gt → baseline✗ / agentic✓):")
        for i, gt, bp, ap in a_wins[:5]:
            print(f"    #{i+1:02d}  [{gt}]  base={bp}  agen={ap}")

    if b_wins:
        print(f"\n  Top Agentic regressions (gt → baseline✓ / agentic✗):")
        for i, gt, bp, ap in b_wins[:5]:
            print(f"    #{i+1:02d}  [{gt}]  base={bp}  agen={ap}")

    print()


def main():
    parser = argparse.ArgumentParser(description="CSKH EmotionAI Evaluation Framework")
    parser.add_argument("--skip-gen",       action="store_true", help="Dùng cached scenarios")
    parser.add_argument("--pipelines",      default="1,2,3,4",   help="Pipeline cần chạy, VD: 1,4")
    parser.add_argument("--max-scenarios",  type=int, default=0,  help="Giới hạn số scenarios (0=all)")
    args = parser.parse_args()

    active_pipes = set(args.pipelines.split(","))
    pipe_map = {
        "1": ("baseline",   lambda s, _t, _b: pipeline_1_baseline(s)),
        "2": ("vector_rag", lambda s, t,  _b: pipeline_2_vector_rag(s, t)),
        "3": ("hybrid_rag", lambda s, t,   b: pipeline_3_hybrid_rag(s, t, b)),
        "4": ("agentic",    lambda s, t,   b: pipeline_4_agentic(s, t, b)),
    }

    print("=" * 62)
    print("  CSKH EmotionAI — Evaluation Framework")
    print(f"  Model : {LLM_MODEL}")
    print("=" * 62)

    scenarios = generate_scenarios(skip_gen=args.skip_gen)
    if args.max_scenarios > 0:
        scenarios = scenarios[: args.max_scenarios]

    ground_truth = [s["label"] for s in scenarios]
    total = len(scenarios)

    tfidf, bm25 = build_search_engines(load_kb())

    results: Dict[str, dict] = {}

    for pipe_id, (pipe_name, pipe_fn) in pipe_map.items():
        if pipe_id not in active_pipes:
            continue

        print(f"\n  Running P{pipe_id} [{pipe_name}] ...", end=" ", flush=True)
        preds, lats, errors = [], [], []

        for i, scenario in enumerate(scenarios):
            # Compact progress — overwrite same line
            bar = _progress_bar(i + 1, total)
            print(f"\r  P{pipe_id} [{pipe_name}] {bar}", end="", flush=True)
            t0 = time.time()
            try:
                result = pipe_fn(scenario, tfidf, bm25)
                pred   = result.get("emotion") or "unknown"
            except Exception as e:
                pred, result = "unknown", {}
                errors.append(f"#{i+1}: {e}")
            lat = time.time() - t0
            preds.append(pred)
            lats.append(lat)

        metrics = compute_metrics(ground_truth, preds, lats)
        results[pipe_name] = {"predictions": preds, "latencies": lats, "metrics": metrics}

        correct = metrics["n_correct"]
        print(f"\r  P{pipe_id} [{pipe_name}] done — {correct}/{total} correct ({metrics['accuracy']:.1%})  avg {metrics['avg_latency_s']}s/call")
        if errors:
            print(f"     ⚠ {len(errors)} errors: {errors[0]}")

    # 4. Print comparison (baseline vs agentic)
    _print_comparison(results, ground_truth)

    # 5. Save + Report
    output_data = {
        "timestamp":            time.strftime("%Y-%m-%d %H:%M:%S"),
        "model":                LLM_MODEL,
        "scenarios":            scenarios,
        "ground_truth":         ground_truth,
        "pipeline_predictions": {k: v["predictions"] for k, v in results.items()},
        "metrics":              {k: v["metrics"] for k, v in results.items()},
    }

    RESULT_PATH = CACHE_DIR / "eval_results.json"
    RESULT_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SAVE] ✓ Results → {RESULT_PATH}")

    if len(results) >= 1:
        generate_html_report(output_data)

    return output_data


if __name__ == "__main__":
    main()