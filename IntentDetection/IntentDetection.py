# ════════════════════════════════════════════════
# CELL 1 — Cài đặt (SQLite có sẵn trong Python)
# ════════════════════════════════════════════════
import sqlite3
import json
import re
from datetime import datetime
from pathlib import Path

DB_PATH = "orders.db"
print(f" SQLite path: {Path(DB_PATH).resolve()}")

# ════════════════════════════════════════════════
# CELL 2 — Tạo Database + Schema
# ════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # trả về dict thay vì tuple
    cur  = conn.cursor()

    # Bảng đơn hàng
    cur.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id          TEXT PRIMARY KEY,
        customer    TEXT NOT NULL,
        phone       TEXT NOT NULL,
        total       INTEGER NOT NULL,
        status      TEXT NOT NULL DEFAULT 'pending',
        created_at  TEXT NOT NULL,
        eta         TEXT,
        address     TEXT,
        shipper     TEXT
    )""")

    # Bảng chi tiết sản phẩm
    cur.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id    TEXT NOT NULL,
        name        TEXT NOT NULL,
        qty         INTEGER NOT NULL,
        price       INTEGER NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id)
    )""")

    # Index để tìm theo phone nhanh hơn
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_phone ON orders(phone)
    """)

    conn.commit()
    conn.close()
    print(" Database initialized")

init_db()

# ════════════════════════════════════════════════
# CELL 3 — Seed mock data
# ════════════════════════════════════════════════

MOCK_ORDERS = [
    {
        "id": "DH001", "customer": "Nguyễn Văn An",
        "phone": "0901234567", "total": 300000,
        "status": "delivering", "created_at": "2024-05-10",
        "eta": "2024-05-13", "address": "123 Lê Lợi, Q1, TP.HCM",
        "shipper": "Giao Hàng Nhanh",
        "items": [{"name": "Áo thun nam", "qty": 2, "price": 150000}],
    },
    {
        "id": "DH002", "customer": "Trần Thị Bình",
        "phone": "0912345678", "total": 850000,
        "status": "delivered", "created_at": "2024-05-08",
        "eta": "2024-05-11", "address": "456 Nguyễn Huệ, Q1, TP.HCM",
        "shipper": "GHTK",
        "items": [{"name": "Váy lụa", "qty": 1, "price": 850000}],
    },
    {
        "id": "DH003", "customer": "Lê Văn Cường",
        "phone": "0923456789", "total": 2500000,
        "status": "pending", "created_at": "2024-05-12",
        "eta": "2024-05-15", "address": "789 Trần Hưng Đạo, Q5, TP.HCM",
        "shipper": None,
        "items": [{"name": "Máy lọc nước", "qty": 1, "price": 2500000}],
    },
    {
        "id": "DH004", "customer": "Phạm Thị Dung",
        "phone": "0934567890", "total": 1200000,
        "status": "cancelled", "created_at": "2024-05-09",
        "eta": None, "address": "321 Đinh Tiên Hoàng, Q.Bình Thạnh",
        "shipper": None,
        "items": [{"name": "Giày sneaker", "qty": 1, "price": 1200000}],
    },
    {
        "id": "DH005", "customer": "Hoàng Minh Tuấn",
        "phone": "0945678901", "total": 450000,
        "status": "confirmed", "created_at": "2024-05-12",
        "eta": "2024-05-14", "address": "567 CMT8, Q.3, TP.HCM",
        "shipper": "Viettel Post",
        "items": [
            {"name": "Quần jean",  "qty": 1, "price": 350000},
            {"name": "Thắt lưng", "qty": 1, "price": 100000},
        ],
    },
]

def seed_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    for o in MOCK_ORDERS:
        # Insert order (ignore nếu đã có)
        cur.execute("""
        INSERT OR IGNORE INTO orders
            (id, customer, phone, total, status, created_at, eta, address, shipper)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, (o["id"], o["customer"], o["phone"], o["total"],
              o["status"], o["created_at"], o["eta"],
              o["address"], o["shipper"]))

        # Insert items
        for item in o["items"]:
            cur.execute("""
            INSERT OR IGNORE INTO order_items (order_id, name, qty, price)
            SELECT ?,?,?,? WHERE NOT EXISTS (
                SELECT 1 FROM order_items WHERE order_id=? AND name=?
            )
            """, (o["id"], item["name"], item["qty"], item["price"],
                  o["id"], item["name"]))

    conn.commit()
    conn.close()
    print(f" Seeded {len(MOCK_ORDERS)} orders")

seed_db()

# ════════════════════════════════════════════════
# CELL 4 — DB Helper class
# ════════════════════════════════════════════════

class OrderDB:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_by_id(self, order_id: str) -> dict | None:
        """Tra cứu theo mã đơn"""
        conn = self._connect()
        cur  = conn.cursor()

        # Query join orders + items
        order = cur.execute("""
            SELECT * FROM orders WHERE id = ?
        """, (order_id.upper(),)).fetchone()

        if not order:
            conn.close()
            return None

        items = cur.execute("""
            SELECT name, qty, price FROM order_items WHERE order_id = ?
        """, (order_id.upper(),)).fetchall()

        conn.close()
        return self._format(dict(order), [dict(i) for i in items])

    def get_by_phone(self, phone: str) -> list[dict]:
        """Tra cứu theo số điện thoại — trả về nhiều đơn"""
        conn  = self._connect()
        cur   = conn.cursor()

        orders = cur.execute("""
            SELECT * FROM orders
            WHERE phone = ?
            ORDER BY created_at DESC
            LIMIT 5
        """, (phone,)).fetchall()

        results = []
        for order in orders:
            items = cur.execute("""
                SELECT name, qty, price FROM order_items
                WHERE order_id = ?
            """, (order["id"],)).fetchall()
            results.append(self._format(dict(order), [dict(i) for i in items]))

        conn.close()
        return results

    def get_all(self) -> list[dict]:
        """Lấy tất cả đơn — dùng để debug"""
        conn   = self._connect()
        cur    = conn.cursor()
        orders = cur.execute("SELECT id, customer, status, total FROM orders").fetchall()
        conn.close()
        return [dict(o) for o in orders]

    def _format(self, order: dict, items: list) -> dict:
        """Chuẩn hóa output"""
        STATUS_MAP = {
            "pending":    {"label": "Chờ xác nhận",  "icon": "⏳"},
            "confirmed":  {"label": "Đã xác nhận",   "icon": ""},
            "delivering": {"label": "Đang giao",      "icon": "🚚"},
            "delivered":  {"label": "Đã giao",        "icon": "📦"},
            "cancelled":  {"label": "Đã huỷ",         "icon": "❌"},
        }
        status_info = STATUS_MAP.get(order["status"], {"label": order["status"], "icon": "❓"})
        items_text  = ", ".join([f"{i['name']} x{i['qty']}" for i in items])

        return {
            "found":         True,
            "order_id":      order["id"],
            "customer":      order["customer"],
            "phone":         order["phone"],
            "items":         items_text,
            "items_detail":  items,
            "total":         f"{order['total']:,}đ",
            "status":        order["status"],
            "status_label":  status_info["label"],
            "status_icon":   status_info["icon"],
            "created_at":    order["created_at"],
            "eta":           order["eta"] or "Chưa xác định",
            "address":       order["address"],
            "shipper":       order["shipper"] or "Chưa có",
        }

# Khởi tạo
db = OrderDB(DB_PATH)

# Test
print("\n📋 Tất cả đơn hàng:")
for o in db.get_all():
    print(f"  {o['id']} | {o['customer']:<20} | {o['status']:<12} | {o['total']:,}đ")

# ════════════════════════════════════════════════
# CELL 5 — Tool OrderLookup dùng SQLite
# ════════════════════════════════════════════════

ORDER_PATTERNS = [
    r'\b(DH\d{3,})\b',
    r'\b(ORD[-_]?\d{3,})\b',
    r'đơn\s+(?:hàng\s+)?#?([A-Z0-9]{3,})',
    r'mã\s+(?:đơn\s+)?#?([A-Z0-9]{3,})',
    r'order\s+#?([A-Z0-9]{3,})',
]
PHONE_PATTERN = r'\b(0[0-9]{9})\b'

def extract_order_id(text: str) -> str | None:
    for pattern in ORDER_PATTERNS:
        m = re.search(pattern, text.upper())
        if m:
            return m.group(1)
    return None

def extract_phone(text: str) -> str | None:
    m = re.search(PHONE_PATTERN, text)
    return m.group(1) if m else None

def tool_order_lookup(user_message: str) -> dict:
    """
    Tool chính — tự động detect mã đơn hoặc SĐT
    """
    # Ưu tiên tìm mã đơn trước
    order_id = extract_order_id(user_message)
    if order_id:
        result = db.get_by_id(order_id)
        if result:
            return {"type": "by_id", "data": result}
        return {"type": "not_found", "order_id": order_id}

    # Nếu có SĐT → tìm theo SĐT
    phone = extract_phone(user_message)
    if phone:
        orders = db.get_by_phone(phone)
        if orders:
            return {"type": "by_phone", "data": orders, "phone": phone}
        return {"type": "not_found", "phone": phone}

    # Không có gì → hỏi lại
    return {"type": "need_info"}

# ════════════════════════════════════════════════
# CELL 6 — Format response
# ════════════════════════════════════════════════

def format_response(lookup_result: dict) -> str:
    t = lookup_result["type"]

    if t == "not_found":
        id_or_phone = lookup_result.get("order_id") or lookup_result.get("phone")
        return (
            f"Em không tìm thấy đơn hàng với mã/SĐT **{id_or_phone}** ạ.\n"
            f"Anh/chị kiểm tra lại giúp em, hoặc thử nhập số điện thoại đặt hàng nhé."
        )

    if t == "need_info":
        return (
            "Anh/chị cho em xin **mã đơn hàng** (VD: DH001) "
            "hoặc **số điện thoại** đặt hàng để em tra cứu nhé ạ!"
        )

    if t == "by_id":
        o = lookup_result["data"]
        r = (
            f"{o['status_icon']} Đơn **{o['order_id']}** — {o['status_label']}\n\n"
            f"👤 Khách hàng : {o['customer']}\n"
            f"📋 Sản phẩm  : {o['items']}\n"
            f"💰 Tổng tiền : {o['total']}\n"
            f"📅 Ngày đặt  : {o['created_at']}\n"
            f"📍 Địa chỉ   : {o['address']}\n"
        )
        if o["status"] == "delivering":
            r += (
                f"🚚 Shipper   : {o['shipper']}\n"
                f"📆 Dự kiến  : {o['eta']}\n\n"
                f"Đơn đang trên đường, anh/chị để ý điện thoại nhé! 😊"
            )
        elif o["status"] == "delivered":
            r += "\n Đã giao thành công! Nếu có vấn đề em hỗ trợ ngay nhé."
        elif o["status"] == "pending":
            r += f"\n⏳ Đang chờ kho xử lý, dự kiến giao {o['eta']}."
        elif o["status"] == "cancelled":
            r += "\n❌ Đơn đã huỷ. Nếu đã thanh toán em kiểm tra hoàn tiền giúp ạ."
        return r

    if t == "by_phone":
        orders = lookup_result["data"]
        r = f"📱 Tìm thấy **{len(orders)} đơn hàng** với SĐT {lookup_result['phone']}:\n\n"
        for o in orders:
            r += f"{o['status_icon']} **{o['order_id']}** — {o['items']} — {o['total']} — {o['status_label']}\n"
        r += "\nAnh/chị muốn xem chi tiết đơn nào, cho em mã đơn nhé!"
        return r

# ════════════════════════════════════════════════
# CELL 7 — Chatbot handler
# ════════════════════════════════════════════════

ORDER_KEYWORDS = [
    "đơn hàng", "đơn", "order", "mã đơn",
    "giao hàng", "ship", "tracking",
    "đang ở đâu", "sao rồi", "giao chưa", "khi nào tới",
]

def is_order_query(text: str) -> bool:
    if extract_order_id(text) or extract_phone(text):
        return True
    return any(kw in text.lower() for kw in ORDER_KEYWORDS)

def chatbot(user_message: str) -> str:
    print(f"\n{'─'*50}")
    print(f"👤 {user_message}")

    if not is_order_query(user_message):
        print("  → Không phải query đơn hàng")
        return None   # để Orchestrator xử lý cảm xúc

    result   = tool_order_lookup(user_message)
    response = format_response(result)

    print(f"🤖 {response}")
    return response

# ════════════════════════════════════════════════
# CELL 8 — Test
# ════════════════════════════════════════════════

tests = [
    "cho tôi thông tin đơn hàng DH001",
    "đơn dh002 đang ở đâu vậy?",
    "mã đơn DH003 sao rồi bạn ơi",
    "order DH005 bị sao vậy",
    "số 0901234567 có đơn nào không?",    # tìm theo SĐT
    "đơn hàng DH999 của tôi đâu",         # không tồn tại
    "đơn hàng của tôi giao chưa?",         # không có mã
    "ship lâu vl 5 ngày rồi",              # → emotion pipeline
]

for msg in tests:
    chatbot(msg)