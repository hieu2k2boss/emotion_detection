import sqlite3
from pathlib import Path

DB_PATH = "cskh.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.executescript("""

    -- ══════════════════════════════════════════
    -- KHÁCH HÀNG
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS customers (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        phone       TEXT    UNIQUE NOT NULL,
        email       TEXT,
        address     TEXT,
        tier        TEXT    DEFAULT 'normal',   -- normal / silver / gold / vip
        created_at  TEXT    DEFAULT (datetime('now'))
    );

    -- ══════════════════════════════════════════
    -- SẢN PHẨM
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS products (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        price       INTEGER NOT NULL,
        category    TEXT,
        stock       INTEGER DEFAULT 0
    );

    -- ══════════════════════════════════════════
    -- ĐƠN HÀNG
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS orders (
        id          TEXT    PRIMARY KEY,         -- DH001, DH002...
        customer_id INTEGER NOT NULL,
        total       INTEGER NOT NULL,
        status      TEXT    NOT NULL DEFAULT 'pending',
        -- pending / confirmed / delivering / delivered / cancelled
        created_at  TEXT    DEFAULT (datetime('now')),
        eta         TEXT,
        address     TEXT,
        shipper     TEXT,
        note        TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );

    -- ══════════════════════════════════════════
    -- CHI TIẾT ĐƠN HÀNG
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS order_items (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id    TEXT    NOT NULL,
        product_id  INTEGER NOT NULL,
        qty         INTEGER NOT NULL,
        price       INTEGER NOT NULL,            -- giá tại thời điểm đặt
        FOREIGN KEY (order_id)   REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );

    -- ══════════════════════════════════════════
    -- THANH TOÁN
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS payments (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id    TEXT    NOT NULL,
        amount      INTEGER NOT NULL,
        method      TEXT,                        -- cod / banking / momo / vnpay
        status      TEXT    DEFAULT 'pending',   -- pending / success / failed / refunded
        tx_code     TEXT,                        -- mã giao dịch ngân hàng
        paid_at     TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(id)
    );

    -- ══════════════════════════════════════════
    -- TICKET HỖ TRỢ
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS tickets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        order_id    TEXT,                        -- nullable — có thể hỏi không liên quan đơn
        status      TEXT    DEFAULT 'open',      -- open / in_progress / resolved / closed
        priority    TEXT    DEFAULT 'normal',    -- low / normal / high / urgent
        category    TEXT,                        -- shipping / payment / product / other
        created_at  TEXT    DEFAULT (datetime('now')),
        resolved_at TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (order_id)    REFERENCES orders(id)
    );

    -- ══════════════════════════════════════════
    -- TIN NHẮN HỘI THOẠI
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id   INTEGER NOT NULL,
        role        TEXT    NOT NULL,            -- customer / agent / bot
        content     TEXT    NOT NULL,
        created_at  TEXT    DEFAULT (datetime('now')),
        FOREIGN KEY (ticket_id) REFERENCES tickets(id)
    );

    -- ══════════════════════════════════════════
    -- PHÂN TÍCH CẢM XÚC (Agentic RAG output)
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id   INTEGER NOT NULL,
        message_id  INTEGER NOT NULL,
        emotion     TEXT    NOT NULL,            -- 7 nhãn
        confidence  REAL    NOT NULL,
        reason      TEXT,
        alert       INTEGER DEFAULT 0,           -- 0/1
        model       TEXT,                        -- deepseek-chat / deepseek-reasoner
        created_at  TEXT    DEFAULT (datetime('now')),
        FOREIGN KEY (ticket_id)  REFERENCES tickets(id),
        FOREIGN KEY (message_id) REFERENCES messages(id)
    );

    -- ══════════════════════════════════════════
    -- AGENT (nhân viên CSKH)
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS agents (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        email       TEXT    UNIQUE NOT NULL,
        status      TEXT    DEFAULT 'online',    -- online / busy / offline
        created_at  TEXT    DEFAULT (datetime('now'))
    );

    -- ══════════════════════════════════════════
    -- PHÂN CÔNG TICKET CHO AGENT
    -- ══════════════════════════════════════════
    CREATE TABLE IF NOT EXISTS ticket_agents (
        ticket_id   INTEGER NOT NULL,
        agent_id    INTEGER NOT NULL,
        assigned_at TEXT    DEFAULT (datetime('now')),
        PRIMARY KEY (ticket_id, agent_id),
        FOREIGN KEY (ticket_id) REFERENCES tickets(id),
        FOREIGN KEY (agent_id)  REFERENCES agents(id)
    );

    -- ══════════════════════════════════════════
    -- INDEX — tăng tốc query thường dùng
    -- ══════════════════════════════════════════
    CREATE INDEX IF NOT EXISTS idx_orders_customer  ON orders(customer_id);
    CREATE INDEX IF NOT EXISTS idx_orders_status    ON orders(status);
    CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id);
    CREATE INDEX IF NOT EXISTS idx_tickets_status   ON tickets(status);
    CREATE INDEX IF NOT EXISTS idx_messages_ticket  ON messages(ticket_id);
    CREATE INDEX IF NOT EXISTS idx_emotion_ticket   ON emotion_logs(ticket_id);
    CREATE INDEX IF NOT EXISTS idx_emotion_alert    ON emotion_logs(alert);
    CREATE INDEX IF NOT EXISTS idx_customer_phone   ON customers(phone);

    """)

    conn.commit()
    conn.close()
    print(f"✅ DB initialized: {Path(DB_PATH).resolve()}")

init_db()

# ════════════════════════════════════════════════
# CELL 2 — Seed mock data
# ════════════════════════════════════════════════

def seed_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # Customers
    customers = [
        ("Nguyễn Văn An",  "0901234567", "an@gmail.com",   "123 Lê Lợi Q1",        "gold"),
        ("Trần Thị Bình",  "0912345678", "binh@gmail.com", "456 Nguyễn Huệ Q1",     "normal"),
        ("Lê Văn Cường",   "0923456789", "cuong@gmail.com","789 Trần Hưng Đạo Q5",  "silver"),
        ("Phạm Thị Dung",  "0934567890", "dung@gmail.com", "321 Đinh Tiên Hoàng BT","normal"),
        ("Hoàng Minh Tuấn","0945678901", "tuan@gmail.com", "567 CMT8 Q3",           "vip"),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO customers (name, phone, email, address, tier)
        VALUES (?,?,?,?,?)
    """, customers)

    # Products
    products = [
        ("Áo thun nam",    150000, "thoi-trang", 100),
        ("Váy lụa",        850000, "thoi-trang", 50),
        ("Máy lọc nước",  2500000, "dien-tu",    20),
        ("Giày sneaker",  1200000, "giay-dep",   30),
        ("Quần jean",      350000, "thoi-trang", 80),
        ("Thắt lưng",      100000, "phu-kien",   60),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO products (name, price, category, stock)
        VALUES (?,?,?,?)
    """, products)

    # Orders
    orders = [
        ("DH001", 1, 300000,   "delivering", "2024-05-10", "2024-05-13", "123 Lê Lợi Q1",        "Giao Hàng Nhanh", None),
        ("DH002", 2, 850000,   "delivered",  "2024-05-08", "2024-05-11", "456 Nguyễn Huệ Q1",     "GHTK",            None),
        ("DH003", 3, 2500000,  "pending",    "2024-05-12", "2024-05-15", "789 Trần Hưng Đạo Q5",  None,              None),
        ("DH004", 4, 1200000,  "cancelled",  "2024-05-09", None,         "321 Đinh Tiên Hoàng BT", None,             "Khách huỷ"),
        ("DH005", 5, 450000,   "confirmed",  "2024-05-12", "2024-05-14", "567 CMT8 Q3",           "Viettel Post",    None),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO orders
            (id, customer_id, total, status, created_at, eta, address, shipper, note)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, orders)

    # Order items
    order_items = [
        ("DH001", 1, 2, 150000),
        ("DH002", 2, 1, 850000),
        ("DH003", 3, 1, 2500000),
        ("DH004", 4, 1, 1200000),
        ("DH005", 5, 1, 350000),
        ("DH005", 6, 1, 100000),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO order_items (order_id, product_id, qty, price)
        VALUES (?,?,?,?)
    """, order_items)

    # Payments
    payments = [
        ("DH001", 300000,  "banking", "success",  "TX001", "2024-05-10"),
        ("DH002", 850000,  "momo",    "success",  "TX002", "2024-05-08"),
        ("DH003", 2500000, "cod",     "pending",  None,    None),
        ("DH004", 1200000, "banking", "refunded", "TX004", "2024-05-10"),
        ("DH005", 450000,  "vnpay",   "success",  "TX005", "2024-05-12"),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO payments
            (order_id, amount, method, status, tx_code, paid_at)
        VALUES (?,?,?,?,?,?)
    """, payments)

    # Agents
    agents = [
        ("Minh Châu",  "chau@cskh.com",  "online"),
        ("Bảo Trân",   "tran@cskh.com",  "busy"),
        ("Quốc Hùng",  "hung@cskh.com",  "online"),
    ]
    cur.executemany("""
        INSERT OR IGNORE INTO agents (name, email, status)
        VALUES (?,?,?)
    """, agents)

    conn.commit()
    conn.close()
    print("✅ Seeded mock data")

seed_db()

# ════════════════════════════════════════════════
# CELL 3 — Query thường dùng
# ════════════════════════════════════════════════

class CSKH_DB:
    def __init__(self, path=DB_PATH):
        self.path = path

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Orders ───────────────────────────────────
    def get_order(self, order_id: str) -> dict | None:
        conn = self._conn()
        row  = conn.execute("""
            SELECT o.*, c.name as customer_name, c.phone, c.tier,
                   p.amount, p.method, p.status as pay_status, p.tx_code
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            LEFT JOIN payments p ON o.id = p.order_id
            WHERE o.id = ?
        """, (order_id.upper(),)).fetchone()

        if not row:
            conn.close()
            return None

        items = conn.execute("""
            SELECT pr.name, oi.qty, oi.price
            FROM order_items oi
            JOIN products pr ON oi.product_id = pr.id
            WHERE oi.order_id = ?
        """, (order_id.upper(),)).fetchall()

        conn.close()
        return {"order": dict(row), "items": [dict(i) for i in items]}

    def get_orders_by_phone(self, phone: str) -> list:
        conn   = self._conn()
        rows   = conn.execute("""
            SELECT o.id, o.status, o.total, o.created_at, o.eta
            FROM orders o
            JOIN customers c ON o.customer_id = c.id
            WHERE c.phone = ?
            ORDER BY o.created_at DESC LIMIT 5
        """, (phone,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Tickets ──────────────────────────────────
    def create_ticket(self, customer_id: int, order_id: str = None,
                      category: str = "other") -> int:
        conn = self._conn()
        cur  = conn.execute("""
            INSERT INTO tickets (customer_id, order_id, category)
            VALUES (?,?,?)
        """, (customer_id, order_id, category))
        ticket_id = cur.lastrowid
        conn.commit()
        conn.close()
        return ticket_id

    def add_message(self, ticket_id: int, role: str, content: str) -> int:
        conn = self._conn()
        cur  = conn.execute("""
            INSERT INTO messages (ticket_id, role, content)
            VALUES (?,?,?)
        """, (ticket_id, role, content))
        msg_id = cur.lastrowid
        conn.commit()
        conn.close()
        return msg_id

    def log_emotion(self, ticket_id: int, message_id: int,
                    emotion: str, confidence: float,
                    reason: str, alert: bool, model: str):
        conn = self._conn()
        conn.execute("""
            INSERT INTO emotion_logs
                (ticket_id, message_id, emotion, confidence, reason, alert, model)
            VALUES (?,?,?,?,?,?,?)
        """, (ticket_id, message_id, emotion,
              confidence, reason, int(alert), model))
        conn.commit()
        conn.close()

    def get_alert_tickets(self) -> list:
        """Lấy các ticket có alert cần escalate"""
        conn = self._conn()
        rows = conn.execute("""
            SELECT t.id, t.status, t.priority, c.name, c.phone,
                   el.emotion, el.confidence, el.reason, el.created_at
            FROM emotion_logs el
            JOIN tickets t  ON el.ticket_id  = t.id
            JOIN customers c ON t.customer_id = c.id
            WHERE el.alert = 1
            ORDER BY el.created_at DESC
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

# ════════════════════════════════════════════════
# CELL 4 — Test queries
# ════════════════════════════════════════════════

db = CSKH_DB()

# Test get_order
print("📦 Tra cứu DH001:")
result = db.get_order("DH001")
if result:
    o = result["order"]
    print(f"  {o['id']} | {o['customer_name']} | {o['status']} | {o['total']:,}đ")
    print(f"  Thanh toán: {o['method']} | {o['pay_status']} | {o['tx_code']}")
    for item in result["items"]:
        print(f"  - {item['name']} x{item['qty']} = {item['price']:,}đ")

# Test get_orders_by_phone
print(f"\n📱 Đơn hàng SĐT 0901234567:")
for o in db.get_orders_by_phone("0901234567"):
    print(f"  {o['id']} | {o['status']} | {o['total']:,}đ")

# Test ticket + emotion log
print(f"\n🎫 Tạo ticket + log cảm xúc:")
ticket_id = db.create_ticket(customer_id=1, order_id="DH001", category="shipping")
msg_id    = db.add_message(ticket_id, "customer", "ship lâu vl 5 ngày rồi")
db.log_emotion(ticket_id, msg_id, "frustrated", 0.85,
               "Chờ lâu + dùng 'vl'", False, "deepseek-chat")
print(f"  ticket_id={ticket_id} | msg_id={msg_id}")

# Test alert
print(f"\n⚠️  Alert tickets:")
ticket_id2 = db.create_ticket(customer_id=1, order_id="DH001", category="shipping")
msg_id2    = db.add_message(ticket_id2, "customer", "lần đầu mà cũng lần cuối luôn")
db.log_emotion(ticket_id2, msg_id2, "angry", 0.93,
               "Tuyên bố không quay lại", True, "deepseek-reasoner")

for row in db.get_alert_tickets():
    print(f"  [{row['emotion']}] {row['name']} | {row['reason']}")