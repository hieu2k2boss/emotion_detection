import sqlite3
from pathlib import Path

DB_PATH = "chatbot_api/cskh.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT    NOT NULL,
        phone      TEXT    UNIQUE NOT NULL,
        email      TEXT,
        address    TEXT,
        tier       TEXT    DEFAULT 'normal'
    );
    CREATE TABLE IF NOT EXISTS products (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT    NOT NULL,
        price      INTEGER NOT NULL,
        category   TEXT,
        stock      INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS orders (
        id         TEXT    PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        total      INTEGER NOT NULL,
        status     TEXT    NOT NULL DEFAULT 'pending',
        created_at TEXT    DEFAULT (datetime('now')),
        eta        TEXT,
        address    TEXT,
        shipper    TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    CREATE TABLE IF NOT EXISTS order_items (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id   TEXT    NOT NULL,
        product_id INTEGER NOT NULL,
        qty        INTEGER NOT NULL,
        price      INTEGER NOT NULL,
        FOREIGN KEY (order_id)   REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    );
    CREATE TABLE IF NOT EXISTS payments (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id   TEXT    NOT NULL,
        amount     INTEGER NOT NULL,
        method     TEXT,
        status     TEXT    DEFAULT 'pending',
        tx_code    TEXT,
        paid_at    TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(id)
    );
    CREATE TABLE IF NOT EXISTS tickets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        order_id    TEXT,
        status      TEXT DEFAULT 'open',
        category    TEXT DEFAULT 'other',
        created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id  INTEGER NOT NULL,
        role       TEXT    NOT NULL,
        content    TEXT    NOT NULL,
        created_at TEXT    DEFAULT (datetime('now')),
        FOREIGN KEY (ticket_id) REFERENCES tickets(id)
    );
    CREATE TABLE IF NOT EXISTS emotion_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id  INTEGER NOT NULL,
        message_id INTEGER NOT NULL,
        emotion    TEXT    NOT NULL,
        confidence REAL    NOT NULL,
        reason     TEXT,
        alert      INTEGER DEFAULT 0,
        created_at TEXT    DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_customer_phone ON customers(phone);
    CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
    CREATE INDEX IF NOT EXISTS idx_messages_ticket ON messages(ticket_id);
    """)
    conn.commit()
    conn.close()

def seed_db():
    conn = get_conn()
    cur  = conn.cursor()

    cur.executemany("INSERT OR IGNORE INTO customers (name,phone,email,address,tier) VALUES (?,?,?,?,?)", [
        ("Nguyễn Văn An",   "0901234567", "an@gmail.com",   "123 Lê Lợi Q1",       "gold"),
        ("Trần Thị Bình",   "0912345678", "binh@gmail.com", "456 Nguyễn Huệ Q1",    "normal"),
        ("Lê Văn Cường",    "0923456789", "cuong@gmail.com","789 Trần Hưng Đạo Q5", "silver"),
    ])
    cur.executemany("INSERT OR IGNORE INTO products (name,price,category,stock) VALUES (?,?,?,?)", [
        ("Áo thun nam",  150000, "thoi-trang", 100),
        ("Váy lụa",      850000, "thoi-trang",  50),
        ("Máy lọc nước",2500000, "dien-tu",     20),
    ])
    cur.executemany("""INSERT OR IGNORE INTO orders
        (id,customer_id,total,status,created_at,eta,address,shipper) VALUES (?,?,?,?,?,?,?,?)""", [
        ("DH001", 1, 300000,  "delivering","2024-05-10","2024-05-13","123 Lê Lợi Q1",      "GHN"),
        ("DH002", 2, 850000,  "delivered", "2024-05-08","2024-05-11","456 Nguyễn Huệ Q1",  "GHTK"),
        ("DH003", 3, 2500000, "pending",   "2024-05-12","2024-05-15","789 Trần Hưng Đạo Q5",None),
    ])
    cur.executemany("INSERT OR IGNORE INTO order_items (order_id,product_id,qty,price) VALUES (?,?,?,?)", [
        ("DH001", 1, 2, 150000),
        ("DH002", 2, 1, 850000),
        ("DH003", 3, 1, 2500000),
    ])
    cur.executemany("INSERT OR IGNORE INTO payments (order_id,amount,method,status,tx_code,paid_at) VALUES (?,?,?,?,?,?)", [
        ("DH001", 300000,  "banking", "success", "TX001", "2024-05-10"),
        ("DH002", 850000,  "momo",    "success", "TX002", "2024-05-08"),
        ("DH003", 2500000, "cod",     "pending",  None,    None),
    ])
    conn.commit()
    conn.close()

# ── Query helpers ─────────────────────────────

# db.py — thêm DISTINCT hoặc GROUP BY
def get_order(order_id: str) -> dict | None:
    conn  = get_conn()
    order = conn.execute("""
        SELECT DISTINCT o.*, c.name as customer_name, c.phone, c.tier,
               p.method, p.status as pay_status, p.tx_code
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        LEFT JOIN payments p ON o.id = p.order_id
        WHERE o.id = ?
        LIMIT 1          -- chỉ lấy 1 row đơn hàng
    """, (order_id.upper(),)).fetchone()

    if not order:
        conn.close()
        return None

    items = conn.execute("""
        SELECT pr.name, oi.qty, oi.price
        FROM order_items oi
        JOIN products pr ON oi.product_id = pr.id
        WHERE oi.order_id = ?
    """, (order_id.upper(),)).fetchall()

    conn.close()
    return {"order": dict(order), "items": [dict(i) for i in items]}

def get_orders_by_phone(phone: str) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT o.id, o.status, o.total, o.created_at
        FROM orders o JOIN customers c ON o.customer_id = c.id
        WHERE c.phone = ?
        ORDER BY o.created_at DESC LIMIT 5
    """, (phone,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_ticket(customer_id: int | None, order_id: str | None, category: str) -> int:
    conn = get_conn()
    cur  = conn.execute(
        "INSERT INTO tickets (customer_id, order_id, category) VALUES (?,?,?)",
        (customer_id, order_id, category)
    )
    tid = cur.lastrowid
    conn.commit(); conn.close()
    return tid

def save_message(ticket_id: int, role: str, content: str) -> int:
    conn = get_conn()
    cur  = conn.execute(
        "INSERT INTO messages (ticket_id, role, content) VALUES (?,?,?)",
        (ticket_id, role, content)
    )
    mid = cur.lastrowid
    conn.commit(); conn.close()
    return mid

def save_emotion(ticket_id: int, message_id: int,
                 emotion: str, confidence: float,
                 reason: str, alert: bool):
    conn = get_conn()
    conn.execute("""
        INSERT INTO emotion_logs
            (ticket_id, message_id, emotion, confidence, reason, alert)
        VALUES (?,?,?,?,?,?)
    """, (ticket_id, message_id, emotion, confidence, reason, int(alert)))
    conn.commit(); conn.close()

def get_messages(ticket_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE ticket_id=? ORDER BY id",
        (ticket_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]