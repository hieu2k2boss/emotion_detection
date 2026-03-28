import re
import chatbot_api.db as db

ORDER_PATTERNS = [
    r'\b(DH\d{3,})\b',
    r'đơn\s+(?:hàng\s+)?#?([A-Z0-9]{3,})',
    r'mã\s+(?:đơn\s+)?#?([A-Z0-9]{3,})',
    r'order\s+#?([A-Z0-9]{3,})',
]
PHONE_PATTERN  = r'\b(0[0-9]{9})\b'
ORDER_KEYWORDS = [
    "đơn hàng", "mã đơn", "order",
    "tracking", "tra cứu đơn",
    "đơn của tôi", "đơn của tui",
    "đang ở đâu rồi", "giao chưa vậy",
]
SHIP_ONLY = ["ship lâu", "giao lâu", "chờ lâu", "chưa thấy hàng"]

STATUS_MAP = {
    "pending":    {"label": "Chờ xác nhận", "icon": "⏳"},
    "confirmed":  {"label": "Đã xác nhận",  "icon": ""},
    "delivering": {"label": "Đang giao",     "icon": "🚚"},
    "delivered":  {"label": "Đã giao",       "icon": "📦"},
    "cancelled":  {"label": "Đã huỷ",        "icon": "❌"},
}

def extract_order_id(text: str) -> str | None:
    for p in ORDER_PATTERNS:
        m = re.search(p, text.upper())
        if m: return m.group(1)
    return None

def extract_phone(text: str) -> str | None:
    m = re.search(PHONE_PATTERN, text)
    return m.group(1) if m else None

def is_order_query(text: str) -> bool:
    # Có mã đơn hoặc SĐT → chắc chắn là order query
    if extract_order_id(text) or extract_phone(text):
        return True

    text_lower = text.lower()

    # Là phàn nàn ship → KHÔNG phải order query
    if any(kw in text_lower for kw in SHIP_ONLY):
        return False

    # Có keyword tra cứu đơn
    return any(kw in text_lower for kw in ORDER_KEYWORDS)

def order_lookup(text: str) -> str:
    """OrderLookupTool — trả về response text"""

    order_id = extract_order_id(text)
    if order_id:
        result = db.get_order(order_id)
        if not result:
            return f"Em không tìm thấy đơn **{order_id}** ạ. Anh/chị kiểm tra lại mã đơn giúp em nhé!"
        o    = result["order"]
        info = STATUS_MAP.get(o["status"], {"label": o["status"], "icon": "❓"})
        items_text = ", ".join([f"{i['name']} x{i['qty']}" for i in result["items"]])
        resp = (
            f"{info['icon']} Đơn **{o['id']}** — {info['label']}\n\n"
            f"👤 Khách hàng : {o['customer_name']}\n"
            f"📋 Sản phẩm  : {items_text}\n"
            f"💰 Tổng tiền : {o['total']:,}đ\n"
            f"📅 Ngày đặt  : {o['created_at']}\n"
            f"📍 Địa chỉ   : {o['address']}\n"
        )
        if o["status"] == "delivering":
            resp += f"🚚 Shipper   : {o['shipper']}\n📆 Dự kiến  : {o['eta']}\n\nĐang trên đường giao, anh/chị để ý điện thoại nhé! 😊"
        elif o["status"] == "delivered":
            resp += "\n Đã giao thành công! Nếu có vấn đề em hỗ trợ ngay nhé."
        elif o["status"] == "pending":
            resp += f"\n⏳ Đang chờ kho xử lý, dự kiến giao {o['eta']}."
        elif o["status"] == "cancelled":
            resp += "\n❌ Đơn đã huỷ. Nếu đã thanh toán em kiểm tra hoàn tiền giúp ạ."
        return resp

    phone = extract_phone(text)
    if phone:
        orders = db.get_orders_by_phone(phone)
        if not orders:
            return f"Em không tìm thấy đơn hàng nào với SĐT **{phone}** ạ."
        resp = f"📱 Tìm thấy **{len(orders)} đơn** với SĐT {phone}:\n\n"
        for o in orders:
            info = STATUS_MAP.get(o["status"], {"label": o["status"], "icon": "❓"})
            resp += f"{info['icon']} **{o['id']}** — {o['total']:,}đ — {info['label']}\n"
        resp += "\nAnh/chị muốn xem chi tiết đơn nào không ạ?"
        return resp

    return "Anh/chị cho em xin **mã đơn hàng** (VD: DH001) hoặc **số điện thoại** đặt hàng nhé ạ!"