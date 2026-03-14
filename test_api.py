# Sửa test_api.py — thêm proxies={}
import requests, json

BASE = "http://127.0.0.1:8005"  # dùng 127.0.0.1 thay vì localhost

def test(message: str, ticket_id: int = None):
    res = requests.post(
        f"{BASE}/chat",
        json    = {"message": message, "ticket_id": ticket_id},
        proxies = {"http": None, "https": None},  # bypass proxy
        timeout = 30,
    )
    print(f"Status: {res.status_code}")
    if res.status_code != 200:
        print(f"Error: {res.text[:200]}")
        return None
    data = res.json()
    print(f"\n👤 {message}")
    print(f"🤖 {data['reply']}")
    print(f"   😶 {data['emotion']} ({data['confidence']*100:.0f}%) | alert={data['alert']}")
    print(f"   🎫 ticket_id={data['ticket_id']}")
    return data["ticket_id"]

# Test
tid  = test("cho tôi thông tin đơn hàng DH001")
tid  = test("ship lâu vl 5 ngày rồi", ticket_id=tid)
tid2 = test("SĐT 0901234567 có đơn nào không?")
tid3 = test("chính sách đổi trả như thế nào?")

if tid:
    print(f"\n📜 Lịch sử ticket {tid}:")
    history = requests.get(
        f"{BASE}/history/{tid}",
        proxies={"http": None, "https": None}
    ).json()
    for m in history["messages"]:
        icon = "👤" if m["role"] == "customer" else "🤖"
        print(f"  {icon} {m['content'][:60]}")