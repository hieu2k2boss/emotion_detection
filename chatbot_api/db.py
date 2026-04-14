import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from chatbot_api.database import db

logger = logging.getLogger(__name__)

# ── Write helpers ─────────────────────────────

async def create_ticket(customer_id: Optional[int], order_id: Optional[str], category: str) -> int:
    """Tạo ticket mới trong MongoDB."""
    try:
        # Trong MongoDB, chúng ta có thể dùng ID tự tăng hoặc auto-generated.
        # Để giữ tương thích với code cũ dùng int, ta sẽ lấy count + 1 hoặc dùng timestamp.
        # Ở đây dùng timestamp đơn giản cho demo.
        ticket_id = int(datetime.utcnow().timestamp() * 1000)
        await db.db["tickets"].insert_one({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "order_id": order_id,
            "category": category,
            "status": "open",
            "created_at": datetime.utcnow()
        })
        return ticket_id
    except Exception as e:
        logger.error(f"Lỗi create_ticket: {e}")
        return 0

async def save_message(ticket_id: int, role: str, content: str, ai_analysis: Optional[Dict] = None) -> str:
    """Lưu tin nhắn thô, có thể đính kèm kết quả phân tích AI."""
    try:
        doc = {
            "ticket_id": ticket_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "ai_analysis": ai_analysis
        }
        result = await db.db["messages"].insert_one(doc)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Lỗi save_message: {e}")
        return ""

async def update_message_analysis(message_id: str, analysis: Dict):
    """Cập nhật kết quả phân tích AI cho một tin nhắn đã tồn tại."""
    try:
        from bson import ObjectId
        await db.db["messages"].update_one(
            {"_id": ObjectId(message_id)},
            {"$set": {"ai_analysis": analysis}}
        )
    except Exception as e:
        logger.error(f"Lỗi update_message_analysis: {e}")

async def save_emotion(ticket_id: int, message_id: str,
                       emotion: str, confidence: float,
                       reason: str, alert: bool, processing_time_ms: int = 0):
    """
    Refactor: Gộp vào collection messages.
    Tìm tin nhắn theo ID và update field ai_analysis.
    """
    analysis = {
        "emotion_label": emotion,
        "confidence": confidence,
        "reason": reason,
        "has_alert": alert,
        "processing_time_ms": processing_time_ms
    }
    await update_message_analysis(message_id, analysis)

# ── Query helpers ─────────────────────────────

async def get_messages(ticket_id: int) -> List[Dict]:
    """Lấy danh sách tin nhắn của một ticket."""
    try:
        cursor = db.db["messages"].find({"ticket_id": ticket_id}).sort("timestamp", 1)
        messages = await cursor.to_list(length=100)
        for m in messages:
            m["_id"] = str(m["_id"])
            # Format lại để tương thích với code cũ nếu cần
            m["created_at"] = m["timestamp"].isoformat()
        return messages
    except Exception as e:
        logger.error(f"Lỗi get_messages: {e}")
        return []

async def get_all_tickets_with_emotions() -> List[Dict]:
    """
    Sử dụng Aggregation để lấy group messages theo ticket và đính kèm emotion mới nhất.
    """
    pipeline = [
        {"$sort": {"ticket_id": 1, "timestamp": 1}},
        {
            "$group": {
                "_id": "$ticket_id",
                "messages": {
                    "$push": {
                        "role": "$role",
                        "content": "$content"
                    }
                },
                # Lấy ai_analysis cuối cùng mà khác null
                "analyses": {
                    "$push": "$ai_analysis"
                }
            }
        },
        {
            "$project": {
                "ticket_id": "$_id",
                "messages": 1,
                "last_analysis": {
                    "$reduce": {
                        "input": "$analyses",
                        "initialValue": None,
                        "in": {"$ifNull": ["$$this", "$$value"]}
                    }
                }
            }
        },
        {
            "$project": {
                "ticket_id": 1,
                "messages": 1,
                "emotion": {"$ifNull": ["$last_analysis.emotion_label", "neutral"]},
                "confidence": {"$ifNull": ["$last_analysis.confidence", 0]},
                "alert": {"$ifNull": ["$last_analysis.has_alert", False]},
                "reason": {"$ifNull": ["$last_analysis.reason", ""]}
            }
        }
    ]
    try:
        cursor = db.db["messages"].aggregate(pipeline)
        return await cursor.to_list(length=1000)
    except Exception as e:
        logger.error(f"Lỗi get_all_tickets_with_emotions: {e}")
        return []

async def get_order(order_id: str) -> Optional[Dict]:
    """Lấy thông tin đơn hàng từ MongoDB (cần migrate collection orders)."""
    try:
        order = await db.db["orders"].find_one({"order_id": order_id.upper()})
        if not order: return None
        
        items_cursor = db.db["order_items"].find({"order_id": order_id.upper()})
        items = await items_cursor.to_list(length=100)
        
        return {
            "order": order,
            "items": items
        }
    except Exception as e:
        logger.error(f"Lỗi get_order: {e}")
        return None

# Seed data functions (Refactored to Mongo)
async def init_db():
    """Trigger index creation (đã có trong database.py)."""
    await db.create_indexes()

async def seed_db():
    """Seed dữ liệu mẫu vào MongoDB cho demo."""
    try:
        # Customers
        customers = [
            {"id": 1, "name": "Nguyễn Văn An", "phone": "0901234567", "tier": "gold"},
            {"id": 2, "name": "Trần Thị Bình", "phone": "0912345678", "tier": "normal"},
        ]
        await db.db["customers"].delete_many({})
        await db.db["customers"].insert_many(customers)
        
        # Products
        products = [
            {"id": 1, "name": "Áo thun nam", "price": 150000, "category": "thoi-trang"},
            {"id": 2, "name": "Váy lụa", "price": 850000, "category": "thoi-trang"},
        ]
        await db.db["products"].delete_many({})
        await db.db["products"].insert_many(products)

        # Orders
        orders = [
            {"order_id": "DH001", "customer_id": 1, "total": 300000, "status": "delivering"},
            {"order_id": "DH002", "customer_id": 2, "total": 850000, "status": "delivered"},
        ]
        await db.db["orders"].delete_many({})
        await db.db["orders"].insert_many(orders)
        
        logger.info("Đã seed dữ liệu mẫu thành công.")
    except Exception as e:
        logger.error(f"Lỗi seed_db: {e}")