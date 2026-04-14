import logging
from motor.motor_asyncio import AsyncIOMotorClient
from chatbot_api.config import settings

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        """Khởi tạo kết nối MongoDB và thiết lập database."""
        try:
            cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
            cls.db = cls.client[settings.DATABASE_NAME]
            logger.info(f"Đã kết nối tới MongoDB: {settings.MONGODB_URL}")
            
            # Tự động tạo Index
            await cls.create_indexes()
        except Exception as e:
            logger.error(f"Lỗi kết nối MongoDB: {e}")
            raise

    @classmethod
    async def close(cls):
        """Đóng kết nối MongoDB."""
        if cls.client:
            cls.client.close()
            logger.info("Đã đóng kết nối MongoDB.")

    @classmethod
    async def create_indexes(cls):
        """
        Khởi tạo các index cần thiết cho collection messages:
        - Index trên timestamp để truy vấn theo thời gian (giảm dần).
        - Index trên ai_analysis.emotion_label để thống kê.
        - Index trên ticket_id để tìm tin nhắn theo cuộc hội thoại.
        """
        try:
            # Collection: messages
            messages_collection = cls.db["messages"]
            
            # Index cho truy vấn thời gian - quan trọng cho Batch Layer
            await messages_collection.create_index([("timestamp", -1)])
            
            # Index cho thống kê cảm xúc - quan trọng cho Serving Layer
            await messages_collection.create_index([("ai_analysis.emotion_label", 1)])
            
            # Index cho liên kết cuộc hội thoại
            await messages_collection.create_index([("ticket_id", 1)])
            
            # Index cho customers (ví đạo tìm kiếm theo SĐT)
            await cls.db["customers"].create_index([("phone", 1)], unique=True)
            
            logger.info("Hoàn tất khởi tạo MongoDB Indexes.")
        except Exception as e:
            logger.error(f"Lỗi khi tạo Index: {e}")

db = MongoDB()
