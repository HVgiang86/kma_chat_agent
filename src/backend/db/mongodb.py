import logging
from pydantic_settings import BaseSettings
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

class MongoDBSettings(BaseSettings):
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "ai_chat"
    RAG_MODEL: str = "RAG"
    GEMINI_MODEL: str = "GEMINI"
    GOOGLE_API_KEY: str = "GOOGLE_API_KEY"
    LANGCHAIN_API_KEY: str = "LANCHAIN_API_KEY"
    LANGCHAIN_TRACING_V2: str = "LANCHAIN_TRACING_V2"
    LANGCHAIN_PROJECT: str = "LANCHAIN_PROJECT"
    POSTGRES_URI: str = "POSTGRES_URI"
    JWT_SECRET_KEY: str = "your_key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    PORT: int = 8000
    DEV_MODE: str = "False"
    LOG_LEVEL: str = "DEBUG"
    # ... other settings ...

    class Config:
        env_file = ".env"

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect_to_mongodb(cls):
        settings = MongoDBSettings()
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URL}")
        cls.client = AsyncIOMotorClient(settings.MONGODB_URL)
        cls.db = cls.client[settings.MONGODB_DB_NAME]
        logger.info("Connected to MongoDB")
        collections = await cls.db.list_collection_names()
        logger.info(f"Collections: {collections}")

    @classmethod
    async def close_mongodb_connection(cls):
        if cls.client:
            logger.info("Closing MongoDB connection")
            cls.client.close()
            logger.info("MongoDB connection closed")

    @property
    def conversations(self):
        return self.db.conversations

    @property
    def user(self):
        return self.db.user

    @property
    def messages(self):
        return self.db.messages

mongodb = MongoDB()
