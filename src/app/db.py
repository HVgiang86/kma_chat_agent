from datetime import datetime
from typing import List, Dict, Any, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel
from app.utils.logger import logger

# We'll use a global variable to store db reference when set
_db = None

def get_db():
    """Get the database reference"""
    global _db
    if _db is None:
        from app.main import db
        _db = db
    return _db

class Database:
    @staticmethod
    async def find_one(collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a document in the database"""
        try:
            query = {**query, "deleted_at": None}
            logger.debug(f"Finding one document in {collection}", {"query": str(query)})
            logger.info(f"Finding one document in {collection}", {"query": str(query)})
            result = await get_db()[collection].find_one(query)
            logger.log_db_operation("find_one", collection, success=True)
            logger.info(f"Document found in {collection}", {"result": str(result)})
            return result
        except Exception as e:
            logger.error(f"Error finding document in {collection}", {"error": str(e), "query": str(query)}, exc_info=True)
            raise

    @staticmethod
    async def find(collection: str, query: Dict[str, Any], 
                   skip: int = 0, limit: int = 0, 
                   sort: List[tuple] = None) -> List[Dict[str, Any]]:
        """Find documents in the database"""
        try:
            query = {**query, "deleted_at": None}
            logger.debug(f"Finding documents in {collection}", 
                        {"query": str(query), "skip": skip, "limit": limit, "sort": str(sort)})
            
            cursor = get_db()[collection].find(query).skip(skip)
            
            if limit > 0:
                cursor = cursor.limit(limit)
                
            if sort:
                cursor = cursor.sort(sort)
                
            results = await cursor.to_list(length=None)
            logger.log_db_operation("find", collection, success=True, 
                                   data={"count": len(results), "skip": skip, "limit": limit})
            return results
        except Exception as e:
            logger.error(f"Error finding documents in {collection}", 
                        {"error": str(e), "query": str(query)}, 
                        exc_info=True)
            raise

    @staticmethod
    async def insert_one(collection: str, document: Dict[str, Any]) -> str:
        """Insert a document into the database"""
        try:
            logger.debug(f"Inserting document into {collection}")
            result = await get_db()[collection].insert_one(document)
            inserted_id = str(result.inserted_id)
            logger.log_db_operation("insert_one", collection, success=True, 
                                   data={"inserted_id": inserted_id})
            return inserted_id
        except Exception as e:
            logger.error(f"Error inserting document into {collection}", 
                        {"error": str(e)}, 
                        exc_info=True)
            raise

    @staticmethod
    async def update_one(collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update a document in the database"""
        try:
            query = {**query, "deleted_at": None}
            update["updated_at"] = datetime.utcnow()
            
            logger.debug(f"Updating document in {collection}", 
                        {"query": str(query), "update": str(update)})
            
            result = await get_db()[collection].update_one(query, {"$set": update})
            success = result.modified_count > 0
            
            logger.log_db_operation("update_one", collection, success=success, 
                                   data={"modified_count": result.modified_count})
            
            return success
        except Exception as e:
            logger.error(f"Error updating document in {collection}", 
                        {"error": str(e), "query": str(query)}, 
                        exc_info=True)
            raise

    @staticmethod
    async def delete_one(collection: str, query: Dict[str, Any]) -> bool:
        """Soft delete a document in the database"""
        try:
            query = {**query, "deleted_at": None}
            update = {"deleted_at": datetime.utcnow()}
            
            logger.debug(f"Soft deleting document in {collection}", {"query": str(query)})
            
            result = await get_db()[collection].update_one(query, {"$set": update})
            success = result.modified_count > 0
            
            logger.log_db_operation("delete_one (soft)", collection, success=success, 
                                   data={"modified_count": result.modified_count})
            
            return success
        except Exception as e:
            logger.error(f"Error soft deleting document in {collection}", 
                        {"error": str(e), "query": str(query)}, 
                        exc_info=True)
            raise

    @staticmethod
    async def hard_delete_one(collection: str, query: Dict[str, Any]) -> bool:
        """Hard delete a document from the database"""
        try:
            logger.debug(f"Hard deleting document in {collection}", {"query": str(query)})
            
            result = await get_db()[collection].delete_one(query)
            success = result.deleted_count > 0
            
            logger.log_db_operation("delete_one (hard)", collection, success=success, 
                                   data={"deleted_count": result.deleted_count})
            
            return success
        except Exception as e:
            logger.error(f"Error hard deleting document in {collection}", 
                        {"error": str(e), "query": str(query)}, 
                        exc_info=True)
            raise

    @staticmethod
    def get_object_id(id: str) -> ObjectId:
        """Convert a string to an ObjectId"""
        try:
            return ObjectId(id)
        except Exception as e:
            logger.error(f"Error converting to ObjectId", {"id": id, "error": str(e)})
            raise

# Helper functions for converting between model objects and database documents
def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    """Convert a model to a dictionary"""
    return model.model_dump(by_alias=True)

def dict_to_model(model_class, data: Dict[str, Any]) -> BaseModel:
    """Convert a dictionary to a model"""
    return model_class(**data) 