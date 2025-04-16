from fastapi import APIRouter, HTTPException, status, Request
from typing import List, Dict
from bson import ObjectId
from app.models.user import UserCreate, UserUpdate, UserInDB, UserResponse, UserBase
from app.db import Database, model_to_dict
from app.utils.logger import logger

router = APIRouter()

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, request: Request):
    logger.info(f"Creating new user with student code: {user.student_code}")
    
    # Check if student code already exists
    existing_user = await Database.find_one("users", {"student_code": user.student_code})
    if existing_user:
        logger.warning(f"User creation failed: student code {user.student_code} already registered")
        logger.info(f"User creation failed: student code {user.student_code} already registered")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student code already registered"
        )
    
    # Create user base data with only required fields
    user_data = {"student_code": user.student_code}
    
    # Add optional fields if provided
    if user.name is not None:
        user_data["name"] = user.name
    if user.student_class is not None:
        user_data["student_class"] = user.student_class
    if user.app_settings is not None:
        user_data["app_settings"] = user.app_settings
    else:
        user_data["app_settings"] = {
            "theme": "light",
        }

    if user.is_guest is not None:
        user_data["is_guest"] = user.is_guest
    if user.student_code is not None:
        user_data["student_code"] = user.student_code

    # Set default values for optional fields
    user_data["is_guest"] = user_data.get("is_guest", False)
    user_data["is_active"] = user_data.get("is_active", True)
    
    # Create user document
    user_db = UserInDB(**user_data)
    user_dict = model_to_dict(user_db)
    user_id = await Database.insert_one("users", user_dict)

    user_data["id"] = str(user_id)

    logger.info(f"User created successfully", user_dict)
    # Return created user
    created_user = {**user_dict, "_id": user_id}

    logger.info(f"Haha", created_user["student_code"])

    logger.info(f"User created successfully", created_user)
    return UserResponse(**user_data)

@router.get("/{student_code}", response_model=UserResponse)
async def get_user(student_code: str):
    try:
        logger.debug(f"Getting user by Student code: {student_code}")
        user = await Database.find_one("users", {"student_code": student_code})

        if not user:
            logger.warning(f"User not found with Code: {student_code}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        user["id"] = str(user["_id"])
        logger.debug(f"User found with ID: {student_code}")
        return UserResponse(**user)
    except Exception as e:
        logger.error(f"Error getting user", {"user_id": student_code, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID: {str(e)}"
        )

@router.get("/by-id/{user_id}", response_model=UserResponse)
async def get_user_by_student_code(user_id: str):
    logger.debug(f"Getting user by student code: {user_id}")
    user = await Database.find_one("users", {"_id": ObjectId(user_id)})
    if not user:
        logger.warning(f"User not found with student code: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    user["id"] = str(user["_id"])
    logger.debug(f"User found with student code: {user_id}")
    return UserResponse(**user)

@router.put("/{student_code}", response_model=UserResponse)
async def update_user(student_code: str, user_update: UserUpdate):
    try:
        logger.info(f"Updating user with ID: {student_code}")
        
        # Check if user exists
        user = await Database.find_one("users", {"student_code": student_code})
        if not user:
            logger.warning(f"User not found with ID: {student_code}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update user
        update_data = {k: v for k, v in user_update.model_dump().items() if v is not None}
        if not update_data:
            logger.info(f"No fields to update for user: {student_code}")
            # No valid fields to update
            user["id"] = str(user["_id"])
            return UserResponse(**user)
        
        logger.info(f"Updating user fields", {"student_code": student_code, "fields": list(update_data.keys())})
        
        success = await Database.update_one("users", {"student_code": student_code}, update_data)
        if not success:
            logger.error(f"Failed to update user", {"student_code": student_code})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user"
            )
        
        # Get updated user
        updated_user = await Database.find_one("users", {"student_code": student_code})
        logger.info(f"User updated successfully", {"user_id": student_code})
        updated_user["id"] = str(updated_user["_id"])
        return UserResponse(**updated_user)
    except Exception as e:
        logger.error(f"Error updating user", {"user_id": student_code, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID or data: {str(e)}"
        )

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str):
    try:
        logger.info(f"Deleting user with ID: {user_id}")
        
        # Check if user exists
        user = await Database.find_one("users", {"_id": ObjectId(user_id)})
        if not user:
            logger.warning(f"User not found with ID: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Soft delete the user
        success = await Database.delete_one("users", {"_id": ObjectId(user_id)})
        if not success:
            logger.error(f"Failed to delete user", {"user_id": user_id})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )
        
        logger.info(f"User deleted successfully", {"user_id": user_id})
    except Exception as e:
        logger.error(f"Error deleting user", {"user_id": user_id, "error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user ID: {str(e)}"
        ) 