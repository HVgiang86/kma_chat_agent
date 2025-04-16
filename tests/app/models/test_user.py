from datetime import datetime
from bson import ObjectId
from app.models.user import (
    UserBase,
    UserCreate,
    UserUpdate,
    UserInDB,
    UserResponse,
    PyObjectId
)

def test_pyobjectid_validation():
    # Test valid ObjectId
    valid_id = str(ObjectId())
    assert PyObjectId.validate(valid_id) == ObjectId(valid_id)
    
    # Test invalid ObjectId
    import pytest
    with pytest.raises(ValueError):
        PyObjectId.validate("invalid_id")

def test_userbase_creation():
    user = UserBase(
        student_code="123456",
        name="Test User",
        student_class="K66",
        app_settings={"theme": "dark"},
        is_guest=False,
        is_active=True
    )
    
    assert user.student_code == "123456"
    assert user.name == "Test User"
    assert user.student_class == "K66"
    assert user.app_settings == {"theme": "dark"}
    assert user.is_guest is False
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)
    assert isinstance(user.updated_at, datetime)
    assert user.deleted_at is None

def test_usercreate_creation():
    user = UserCreate(
        student_code="123456",
        name="Test User",
        student_class="K66",
        app_settings={"theme": "dark"},
        is_guest=False
    )
    
    assert user.student_code == "123456"
    assert user.name == "Test User"
    assert user.student_class == "K66"
    assert user.app_settings == {"theme": "dark"}
    assert user.is_guest is False

def test_userupdate_creation():
    user = UserUpdate(
        name="Updated Name",
        student_class="K67",
        app_settings={"theme": "light"},
        is_active=False
    )
    
    assert user.name == "Updated Name"
    assert user.student_class == "K67"
    assert user.app_settings == {"theme": "light"}
    assert user.is_active is False

def test_userindb_creation():
    user_id = ObjectId()
    user = UserInDB(
        _id=user_id,
        student_code="123456",
        name="Test User",
        student_class="K66",
        app_settings={"theme": "dark"},
        is_guest=False,
        is_active=True
    )
    
    assert user.id == user_id
    assert user.student_code == "123456"
    assert user.name == "Test User"
    assert user.student_class == "K66"
    assert user.app_settings == {"theme": "dark"}
    assert user.is_guest is False
    assert user.is_active is True

def test_userresponse_creation():
    user = UserResponse(
        id=str(ObjectId()),
        student_code="123456",
        name="Test User",
        student_class="K66",
        app_settings={"theme": "dark"},
        is_guest=False,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    assert isinstance(user.id, str)
    assert user.student_code == "123456"
    assert user.name == "Test User"
    assert user.student_class == "K66"
    assert user.app_settings == {"theme": "dark"}
    assert user.is_guest is False
    assert user.is_active is True
    assert isinstance(user.created_at, datetime)
    assert isinstance(user.updated_at, datetime) 