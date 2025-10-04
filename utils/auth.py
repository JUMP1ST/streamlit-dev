import hashlib
import jwt
from datetime import datetime, timedelta

# Mock user database (replace with real DB)
USERS = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "administrator",
        "permissions": ["all"]
    },
    "user": {
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user",
        "permissions": ["read", "write", "execute"]
    },
    "viewer": {
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(),
        "role": "viewer",
        "permissions": ["read"]
    }
}

SECRET_KEY = "your-secret-key-here"

def check_authentication(username: str, password: str) -> bool:
    """Verify user credentials"""
    if username not in USERS:
        return False
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return USERS[username]["password_hash"] == password_hash

def get_user_role(username: str) -> str:
    """Get user role"""
    return USERS.get(username, {}).get("role", "unknown")

def get_user_permissions(username: str) -> list:
    """Get user permissions"""
    return USERS.get(username, {}).get("permissions", [])

def generate_token(username: str) -> str:
    """Generate JWT token"""
    payload = {
        "username": username,
        "role": get_user_role(username),
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
