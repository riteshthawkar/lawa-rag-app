import jwt
from .config import SECRET_KEY, logger

def verify_token(token: str):
    """
    Verifies the JWT token using the shared SECRET_KEY.
    Returns the payload if valid, raises an exception otherwise.
    """
    try:
        # Decode the token
        # Note: Django SimpleJWT uses HS256 by default
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        logger.error("Token has expired")
        raise Exception("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        raise Exception("Invalid token")
