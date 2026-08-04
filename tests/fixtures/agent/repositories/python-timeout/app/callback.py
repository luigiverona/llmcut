from app.config import TIMEOUT_SECONDS


def callback_timeout() -> int:
    return TIMEOUT_SECONDS * 1000
