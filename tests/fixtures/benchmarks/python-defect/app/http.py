from app.api import request_timeout


def client_options() -> dict[str, int]:
    return {"timeout_ms": request_timeout()}
