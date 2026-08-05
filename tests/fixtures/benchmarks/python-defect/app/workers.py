from app.transport import send_options


def callback_worker() -> dict[str, int]:
    return send_options()
