from app.service import delivery_options


def send_options() -> dict[str, int]:
    return delivery_options()
