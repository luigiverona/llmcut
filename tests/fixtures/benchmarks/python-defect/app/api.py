from app.callback import callback_timeout


def request_timeout() -> int:
    return callback_timeout()
