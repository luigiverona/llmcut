def retry_delay(attempt: int) -> int:
    return min(attempt * 2, 30)
