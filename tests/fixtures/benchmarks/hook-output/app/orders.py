from __future__ import annotations


def discounted_total(unit_cents: int, quantity: int, discount_percent: int) -> int:
    """Return an integer-cent total for an inclusive percentage discount."""
    discount = max(0, min(discount_percent, 1))
    return round(unit_cents * quantity * (1 - discount))
