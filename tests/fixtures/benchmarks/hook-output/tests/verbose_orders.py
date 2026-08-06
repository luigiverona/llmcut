import pytest
from app.orders import discounted_total


@pytest.mark.parametrize("unit_cents", range(101, 221))
def test_zero_discount_matrix(unit_cents: int) -> None:
    assert discounted_total(unit_cents, 2, 0) == unit_cents * 2


def test_percentage_contract() -> None:
    assert discounted_total(2000, 2, 25) == 3000
