import pytest
from app.permissions import may_access


@pytest.mark.parametrize("number", range(120))
def test_ascii_principal_matrix(number: int) -> None:
    principal = f"USER-{number}"
    assert may_access(principal.lower(), {principal})


def test_unicode_casefold_contract() -> None:
    allowed = {"STRASSE@example.test"}
    assert may_access("Straße@example.test", allowed)
    assert may_access("  straße@EXAMPLE.TEST  ", allowed)
