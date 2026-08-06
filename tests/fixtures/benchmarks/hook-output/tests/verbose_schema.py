from app.schema import DEFAULT_POLICY, RetryPolicy, next_delay


def test_runtime_policy_contract() -> None:
    assert isinstance(DEFAULT_POLICY, RetryPolicy)
    assert next_delay(DEFAULT_POLICY, 2) == 0.5
