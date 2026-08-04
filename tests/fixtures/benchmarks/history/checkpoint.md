# Verified checkpoint

Objective: make transient job delivery use three attempts. Constraints: preserve backoff and
validation. Accepted decision: change only the worker attempt count. Rejected alternative: retry
forever. Changed files: none yet. Unresolved work: update worker and run validation.
