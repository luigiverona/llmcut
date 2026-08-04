"""Irrelevant reporting implementation.

This module represents a substantial unrelated subsystem: monthly exports, currency conversion,
format negotiation, pagination, retry policies, legacy compatibility, audit metadata, rendering,
delivery scheduling, retention, and historical reconciliation. It is deliberately materialized as
source evidence so an honest provider-bound baseline must carry it while a task-scoped request can
leave it recoverable. None of these reporting concerns affect OAuth callback timeout units.
"""

REPORT_FORMATS = ("csv", "json", "parquet", "pdf")
