# SPDX-License-Identifier: Apache-2.0
"""The /metrics Prometheus endpoint on the omni FastAPI app.

omni runs its own FastAPI app (not sglang's http_server), so /metrics is wired
here in create_app rather than inherited. These tests exercise the endpoint and
the request-instrumentation middleware without a live pipeline — the /v1/models
route touches only app.state, no Client.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from sglang_omni.serve import create_app


class _StubClient:
    """Minimal stand-in: create_app only stores it; /v1/models never calls it."""

    def health(self) -> dict:
        return {"running": True}


def _client() -> TestClient:
    return TestClient(create_app(_StubClient(), model_name="test-model"))


def test_metrics_endpoint_serves_prometheus_exposition():
    c = _client()
    r = c.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "sglang_omni_http_requests_total" in r.text
    assert "sglang_omni_http_request_duration_seconds" in r.text


def test_request_is_recorded_under_its_route_template():
    c = _client()
    assert c.get("/v1/models").status_code == 200
    body = c.get("/metrics").text
    # counted by route template + status, with a latency sample
    assert (
        'sglang_omni_http_requests_total{method="GET",path="/v1/models",status="200"}'
        in body
    )
    assert (
        'sglang_omni_http_request_duration_seconds_count{method="GET",path="/v1/models"}'
        in body
    )


def test_scrape_endpoint_itself_is_not_counted():
    c = _client()
    c.get("/metrics")
    body = c.get("/metrics").text
    assert 'path="/metrics"' not in body
