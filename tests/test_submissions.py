from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SUBMISSION_PAYLOAD, is_utc_iso8601


def test_create_submission(client: TestClient) -> None:
    response = client.post("/submissions", json=SUBMISSION_PAYLOAD)

    assert response.status_code == 201
    body = response.json()
    assert body["id"] > 0
    assert body["model_metadata"] == SUBMISSION_PAYLOAD["model_metadata"]
    assert body["reference_answer"] == SUBMISSION_PAYLOAD["reference_answer"]
    assert body["evaluations"] == []
    assert is_utc_iso8601(body["created_at"])


def test_optional_fields_may_be_omitted(client: TestClient) -> None:
    response = client.post(
        "/submissions", json={"prompt": "Hi", "output": "Hello!", "model_name": "m"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["model_version"] is None
    assert body["model_metadata"] is None
    assert body["reference_answer"] is None


def test_required_fields_are_enforced(client: TestClient) -> None:
    response = client.post("/submissions", json={"prompt": "Hi", "output": ""})

    assert response.status_code == 422
    errors = {tuple(e["loc"]) for e in response.json()["detail"]}
    assert ("body", "output") in errors
    assert ("body", "model_name") in errors


def test_get_submission_includes_evaluations(
    client: TestClient, submission: dict[str, Any], score: Any
) -> None:
    assert score([5, 4, 3]).status_code == 201

    response = client.get(f"/submissions/{submission['id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["prompt"] == SUBMISSION_PAYLOAD["prompt"]
    assert len(body["evaluations"]) == 1
    assert body["evaluations"][0]["rater"]["external_id"] == "rater-001"


def test_get_unknown_submission_returns_404(client: TestClient) -> None:
    response = client.get("/submissions/12345")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "SUBMISSION_NOT_FOUND"


def test_list_submissions_newest_first_with_counts(
    client: TestClient, submission: dict[str, Any], score: Any
) -> None:
    score([5, 4, 3])  # scores the fixture submission (id 1)
    client.post("/submissions", json={**SUBMISSION_PAYLOAD, "prompt": "Why? " * 100})

    response = client.get("/submissions")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    newest, oldest = body["items"]
    assert (newest["id"], newest["evaluation_count"]) == (2, 0)
    assert (oldest["id"], oldest["evaluation_count"]) == (submission["id"], 1)
    assert oldest["prompt_preview"] == SUBMISSION_PAYLOAD["prompt"]
    assert len(newest["prompt_preview"]) <= 120
    assert newest["prompt_preview"].endswith("…")
    assert "output" not in newest


def test_list_submissions_paginates(client: TestClient) -> None:
    for _ in range(3):
        client.post("/submissions", json=SUBMISSION_PAYLOAD)

    body = client.get("/submissions", params={"limit": 1, "offset": 1}).json()

    assert body["total"] == 3
    assert [s["id"] for s in body["items"]] == [2]
    assert client.get("/submissions", params={"limit": 0}).status_code == 422


# --- Filtering by model name -----------------------------------------------------------

FILTER_MODELS = ["demo-llm", "demo-llm-1", "other-model", "Demo-LLM-Pro", "eval_100%"]


@pytest.fixture
def model_submissions(client: TestClient) -> dict[str, int]:
    """One submission per model name, created in FILTER_MODELS order (ids 1..5)."""
    ids = {}
    for name in FILTER_MODELS:
        response = client.post("/submissions", json={**SUBMISSION_PAYLOAD, "model_name": name})
        ids[name] = response.json()["id"]
    return ids


def _filter(client: TestClient, model_name: str, **params: Any) -> dict[str, Any]:
    response = client.get("/submissions", params={"model_name": model_name, **params})
    assert response.status_code == 200, response.text
    return response.json()


def _names(body: dict[str, Any]) -> list[str]:
    return [item["model_name"] for item in body["items"]]


def test_unfiltered_list_is_unchanged(client: TestClient, model_submissions: dict) -> None:
    body = client.get("/submissions").json()

    assert body["total"] == len(FILTER_MODELS)
    assert _names(body) == list(reversed(FILTER_MODELS))


def test_filter_exact_model_name(client: TestClient, model_submissions: dict) -> None:
    body = _filter(client, "demo-llm-1")

    assert _names(body) == ["demo-llm-1"]
    assert body["total"] == 1


def test_filter_is_case_insensitive(client: TestClient, model_submissions: dict) -> None:
    assert _names(_filter(client, "DEMO-llm-pro")) == ["Demo-LLM-Pro"]


def test_filter_matches_substrings_and_excludes_other_models(
    client: TestClient, model_submissions: dict
) -> None:
    body = _filter(client, "demo-llm")

    assert _names(body) == ["Demo-LLM-Pro", "demo-llm-1", "demo-llm"]  # newest first
    assert "other-model" not in _names(body)
    assert _names(_filter(client, "llm")) == ["Demo-LLM-Pro", "demo-llm-1", "demo-llm"]


def test_filter_with_no_matches_returns_empty_page(
    client: TestClient, model_submissions: dict
) -> None:
    body = _filter(client, "nonexistent-model")

    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_filter_preserves_pagination_and_ordering(
    client: TestClient, model_submissions: dict
) -> None:
    first = _filter(client, "demo-llm", limit=2, offset=0)
    second = _filter(client, "demo-llm", limit=2, offset=2)

    assert first["total"] == second["total"] == 3
    assert _names(first) == ["Demo-LLM-Pro", "demo-llm-1"]
    assert _names(second) == ["demo-llm"]


def test_filter_trims_surrounding_whitespace(client: TestClient, model_submissions: dict) -> None:
    assert _filter(client, "  demo-llm-1\t") == _filter(client, "demo-llm-1")


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_filter_means_no_filter(
    client: TestClient, model_submissions: dict, blank: str
) -> None:
    assert _filter(client, blank)["total"] == len(FILTER_MODELS)


def test_like_wildcards_in_filter_match_literally(
    client: TestClient, model_submissions: dict
) -> None:
    assert _names(_filter(client, "%")) == ["eval_100%"]
    assert _names(_filter(client, "_")) == ["eval_100%"]
    assert _filter(client, "demo%llm")["total"] == 0


def test_overlong_filter_is_rejected(client: TestClient) -> None:
    response = client.get("/submissions", params={"model_name": "x" * 201})

    assert response.status_code == 422
