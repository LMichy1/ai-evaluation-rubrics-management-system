from fastapi.testclient import TestClient


def test_index_serves_frontend(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "AI Model Evaluation &amp; Rubric Management" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served(client: TestClient) -> None:
    for path in ("/static/app.js", "/static/styles.css"):
        assert client.get(path).status_code == 200


def test_frontend_does_not_shadow_api_or_docs(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert "/" not in client.get("/openapi.json").json()["paths"]
    assert client.get("/rubrics").status_code == 200


def test_index_has_service_status_region_and_retry(client: TestClient) -> None:
    html = client.get("/").text

    assert 'id="service-status"' in html
    assert 'role="status"' in html
    assert 'id="service-retry"' in html
