import os

os.environ.setdefault("NAV_AION_API_KEY", "test-nav-aion-key-123456")
os.environ.setdefault("NAV_AION_ALLOWED_ORIGINS", "http://testserver")

from fastapi.testclient import TestClient

from backend.api import app


client = TestClient(app)
API_KEY = os.environ["NAV_AION_API_KEY"]


def test_ui_is_public():
    response = client.get("/")
    assert response.status_code == 200
    assert "NAV-AION" in response.text


def test_api_requires_bearer_token():
    response = client.get("/health")
    assert response.status_code == 401


def test_api_rejects_wrong_token():
    response = client.get("/health", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401


def test_api_accepts_valid_token():
    response = client.get("/health", headers={"Authorization": f"Bearer {API_KEY}"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_docs_are_disabled():
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
