import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import app  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
REF = FIXTURES / "ref_real.jpeg"
CUR_DEFECTIVE = FIXTURES / "cur_real.jpeg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_load_reference(client):
    r = client.post(
        "/load_reference",
        data={"object_id": "test_plate"},
        files={"image": ("ref.jpeg", REF.read_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object_id"] == "test_plate"
    assert body["reference_count"] >= 1


def test_load_reference_invalid_bytes(client):
    r = client.post(
        "/load_reference",
        data={"object_id": "bad"},
        files={"image": ("x.png", b"not an image", "image/png")},
    )
    assert r.status_code == 400


def test_inspect_unknown_object(client):
    r = client.post(
        "/inspect_detail",
        data={"object_id": "no_such_object"},
        files={"image": ("x.jpeg", REF.read_bytes(), "image/jpeg")},
    )
    assert r.status_code == 404


def test_inspect_defective(client):
    r = client.post(
        "/inspect_detail",
        data={"object_id": "test_plate"},
        files={"image": ("cur.jpeg", CUR_DEFECTIVE.read_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["alignment"]["mask_iou"] > 0.7
    assert body["defects"], "defects must be detected on the defective part"
    for d in body["defects"]:
        x0, y0, x1, y1 = d["bbox_xyxy"]
        assert 0 <= x0 < x1 and 0 <= y0 < y1
        assert 0.0 <= d["max_score"] <= 1.0
    # annotated image decodes to a valid PNG
    png = base64.b64decode(body["annotated_image_b64"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_inspect_reference_against_itself(client):
    """Негативный кейс: эталон против самого себя — дефектов быть не должно."""
    r = client.post(
        "/inspect_detail",
        data={"object_id": "test_plate"},
        files={"image": ("ref.jpeg", REF.read_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["alignment"]["mask_iou"] > 0.9
    assert body["defects"] == []
    assert body["image_score"] < 0.5


def test_delete_object(client):
    r = client.delete("/objects/test_plate")
    assert r.status_code == 200
    r = client.delete("/objects/test_plate")
    assert r.status_code == 404
