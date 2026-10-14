import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, compute_red_flags, ExtractedInvoice, LineItem

client = TestClient(app)


class FakeInteraction:
    """Stands in for whatever client.interactions.create() actually returns.
    Our code only ever reads .output_text off that object, so that's the
    only thing our fake needs to have."""

    def __init__(self, output_text: str):
        self.output_text = output_text


def test_extract_empty_input_returns_400():
    response = client.post("/extract", json={"invoice_text": ""})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


@patch("app.main._call_gemini")
def test_extract_happy_path_no_red_flags(mock_call):
    fake_json = json.dumps({
        "vendor_name": "Accra Office Supplies Ltd",
        "invoice_date": "2026-09-15",
        "due_date": "2026-09-30",
        "total_amount": 1250.00,
        "currency": "GHS",
        "line_items": [
            {"description": "Printer paper", "amount": 750.00},
            {"description": "Ink cartridges", "amount": 500.00},
        ],
    })
    mock_call.return_value = FakeInteraction(fake_json)

    response = client.post("/extract", json={"invoice_text": "some invoice text"})

    assert response.status_code == 200
    body = response.json()
    assert body["extracted"]["vendor_name"] == "Accra Office Supplies Ltd"
    assert body["red_flags"] == []


@patch("app.main._call_gemini")
def test_extract_missing_due_date_is_flagged(mock_call):
    fake_json = json.dumps({
        "vendor_name": "Accra Office Supplies Ltd",
        "invoice_date": "2026-09-15",
        "due_date": None,
        "total_amount": 1250.00,
        "currency": "GHS",
        "line_items": [],
    })
    mock_call.return_value = FakeInteraction(fake_json)

    response = client.post("/extract", json={"invoice_text": "some invoice text"})

    assert response.status_code == 200
    red_flags = response.json()["red_flags"]
    assert any("due date" in flag.lower() for flag in red_flags)


@patch("app.main._call_gemini")
def test_extract_line_item_mismatch_is_flagged(mock_call):
    fake_json = json.dumps({
        "vendor_name": "Accra Office Supplies Ltd",
        "invoice_date": "2026-09-15",
        "due_date": "2026-09-30",
        "total_amount": 1250.00,
        "currency": "GHS",
        "line_items": [
            {"description": "Printer paper", "amount": 750.00},
            {"description": "Ink cartridges", "amount": 400.00},  # doesn't add up to 1250
        ],
    })
    mock_call.return_value = FakeInteraction(fake_json)

    response = client.post("/extract", json={"invoice_text": "some invoice text"})

    assert response.status_code == 200
    red_flags = response.json()["red_flags"]
    assert any("don't match" in flag for flag in red_flags)


@patch("app.main._call_gemini")
def test_extract_malformed_llm_output_returns_502(mock_call):
    mock_call.return_value = FakeInteraction("this is not valid JSON at all")

    response = client.post("/extract", json={"invoice_text": "some invoice text"})

    assert response.status_code == 502
    assert "didn't match the expected format" in response.json()["detail"]


def test_compute_red_flags_detects_duplicate_line_items():
    invoice = ExtractedInvoice(
        vendor_name="Test Vendor",
        invoice_date="2026-01-01",
        due_date="2026-01-15",
        total_amount=200.0,
        currency="GHS",
        line_items=[
            LineItem(description="Consulting", amount=100.0),
            LineItem(description="Consulting", amount=100.0),
        ],
    )
    flags = compute_red_flags(invoice)
    assert any("duplicate" in flag.lower() for flag in flags)