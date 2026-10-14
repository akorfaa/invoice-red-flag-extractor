import concurrent.futures
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="Invoice Red-Flag Extractor")
client = genai.Client()

MODEL_NAME = "gemini-3.8-flash"
TIMEOUT_SECONDS = 20


class LineItem(BaseModel):
    description: str = Field(description="What this line item is for.")
    amount: Optional[float] = Field(description="The amount for this line item, as a number with no currency symbol.")


class ExtractedInvoice(BaseModel):
    vendor_name: Optional[str] = Field(description="The name of the vendor or company issuing the invoice.")
    invoice_date: Optional[str] = Field(description="The invoice date, in whatever format it appears in the text.")
    due_date: Optional[str] = Field(description="The payment due date, if present. Null if not mentioned.")
    total_amount: Optional[float] = Field(description="The total amount due, as a number with no currency symbol.")
    currency: Optional[str] = Field(description="The currency code or symbol used, e.g. GHS, USD, $.")
    line_items: List[LineItem] = Field(default_factory=list, description="Individual line items on the invoice, if listed.")


class ExtractRequest(BaseModel):
    invoice_text: str


def _call_gemini(prompt: str):
    """Runs the actual Gemini call, asking for JSON matching our schema.
    Kept as its own function so we can run it in a worker thread and
    enforce our own timeout around it."""
    return client.interactions.create(
        model=MODEL_NAME,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": ExtractedInvoice.model_json_schema(),
        },
    )


def compute_red_flags(invoice: ExtractedInvoice) -> List[str]:
    """Pure Python checks on the extracted data — no LLM call involved.
    This is the part of the project we can actually unit test
    deterministically, unlike the LLM's own output."""
    flags = []

    if not invoice.vendor_name:
        flags.append("No vendor name found on this invoice.")
    if not invoice.due_date:
        flags.append("No due date found on this invoice.")

    if invoice.line_items and invoice.total_amount is not None:
        line_sum = sum(item.amount for item in invoice.line_items if item.amount is not None)
        if abs(line_sum - invoice.total_amount) > 0.01:
            flags.append(
                f"Line items sum to {line_sum:.2f} but the stated total is "
                f"{invoice.total_amount:.2f} — these don't match."
            )

    descriptions = [item.description.strip().lower() for item in invoice.line_items if item.description]
    duplicates = sorted({d for d in descriptions if descriptions.count(d) > 1})
    if duplicates:
        flags.append(f"Possible duplicate line items: {', '.join(duplicates)}")

    return flags


@app.post("/extract")
def extract_invoice(request: ExtractRequest):
    if not request.invoice_text.strip():
        raise HTTPException(status_code=400, detail="invoice_text cannot be empty")

    prompt = (
        "Extract the invoice details from this text. If a field isn't "
        "present in the text, leave it null rather than guessing.\n\n"
        f"Invoice text:\n{request.invoice_text}"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call_gemini, prompt)
        try:
            interaction = future.result(timeout=TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"LLM call timed out after {TIMEOUT_SECONDS}s — check network/API key",
            )
        except Exception as e:
            # The Interactions API (client.interactions.create) is a preview
            # feature as of this writing, and its error classes live under a
            # private module path (google.genai._gaos.lib.compat_errors) that
            # isn't part of the SDK's documented public surface. Importing
            # directly from a "_"-prefixed path is fragile, so instead we
            # branch on the exception's class NAME (always available via
            # type(e).__name__, no import required). Deliberate workaround
            # for a preview API's instability, not a permanent design choice.
            error_name = type(e).__name__
            message = str(e)

            if error_name == "RateLimitError" or "rate limit" in message.lower():
                raise HTTPException(
                    status_code=503,
                    detail="Rate limited by the LLM provider — try again in a moment.",
                )
            if error_name in ("AuthenticationError", "PermissionDeniedError", "BadRequestError"):
                print(f"ERROR — Gemini client error ({error_name}): {message}")
                raise HTTPException(
                    status_code=502,
                    detail="The LLM provider rejected this request — check server configuration.",
                )
            if error_name in ("InternalServerError", "APIStatusError", "ServerError"):
                print(f"ERROR — Gemini server error ({error_name}): {message}")
                raise HTTPException(
                    status_code=503,
                    detail="The LLM provider is having issues right now — try again shortly.",
                )

            print(f"ERROR — unexpected error ({error_name}): {message}")
            raise HTTPException(status_code=502, detail=f"LLM call failed unexpectedly: {message}")

    try:
        extracted = ExtractedInvoice.model_validate_json(interaction.output_text)
    except ValidationError as e:
        print(f"ERROR — LLM output didn't match expected schema: {e}")
        raise HTTPException(
            status_code=502,
            detail="The LLM returned data that didn't match the expected format.",
        )

    red_flags = compute_red_flags(extracted)

    return {
        "extracted": extracted.model_dump(),
        "red_flags": red_flags,
    }