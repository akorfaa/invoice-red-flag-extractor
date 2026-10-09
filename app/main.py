import concurrent.futures

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="Invoice Red-Flag Extractor")
client = genai.Client()

MODEL_NAME = "gemini-3.8-flash"
TIMEOUT_SECONDS = 20


class ExtractRequest(BaseModel):
    invoice_text: str


def _call_gemini(prompt: str):
    """Runs the actual Gemini call. Kept as its own function so we can
    run it in a worker thread and enforce a timeout around it — we can't
    fully trust that the SDK itself won't hang, so we add our own."""
    return client.interactions.create(model=MODEL_NAME, input=prompt)


@app.post("/extract")
def extract_invoice(request: ExtractRequest):
    if not request.invoice_text.strip():
        raise HTTPException(status_code=400, detail="invoice_text cannot be empty")

    prompt = (
        "Extract the vendor name, invoice date, and total amount from this "
        "invoice text. Reply in plain text for now.\n\n"
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
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    return {"result": interaction.output_text}