# Build Plan — Project 3: Invoice Red-Flag Extractor

Tier 1 (junior). Proves: integrating a real LLM API call into a product with
error handling, structured output, and tests — not a notebook script.

## What it does
Paste or upload invoice/receipt text -> one LLM call -> structured JSON
(vendor, date, amount, due date, line items) plus a plain-English list of
anything that looks off (missing due date, line items that don't sum to the
total, no vendor ID, duplicate-looking lines).

## Sessions
- Session 1: provider + API key setup, project scaffold, first working
  LLM call (raw text response, no parsing yet), smoke test.
- Session 2: structured JSON output, error handling (timeouts, malformed
  JSON, empty input), red-flag logic, first pytest tests (mocked LLM calls).
- Session 3: simple HTML/JS frontend, wire to backend, manual QA with
  sample invoices.
- Session 4: deploy, README, decision-focused write-up, update
  _docs/roadmap-status.md and _docs/progress-log.md.

## Decisions log
- 2026-10-09: Considered Claude API (cost/card friction from Ghana),
  then Groq (free tier unclear/possibly changed), settling provider in
  Session 1 step 1 based on what's actually available without a card.

- 2026-10-09: Settled on Google AI Studio / Gemini API (free tier, no card
  seen required) after Claude API (card/cost friction from Ghana) and Groq
  (free tier unclear on the pricing page) were ruled out. Verified the real
  quickstart docs before writing code rather than guessing package/model
  names (package: google-genai, env var: GEMINI_API_KEY, model used:
  gemini-3.8-flash, API shape: client.interactions.create(...)).
- 2026-10-09: Session 1 finding - Gemini free-tier latency is inconsistent.
  Same request succeeded twice and hit our 20s timeout twice in the same
  short test run. Added an app-level timeout (ThreadPoolExecutor +
  future.result(timeout=20)) rather than trusting the SDK to fail fast on
  its own. Open question for Session 2: is this plain latency variance, or
  the SDK silently retrying on a rate-limit response until it gives up?
  We'll know once we narrow the broad except-Exception into specific
  exception types and can see what actually gets raised.
