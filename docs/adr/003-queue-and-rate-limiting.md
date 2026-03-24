# ADR-003: Request Queue with 429 Rate Limiting

## Status
Accepted

## Context
The server has a fixed concurrency limit (default 4) due to CPU/GPU constraints. When all slots are busy, we need a strategy for excess requests.

### Options considered
1. **503 Service Unavailable** — immediate reject. Standard HTTP, but LiteLLM treats it as a hard failure.
2. **429 Too Many Requests** — reject with retry hint. OpenAI's standard. SDKs auto-retry.
3. **Queue and wait** — hold requests until a slot opens. Transparent to clients.

## Decision
Combine options 2 and 3:
- Requests that can't get a slot immediately enter a bounded queue (default depth 8)
- Queued requests wait for a slot, transparent to the client
- If the queue is also full, return 429 with `Retry-After: 5`
- If a queued request's context expires (request timeout), return 408

This matches OpenAI's API contract. The OpenAI Python SDK has built-in exponential backoff on 429. LiteLLM retries 429 before falling back to alternative models.

## Consequences
- Burst traffic is absorbed silently by the queue
- Sustained overload returns 429 with retry guidance
- Three retry layers: server queue → LiteLLM retry → SDK retry
- A client only sees an error after ~30s+ of sustained overload
- `stt_requests_queued` metric provides visibility into queue depth
