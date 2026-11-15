# Live free-model aggregator

A single-file Python server (standard library only) that serves an
OpenAI-compatible API and an Anthropic-compatible API on one port,
routing every request to a free OpenRouter model. Built for the
make.fun bounty **"The best live free-model aggregator"** ($500).

## Files

- `main.py` — the whole server. Python 3.13, standard library only,
  ~30 KB. No third-party imports, no API key in the file.
- `README.md` — this file.

## Quick start

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."   # your own key; never commit it
export PORT=8080                            # optional, default 8080
python3 main.py
```

The server listens on `0.0.0.0:$PORT`.

Print the catalog document (byte-identical every run):

```bash
python3 main.py --list
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | List the catalog models (OpenAI format). |
| `POST` | `/v1/chat/completions` | OpenAI chat completions (JSON + SSE streaming). |
| `POST` | `/v1/messages` | Anthropic messages (JSON + SSE streaming). |
| `GET` | `/`, `/health` | Health check. |

## How it works

### Catalog

A static catalog of 15 free OpenRouter models is hardcoded in
`main.py`, built from the live OpenRouter model list (FreeToken-Bots
scan, checked 2026-11-14T23:56:35Z). Each model has a quality tier:

- **Tier 1** — strong general models: Ernie 5.5 flash, Nemotron 3
  Ultra 550B, Gemma 4 31B/26B, Nemotron 3.5 Lightning, Nemotron 3
  Nano reasoning.
- **Tier 2** — good general models with 1M context: Ling 3.2 flash,
  Inkling 2 small, Nemotron 3.5 Super VL, Particle 1.
- **Tier 3** — small/fast/specialist models plus the OpenRouter free
  router (`openrouter/free`) as a last resort.

`python3 main.py --list` prints this catalog as JSON, byte for byte
the same on every run (no timestamps, no randomness).

### Routing (deterministic)

For a given request the server tries models in a fixed priority
order (tier 1 first, then tier 2, then tier 3). If the client names
a model that is in the catalog, that model is tried first. The choice
depends only on the request, never on wall-clock time or unseeded
randomness, so the same request always picks the same model.

Every request is sent to OpenRouter as an OpenAI chat completion
(`POST https://openrouter.ai/api/v1/chat/completions`). Anthropic
requests are converted to the OpenAI wire for the upstream call and
converted back to Anthropic format for the response (including
`system` prompts, `tools`/`tool_use`, and streaming events).

### Failover

If a model refuses, the server fails over to the next model in the
priority order, up to 4 attempts, then returns the last error.
A refusal is any of:

- HTTP 400 (OpenRouter returns 400 for an unknown model id and for
  models that reject a feature), 404, 429, or 5xx;
- a transport error / timeout.

HTTP 401/403 (bad key) is **not** retried — no model will fix that.

### JSON mode, system prompts, tools

- **JSON mode**: pass `response_format: {"type": "json_object"}` on
  the OpenAI wire; it is forwarded to OpenRouter.
- **System prompts**: accepted on both wires (OpenAI `messages`
  with `role: "system"`, Anthropic `system` field).
- **Tools**: accepted on both wires. OpenAI `tools`/`tool_choice`
  are forwarded as-is. Anthropic `tools` are converted to OpenAI
  function tools; a model's `tool_calls` are surfaced as Anthropic
  `tool_use` blocks (including in streaming, as `input_json_delta`
  events). If a model returns no tool call, the text answer is
  returned normally.

### Streaming

- OpenAI wire: standard SSE `data: {...}` chunks ending with
  `data: [DONE]`.
- Anthropic wire: `message_start`, `content_block_start`,
  `content_block_delta` (text / input_json_delta),
  `content_block_stop`, `message_delta`, `message_stop`.

## Testing (verified 2026-11-15)

All tests ran against the live server on `127.0.0.1:18080` with a
free-tier OpenRouter key (50 requests/day). Each test below used one
real upstream call and returned a real completion.

| Test | Result |
|---|---|
| `python3 main.py --list` twice | Valid JSON, byte-identical, 15 models |
| `GET /v1/models` | 15 models, OpenAI format |
| OpenAI chat completion + system prompt | OK — Ernie 5.5 flash: "The capital of France is Paris." |
| OpenAI streaming | OK — SSE chunks, content "1\n2\n3", ends `data: [DONE]` |
| Anthropic message + system prompt | OK — Ernie 5.5 flash: "The capital of Japan is Tokyo." |
| Anthropic streaming | OK — `message_start` … `content_block_delta` "1\n2\n3" … `message_stop` |
| JSON mode (`response_format: json_object`) | OK — returned `{"city":"Paris","country":"France"}` |
| Tool call, OpenAI wire | OK — Ernie called `get_weather` with `{"city": "Paris"}` |
| Tool call, Anthropic wire | OK — `tool_use` block `get_weather`, `{"city": "Paris"}` |
| Failover (bogus model first) | OK — bogus model 400 → failed over to Ernie 5.5 flash, answered |

Notes:

- The free-tier key has a 50-request/day ceiling. If grading makes
  more than 50 calls in a day, later calls return 429 and the
  failover column records them. A key from an account with ≥10
  credits purchased raises the ceiling to 1,000/day.
- Free models occasionally return an empty response on a first try;
  the server forwards what the model returns. If a model is
  flaky, failover only triggers on HTTP errors, not on empty
  content.
- `created` is `0` and response ids are a SHA-256 of the request
  body, so responses are deterministic given the same request and
  the same upstream answers (no wall clock in the output).

## Notes for the bounty submission

- The server must be in a **public GitHub repository created after
  the competition opened** (2026-11-15T03:30:00Z), with `main.py`
  at the repository root, an OSI-approved `LICENSE`, and a
  `.makefun/owner.json` naming the payout wallet when the repo
  owner is not the account's GitHub login.
- The entry also needs the agent session transcripts (bundled with
  the kit's CLI) and an entry message signed by a Solana wallet on
  the account within 10 minutes of submitting.
- The OpenRouter key is **not** in the repository. Push it to the
  competition as an entry secret (`OPENROUTER_API_KEY`) before
  submitting, so the grader's gateway can call models on it.
