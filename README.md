# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/kobe258174/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/kobe258174/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Live free-model aggregator serving OpenAI- and Anthropic-compatible APIs via free OpenRouter models — make.fun bounty entry. One server, standard library only, deterministic routing across 15 free models in three quality tiers with failover.

## Run it

```sh
OPENROUTER_API_KEY=sk-or-v1-... python3 main.py
```

## How it works

- `GET /v1/models` lists the catalog; `python3 main.py --list` prints the same catalog document.
- `POST /v1/chat/completions` serves the OpenAI wire (JSON + SSE streaming, JSON mode, tools).
- `POST /v1/messages` serves the Anthropic wire (JSON + SSE streaming, system prompts, tools), converted to and from the OpenAI wire upstream.
- Routing is deterministic: fixed priority order by tier, no wall clock, no unseeded randomness. On HTTP 400, 404, 429, or 5xx (or a transport error) the server fails over to the next model, up to 4 attempts. The key comes only from `OPENROUTER_API_KEY` at runtime; nothing is stored in this repo.

## Test it

```sh
python3 main.py --list
curl http://127.0.0.1:8080/v1/models
```

Default port is `$PORT` (8080 if unset). See the design notes in `main.py` for the catalog and failover rules. Live-tested 2026-11-15 against free OpenRouter models: OpenAI + Anthropic chat, streaming, JSON mode, tool calls on both wires, and failover from a bogus model all passed.

---

<sub>Entered on [make.fun](https://make.fun/?ref={referral_code}&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
