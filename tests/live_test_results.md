# Live API Test Results

## 2026-04-24 00:25 UTC

### Environment
- Python 3.12, archon 0.1.0
- Tested providers: Anthropic, OpenAI

---

### Test 1 — Simple completion (no tools)

| Provider  | Model            | Prompt                              | Response                           | Tokens | Cost       |
|-----------|------------------|-------------------------------------|------------------------------------|--------|------------|
| Anthropic | claude-haiku-4-5 | What is the capital of France?      | The capital of France is Paris.    | 608    | $0.000518  |
| OpenAI    | gpt-4o-mini      | What is the capital of Germany?     | The capital of Germany is Berlin.  | 82     | $0.000016  |

### Test 2 — Tool calling / ReAct loop

Registered tool: `get_weather(location: str) -> str`  
Prompt: "What is the weather in Tokyo?"

| Provider  | Model            | Output                                                         | Iterations | Tokens | Cost       |
|-----------|------------------|----------------------------------------------------------------|------------|--------|------------|
| Anthropic | claude-haiku-4-5 | The current weather in Tokyo is **28°C** (approximately 82°F). It's quite warm! | 2 | 1474 | $0.001435 |
| OpenAI    | gpt-4o-mini      | The current weather in Tokyo is 28°C.                         | 2          | 262    | $0.000051  |

### Result
Both providers passed simple completion and full ReAct tool-calling loop. ✅
