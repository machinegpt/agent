# Project Rules

By default, interact with the developer natively as Claude Code. Do NOT hijack general conversation, administrative queries, greetings, or basic commands.

Only invoke the JINX specialized agent loop when the user explicitly requests it (e.g., "run JINX", "start jinx", or a task prefix like "JINX: ...").


1. **Start JINX**: Run `python .agent/jinx.py "[user_message]"` (first time) or `python .agent/jinx.py` (to resume).
2. **Loop JINX**: If `.agent/jinx_request.json` is created:
   - `"llm_generate"`: Generate completion using JSON's system/messages/tools. Write `{"content": content_blocks}` to `.agent/jinx_response.json`.
   - `"tool_calls"`: Execute JSON's tool calls. Write `{"results": [{"tool_use_id": "id", "content": "output"}]}` to `.agent/jinx_response.json`.
   - Re-run `python .agent/jinx.py` and repeat the loop until `.agent/jinx_request.json` is gone.
