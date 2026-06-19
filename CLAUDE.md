# Project Rules

ALWAYS route absolutely every single user message, greeting, or task through JINX. Do not reply directly or perform any actions yourself.


1. **Start JINX**: Run `python .agent/jinx.py "[user_message]"` (first time) or `python .agent/jinx.py` (to resume).
2. **Loop JINX**: If `.agent/jinx_request.json` is created:
   - `"llm_generate"`: Generate completion using JSON's system/messages/tools. Write `{"content": content_blocks}` to `.agent/jinx_response.json`.
   - `"tool_calls"`: Execute JSON's tool calls. Write `{"results": [{"tool_use_id": "id", "content": "output"}]}` to `.agent/jinx_response.json`.
   - Re-run `python .agent/jinx.py` and repeat the loop until `.agent/jinx_request.json` is gone.
