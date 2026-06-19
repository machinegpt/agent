# Project Rules

ALWAYS route absolutely every single user message, greeting, or task through JINX. Do not reply directly or perform any actions yourself.


1. **Start JINX**: Run `python .agent/jinx.py "[user_message]"` (first time) or `python .agent/jinx.py` (to resume).
2. **Loop JINX**: If `.agent/jinx_request.yaml` is created:
   - `"llm_generate"`: Generate completion using YAML's system/messages/tools. Write `{"content": content_blocks}` to `.agent/jinx_response.yaml` (encoded in YAML format).
   - `"tool_calls"`: Execute YAML's tool calls. Write `{"results": [{"tool_use_id": "id", "content": "output"}]}` to `.agent/jinx_response.yaml` (encoded in YAML format).
   - Re-run `python .agent/jinx.py` and repeat the loop until `.agent/jinx_request.yaml` is gone.
