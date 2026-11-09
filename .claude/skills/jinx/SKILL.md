---
name: jinx
description: Runs a request through JINX. Use for every user message, greeting or task, and to go on with a JINX run in progress.
argument-hint: '<request> | --session <id>'
allowed-tools: Bash(python .agent/jinx.py *)
---
```!
python .agent/jinx.py --via /jinx --stdin <<'JINX_ARGS'
$ARGUMENTS
JINX_ARGS
```
