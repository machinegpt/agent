---
name: jinx
description: Runs a request through JINX. Use for every user message, greeting or task, and to go on with a JINX run in progress.
---
`/jinx <arguments>` means running this in the project folder:

    python .agent/jinx.py <arguments>

Give the user's message as one quoted argument, exactly as written. To go on with a run, run the `resume` command from its `[JINX_SESSION]` line. The instruction file of the project says what to do with what JINX prints.
