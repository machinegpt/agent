---
name: jinx
description: Runs a request through JINX. Use for every user message, greeting or task, and to go on with a JINX run in progress.
---
`/jinx <arguments>` means running this in the project folder, with your shell tool:

    python .agent/jinx.py <arguments>

Give the user's message as one quoted argument, exactly as written. What the command prints is what `/jinx` returns: its `[JINX_SESSION]` line has the `resume` command, which you run again and again until the run ends. Where the rules of the project say `[JINX_EXIT] <code>`, that is the exit code of this command when it is not 0 (1: the launch or the run failed; 2: the run reached its limit of rounds). The instruction file of the project says what to do with what JINX prints.
