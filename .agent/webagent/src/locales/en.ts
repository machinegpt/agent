import { TranslationDict } from "./types";

export const en: TranslationDict = {
  header: {
    spec_interceptor: "MACHINE_GPT_LOOP // SPEC INTERCEPTOR",
    agent: "AGENT",
    logs: "-LOGS",
  },
  sidebar: {
    session_history: "Session History",
    runs: "runs",
    newer_live_available: "◉ Newer live session available",
    switch_to_live: "Switch to live",
  },
  tabs: {
    summary: "Summary",
    thoughts: "Thoughts",
    files: "Files",
    console: "Logs",
    diffs: "Patches",
  },
  session_info: {
    launched_at: "Launched at: ",
    duration: "Duration",
    workspace_files: "Workspace Files",
  },
  cognitive_loop: {
    title: "Cognitive Loop State",
    status: "Status",
  },
  thought_stream: {
    title: "Thought Stream & Monologue",
    desc: "Live feed of the agent's cognitive loops & decisions",
    search_placeholder: "Search thoughts...",
    category_all: "All Categories",
    phase_all: "All Phases",
    no_thoughts: "No thoughts matching your search",
  },
  file_explorer: {
    files_count: "files",
    no_files: "NO WORKSPACE FILES",
    select_file_placeholder: "SELECT A FILE FROM THE .AGENT WORKSPACE",
    copied: "Copied",
    copy_to_clipboard: "Copy to clipboard",
    copy_failed: "Copy failed",
  },
  terminal: {
    terminal_io: "Terminal Standard I/O Log",
    rpc_ipc: "JINX JSON-RPC IPC Interceptor",
    terminal_tab: "Terminal Output",
    rpc_tab: "JSON-RPC IPC Stream",
    no_terminal: "[ NO TERMINAL INTERACTION RECEIVED ]",
    no_ipc: "[ NO IPC MESSAGES RECORDED ]",
    call_sent: "Call Sent",
    reply_rcvd: "Reply Rcvd",
    parameters: "Parameters",
    response_result: "Response Result",
    response_error: "Response Exception",
  },
  diff_viewer: {
    no_diffs: "NO CODE DIFFS OR MODIFIED FILES GENERATED YET",
  },
  run_summary: {
    multi_step_plan: "Agent Multi-Step Plan Progress",
    no_plan: "No plan defined yet",
    elapsed_time: "Elapsed Time",
    subprocess_pid: "Subprocess PID",
    host_node: "Host Node",
  },
  phases: {
    perceive: "Perceive",
    analyze: "Analyze",
    plan: "Plan",
    execute: "Execute",
    verify: "Verify",
    commit: "Commit",
    completed: "Completed",
    error: "Error",
    idle: "Idle",
  },
  categories: {
    monologue: "Monologue",
    question: "Question",
    decision: "Decision",
    check: "Validation",
    system: "System",
  },
  cognitive_loop_phases: {
    perceive: {
      label: "Perceive",
      desc: "Scanner codebase & input conditions",
    },
    analyze: {
      label: "Analyze",
      desc: "Find root cause & vulnerabilities",
    },
    plan: {
      label: "Plan",
      desc: "Draft step-by-step resolution path",
    },
    execute: {
      label: "Execute",
      desc: "Apply file patches & execute tools",
    },
    verify: {
      label: "Verify",
      desc: "Run tests, linter & type checks",
    },
    commit: {
      label: "Commit",
      desc: "Save snapshots & git commits",
    },
    completed: {
      label: "Completed",
      desc: "Subprocess shut down, code committed",
    },
  },
};
