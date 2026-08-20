export interface TranslationDict {
  header: {
    spec_interceptor: string;
    agent: string;
    logs: string;
  };
  sidebar: {
    session_history: string;
    runs: string;
    newer_live_available: string;
    switch_to_live: string;
  };
  tabs: {
    summary: string;
    thoughts: string;
    files: string;
    console: string;
    diffs: string;
  };
  session_info: {
    launched_at: string;
    duration: string;
    workspace_files: string;
  };
  cognitive_loop: {
    title: string;
    status: string;
  };
  thought_stream: {
    title: string;
    desc: string;
    search_placeholder: string;
    category_all: string;
    phase_all: string;
    no_thoughts: string;
  };
  file_explorer: {
    files_count: string;
    no_files: string;
    select_file_placeholder: string;
    copied: string;
    copy_to_clipboard: string;
    copy_failed: string;
  };
  terminal: {
    terminal_io: string;
    rpc_ipc: string;
    terminal_tab: string;
    rpc_tab: string;
    no_terminal: string;
    no_ipc: string;
    call_sent: string;
    reply_rcvd: string;
    parameters: string;
    response_result: string;
    response_error: string;
  };
  diff_viewer: {
    no_diffs: string;
  };
  run_summary: {
    multi_step_plan: string;
    no_plan: string;
    elapsed_time: string;
    subprocess_pid: string;
    host_node: string;
  };
  phases: {
    perceive: string;
    analyze: string;
    plan: string;
    execute: string;
    verify: string;
    commit: string;
    completed: string;
    error: string;
    idle: string;
  };
  categories: {
    monologue: string;
    question: string;
    decision: string;
    check: string;
    system: string;
  };
  cognitive_loop_phases: {
    perceive: { label: string; desc: string };
    analyze: { label: string; desc: string };
    plan: { label: string; desc: string };
    execute: { label: string; desc: string };
    verify: { label: string; desc: string };
    commit: { label: string; desc: string };
    completed: { label: string; desc: string };
  };
}
