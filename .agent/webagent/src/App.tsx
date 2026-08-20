/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Terminal,
  History,
  Trash2,
  FolderOpen,
  LayoutDashboard,
  GitPullRequest,
  FileCode,
  Activity,
  Check,
  X,
  RefreshCw,
  AlertCircle,
  HelpCircle,
  MoreVertical,
  Download,
  Copy
} from "lucide-react";
import { AgentSession } from "./types";
import CognitiveLoop from "./components/CognitiveLoop";
import ThoughtStream from "./components/ThoughtStream";
import FileExplorer from "./components/FileExplorer";
import TerminalConsole from "./components/TerminalConsole";
import DiffViewer from "./components/DiffViewer";
import RunSummary from "./components/RunSummary";
import {
  getSavedSessions,
  saveSessions,
  parseAgentFolder,
  createDefaultLiveSession
} from "./utils";
import { useLanguage } from "./context/LanguageContext";

export default function App() {
  const { language, setLanguage, t } = useLanguage();

  const [sessions, setSessions] = useState<AgentSession[]>(() => {
    const saved = getSavedSessions();
    if (saved.length > 0) {
      return saved;
    }
    const defaultLive = createDefaultLiveSession();
    saveSessions([defaultLive]);
    return [defaultLive];
  });

  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    return localStorage.getItem("jinx_active_session_id") || "live-session";
  });

  // Track previous live session status to detect genuine task completion/error.
  // Initialized to a sentinel so the first poll establishes a baseline without
  // archiving. Reset on unmount so React Strict Mode doesn't trigger spurious archives.
  const prevStatusRef = useRef<string | null>(null);
  const archivedTerminalKeysRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    return () => { prevStatusRef.current = null; archivedTerminalKeysRef.current.clear(); };
  }, []);

  const [activeTab, setActiveTab] = useState<"summary" | "thoughts" | "files" | "console" | "diffs">(() => {
    const saved = localStorage.getItem("jinx_active_tab");
    if (saved === "summary" || saved === "thoughts" || saved === "files" || saved === "console" || saved === "diffs") {
      return saved;
    }
    return "summary";
  });

  const [confirmReset, setConfirmReset] = useState(false);

  // Live polling state
  const [livePollActive, setLivePollActive] = useState<boolean>(() => {
    return localStorage.getItem("jinx_live_poll_active") !== "false";
  });
  const [liveError, setLiveError] = useState<string | null>(null);
  const [searchedPaths, setSearchedPaths] = useState<string[]>([]);
  const [lastSyncedAt, setLastSyncedAt] = useState<string | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);
  const [connectionMode, setConnectionMode] = useState<"sse" | "polling">("sse");

  // Auth token - stored in localStorage so it survives page reloads
  const [apiToken, setApiToken] = useState<string | null>(() => {
    return localStorage.getItem("jinx_api_token");
  });

  // Rename Session state
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameText, setRenameText] = useState("");

  // Three-dot menu state
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);

  useEffect(() => {
    if (!menuOpenId) return;
    const handler = (e: MouseEvent) => {
      const el = e.target as HTMLElement;
      if (!el.closest('[data-menu-root="true"]')) {
        setMenuOpenId(null);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [menuOpenId]);

  // Refs that always mirror the latest sessions/activeSessionId/apiToken. The 2s
  // polling interval below is only re-created when livePollActive changes,
  // so fetchLiveSession's own closure can go stale; reading through these
  // refs instead of the outer state avoids using outdated values when the
  // interval callback fires.
  const sessionsRef = useRef(sessions);
  const activeSessionIdRef = useRef(activeSessionId);
  const apiTokenRef = useRef(apiToken);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    apiTokenRef.current = apiToken;
  }, [apiToken]);

  // Pure function: merges a new live session into the session list.
  // Extracted from applyLiveSessionData to avoid re-creating it on every poll tick.
  const mergeLiveSession = useCallback((prev: AgentSession[], newSession: AgentSession): AgentSession[] => {
    const existingIdx = prev.findIndex((s) => s.id === newSession.id);
    if (existingIdx >= 0) {
      const merged = [...prev];
      merged[existingIdx] = newSession;
      return merged;
    }
    // Remove ALL idle live- sessions (live-session, live-session-1, etc.)
    // to prevent duplicates when JINX restarts with a new task
    return [newSession, ...prev.filter(
      (s) => !(s.id.startsWith("live-") && s.status === "idle")
    )];
  }, []);

  // Reusable handler for live session data from either SSE or polling.
  const applyLiveSessionData = useCallback((data: any) => {
    if (data.exists) {
      setLiveError(null);
      setSearchedPaths([]);
      setLastSyncedAt(new Date().toLocaleTimeString());

      const newSession: AgentSession = data.session;
      const currentStatus = newSession.status;

      // Terminal states that should be archived into a dedicated session entry.
      const isTerminal = currentStatus === "completed" || currentStatus === "error";
      const terminalKey = `${newSession.id}:${currentStatus}`;

      // First poll: establish baseline without archiving.
      if (prevStatusRef.current === null) {
        prevStatusRef.current = currentStatus;
        // If the live session is already terminal on cold start, still surface
        // it — but don't create a separate archive (avoids duplicates on reload).
        if (isTerminal) {
          const nextSessions = mergeLiveSession(sessionsRef.current, newSession);
          setSessions(nextSessions);
          if (!nextSessions.find((s) => s.id === activeSessionIdRef.current)) {
            setActiveSessionId(newSession.id);
          }
          return;
        }
      } else if (isTerminal && prevStatusRef.current !== currentStatus) {
        if (archivedTerminalKeysRef.current.has(terminalKey)) {
          prevStatusRef.current = currentStatus;
          return;
        }
        // Genuine transition to a terminal state — archive as a dedicated
        // session and spin up a fresh live slot.
        const archivedId = `${currentStatus}-${Date.now()}`;
        setSessions((prev) => {
          const archived = { ...newSession, id: archivedId, copyCount: 0 };
          const defaultLive = createDefaultLiveSession();
          archivedTerminalKeysRef.current.add(terminalKey);
          return [defaultLive, archived, ...prev.filter((s) => s.id !== newSession.id)];
        });
        setActiveSessionId(archivedId);
        setActiveTab("summary");
        prevStatusRef.current = currentStatus;
        return;
      }
      prevStatusRef.current = currentStatus;

      // Use functional updater to avoid race conditions on rapid polls.
      setSessions((prev) => {
        const nextSessions = mergeLiveSession(prev, newSession);
        return nextSessions;
      });
      // After merge, ensure the active session is valid.
      // Note: can't read the just-set value synchronously, so use a microtask.
      queueMicrotask(() => {
        setSessions((current) => {
          if (!current.find((s) => s.id === activeSessionIdRef.current)) {
            setActiveSessionId(newSession.id);
          }
          return current;
        });
      });
    } else {
      setLiveError(data.message || "No .agent folder found.");
      setSearchedPaths(data.searchedPaths || []);
    }
  }, []);

  const fetchLiveSession = useCallback(async (silent = false) => {
    if (!silent) setIsSyncing(true);
    try {
      const headers: Record<string, string> = {};
      const token = apiTokenRef.current;
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
      const response = await fetch("/api/live-session", { headers });

      if (response.status === 401) {
        localStorage.removeItem("jinx_api_token");
        setApiToken(null);
        setLiveError("Unauthorized: valid DASHBOARD_API_TOKEN required.");
        return;
      }

      const data = await response.json();
      applyLiveSessionData(data);
    } catch (err) {
      console.error("Live fetch failed", err);
      setLiveError("Failed to communicate with local dashboard backend server.");
    } finally {
      if (!silent) setIsSyncing(false);
    }
  }, [applyLiveSessionData]);

  // Connect to live session — SSE when no token, polling with auth headers
  // when token is set (EventSource cannot set custom headers).
  useEffect(() => {
    const token = apiTokenRef.current;

    if (token) {
      // Polling mode (supports Authorization header)
      setConnectionMode("polling");
      fetchLiveSession();
      if (!livePollActive) return;
      const interval = setInterval(() => fetchLiveSession(true), 2000);
      return () => clearInterval(interval);
    }

    // SSE mode (no token — EventSource can't send custom headers)
    setConnectionMode("sse");
    if (!livePollActive) return;

    const es = new EventSource("/api/live-session/stream");
    es.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        applyLiveSessionData(data);
      } catch (e) {
        console.error("SSE parse error", e);
      }
    };
    es.onerror = () => {
      console.error("SSE connection error, auto-reconnecting...");
    };

    return () => es.close();
  }, [livePollActive, apiToken, fetchLiveSession, applyLiveSessionData]);

  // Check if backend requires auth but we don't have a token yet
  useEffect(() => {
    if (apiToken) return;
    fetch("/api/auth-check")
      .then(r => r.json())
      .then(data => {
        if (data.tokenConfigured) {
          const token = prompt("DASHBOARD_API_TOKEN is set on the server. Enter the token:");
          if (token) {
            localStorage.setItem("jinx_api_token", token);
            setApiToken(token);
          }
        }
      })
      .catch(() => {});
  }, [apiToken]);

  // Sync state preferences with LocalStorage
  useEffect(() => {
    localStorage.setItem("jinx_active_session_id", activeSessionId);
  }, [activeSessionId]);

  useEffect(() => {
    localStorage.setItem("jinx_active_tab", activeTab);
  }, [activeTab]);

  useEffect(() => {
    localStorage.setItem("jinx_live_poll_active", String(livePollActive));
  }, [livePollActive]);

  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  const currentSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  // Upload backup JSON file or .agent folder files
  const handleBackupUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = e.target.files;
    if (!uploadedFiles || uploadedFiles.length === 0) return;

    try {
      const fileArray = Array.from(uploadedFiles);

      // If it's a single .json file, try to parse it as a session backup
      if (fileArray.length === 1 && fileArray[0].name.endsWith(".json")) {
        const text = await fileArray[0].text();
        const parsed: AgentSession = JSON.parse(text);
        if (parsed && typeof parsed.id === "string" && typeof parsed.name === "string" && typeof parsed.timestamp === "string" && Array.isArray(parsed.plan) && Array.isArray(parsed.thoughts)) {
          setSessions((prev) => {
            const existing = prev.find(s => s.id === parsed.id);
            if (existing) {
              return prev.map(s => s.id === parsed.id ? { ...s, copyCount: (s.copyCount || 0) + 1 } : s);
            }
            const updated = [...prev, parsed];
            return updated;
          });
          setActiveSessionId(parsed.id);
          setActiveTab("summary");
          return;
        }
      }

      // Otherwise treat as .agent folder files
      const parsedSession = await parseAgentFolder(fileArray);
      setSessions((prev) => {
        const updated = [...prev, parsedSession];
        return updated;
      });
      setActiveSessionId(parsedSession.id);
      setActiveTab("summary");
    } catch (err) {
      console.error("Backup upload failure", err);
      alert("Failed to parse backup. Ensure the file is a valid session backup (.json) or .agent folder contents.");
    } finally {
      e.target.value = "";
    }
  };

  const deleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = sessions.filter((s) => s.id !== id);
    setSessions(updated);
    if (activeSessionId === id && updated.length > 0) {
      setActiveSessionId(updated[0].id);
    } else if (updated.length === 0) {
      const defaultLive = createDefaultLiveSession();
      setSessions([defaultLive]);
      setActiveSessionId(defaultLive.id);
    }
  };

  const startRename = (id: string, name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(id);
    setRenameText(name);
  };

  const saveRename = (id: string) => {
    if (!renameText.trim()) return;
    const updated = sessions.map((s) => {
      if (s.id === id) {
        return { ...s, name: renameText.trim() };
      }
      return s;
    });
    setSessions(updated);
    setRenamingId(null);
  };

  const saveArchive = (session: AgentSession, e: React.MouseEvent) => {
    e.stopPropagation();
    setMenuOpenId(null);
    const blob = new Blob([JSON.stringify(session, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `jinx-session-${session.id}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const resetAllHistory = () => {
    const defaultLive = createDefaultLiveSession();
    setSessions([defaultLive]);
    setActiveSessionId(defaultLive.id);
    setConfirmReset(false);
  };

  return (
    <div className="min-h-screen bg-[#050505] bg-grid text-neutral-100 flex flex-col font-sans selection:bg-[#4ade80]/20 selection:text-[#4ade80] relative">
      {/* Decorative Subtle Grid overlay */}
      <div className="absolute inset-0 bg-grid pointer-events-none opacity-60 z-0" />

      {/* Primary Navigation Header */}
      <header className="px-6 py-8 md:py-12 flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-50 bg-transparent border-none">
        <div className="flex flex-col z-10">
          <span className="text-[11px] tracking-[0.35em] font-extrabold text-[#4ade80]/60 uppercase mb-3 font-mono">
            {t.header.spec_interceptor || "MACHINE_GPT // LOCAL DATA INTERCEPTOR"}
          </span>
          <h1 className="text-5xl md:text-8xl font-extrabold md:font-black tracking-tighter uppercase leading-none select-none m-0 p-0 flex flex-wrap md:flex-nowrap items-baseline gap-x-4">
            <span className="text-white drop-shadow-[0_4px_12px_rgba(255,255,255,0.05)]">{t.header.agent}</span>
            <span className="text-[#4ade80] drop-shadow-[0_4px_12px_rgba(74,222,128,0.1)]">{t.header.logs}</span>
          </h1>
        </div>

        {/* Global actions and Status section */}
        <div className="flex items-center gap-3 z-10 self-stretch md:self-auto justify-between md:justify-end border-t md:border-t-0 border-white/5 pt-3 md:pt-0">
          <div className="text-right flex flex-col items-end gap-1.5">
            <div className="flex items-center gap-2">
              {/* Language Switcher */}
              <div className="flex items-center gap-1.5 bg-neutral-900 border border-white/10 p-0.5 rounded font-mono text-xs">
                <button
                  onClick={() => setLanguage("en")}
                  className={`px-2 py-1 rounded text-[10px] font-bold uppercase transition-colors cursor-pointer ${
                    language === "en" ? "bg-white text-black font-extrabold" : "text-neutral-400 hover:text-white"
                  }`}
                >
                  EN
                </button>
                <button
                  onClick={() => setLanguage("ru")}
                  className={`px-2 py-1 rounded text-[10px] font-bold uppercase transition-colors cursor-pointer ${
                    language === "ru" ? "bg-white text-black font-extrabold" : "text-neutral-400 hover:text-white"
                  }`}
                >
                  RU
                </button>
              </div>

              <div className="px-3 py-1 border border-white/10 rounded-full text-[11px] font-mono bg-black/40 text-neutral-300">
                MONITORING v1.2.3
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Active Session Badge */}
              <div className={`px-3 py-1 rounded-full text-[10px] font-extrabold uppercase tracking-widest flex items-center gap-1.5 shadow-lg ${
                liveError
                  ? "bg-amber-950/40 text-amber-500 border border-amber-500/20"
                  : "bg-[#4ade80] text-black shadow-[0_0_15px_rgba(74,222,128,0.2)]"
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${liveError ? "bg-amber-500 animate-pulse" : "bg-black animate-ping"}`}></span>
                {liveError
                  ? (language === "ru" ? "ОЖИДАНИЕ АГЕНТА" : "WAITING FOR AGENT")
                  : (language === "ru" ? "АКТИВНЫЙ МОНИТОРИНГ" : "LIVE MONITOR")
                }
              </div>

              {/* Reset History button */}
              {!confirmReset ? (
                <button
                  id="reset-history-btn"
                  onClick={() => setConfirmReset(true)}
                  className="p-1.5 rounded border border-white/10 bg-neutral-950 hover:bg-neutral-900 hover:text-red-400 transition-colors text-neutral-500"
                  title="Wipe Saved History Logs"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              ) : (
                <div className="flex items-center gap-1.5 bg-rose-950/40 border border-rose-500/20 px-2 py-1 rounded text-xs animate-in fade-in zoom-in-95 duration-150">
                  <span className="text-[10px] text-rose-300 font-bold uppercase tracking-wider font-mono">
                    {language === "ru" ? "Стереть?" : "Wipe?"}
                  </span>
                  <button
                    id="reset-history-btn-confirm"
                    onClick={resetAllHistory}
                    className="p-1 rounded bg-rose-600 hover:bg-rose-500 text-white transition-colors flex items-center justify-center"
                    title="Confirm Delete"
                  >
                    <Check className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => setConfirmReset(false)}
                    className="p-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-400 transition-colors flex items-center justify-center"
                    title="Cancel"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 flex flex-col lg:flex-row relative z-10 max-w-7xl w-full mx-auto p-4 md:p-6 gap-6">
        {/* Sidebar Panel */}
        <aside className="w-full lg:w-80 flex-shrink-0 flex flex-col gap-6 relative z-10 order-2 lg:order-none">
          {/* Live Monitor Controls */}
          <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-5 shadow-xl space-y-4">
            <h3 className="text-[11px] uppercase tracking-[0.2em] font-extrabold text-neutral-400 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-[#4ade80]" />
                {language === "ru" ? "Управление Монитором" : "Monitor Controls"}
              </span>
              <span className={`inline-block w-2 h-2 rounded-full ${livePollActive ? "bg-[#4ade80] animate-pulse" : "bg-neutral-600"}`}></span>
            </h3>

            <div className="flex flex-col gap-2 font-mono text-xs">
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-neutral-500">{language === "ru" ? "Обновление" : "Live Refresh"}</span>
                <button
                  onClick={() => setLivePollActive(!livePollActive)}
                  className={`px-2 py-0.5 rounded font-bold text-[10px] tracking-wider transition-colors cursor-pointer uppercase ${
                    livePollActive ? "bg-[#4ade80]/15 text-[#4ade80] border border-[#4ade80]/20" : "bg-neutral-800 text-neutral-400 border border-transparent"
                  }`}
                >
                  {livePollActive ? (language === "ru" ? "ВКЛ" : "ON") : (language === "ru" ? "ВЫКЛ" : "OFF")}
                </button>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-neutral-500">{language === "ru" ? "Транспорт" : "Transport"}</span>
                <span className={`text-[10px] font-extrabold tracking-widest ${connectionMode === "sse" ? "text-[#4ade80]" : "text-amber-400"}`}>
                  {connectionMode === "sse" ? "SSE" : "POLL"}
                </span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-white/5">
                <span className="text-neutral-500">{language === "ru" ? "Состояние сети" : "Backend Status"}</span>
                <span className="text-neutral-300">ONLINE</span>
              </div>
            </div>

            <button
              onClick={() => fetchLiveSession()}
              disabled={isSyncing}
              className="w-full bg-neutral-900 hover:bg-neutral-800 text-neutral-200 border border-white/10 py-2.5 rounded-lg text-xs font-mono font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all cursor-pointer disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isSyncing ? "animate-spin" : ""}`} />
              {language === "ru" ? "Обновить сейчас" : "Refresh Logs"}
            </button>
          </div>

          {/* Session History Log */}
          <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-5 shadow-xl flex-1 flex flex-col min-h-[300px]">
            <h3 className="text-[11px] uppercase tracking-[0.2em] font-extrabold text-neutral-400 mb-4 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <History className="w-3.5 h-3.5 text-neutral-500" />
                {t.sidebar.session_history}
              </span>
              <span className="text-[10px] font-mono bg-black/50 px-2 py-0.5 rounded text-neutral-500 border border-white/5">
                {sessions.length} {t.sidebar.runs}
              </span>
            </h3>

            <LiveSessionBanner
              sessions={sessions}
              activeSessionId={activeSessionId}
              onSwitch={setActiveSessionId}
            />

            {/* Scroll list */}
            <div id="sessions-history-list" className="flex-1 overflow-y-auto space-y-2 max-h-[400px] md:max-h-[500px]">
              {[...sessions].sort((a, b) => {
                // Live sessions always at top, then by timestamp descending
                if (a.id.startsWith("live-") && !b.id.startsWith("live-")) return -1;
                if (!a.id.startsWith("live-") && b.id.startsWith("live-")) return 1;
                return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
              }).map((session) => {
                const isActive = session.id === activeSessionId;
                const isRenaming = renamingId === session.id;

                let stateColor = "bg-neutral-600";
                if (session.status === "completed") stateColor = "bg-[#4ade80]";
                else if (session.status === "error") stateColor = "bg-red-500";
                else if (session.status !== "idle") stateColor = "bg-amber-400 animate-pulse";

                return (
                  <div
                    key={session.id}
                    id={`session-history-item-${session.id}`}
                    onClick={() => {
                      setActiveSessionId(session.id);
                    }}
                    className={`p-3 rounded-lg border text-left cursor-pointer transition-all flex flex-col gap-1.5 group ${
                      isActive
                        ? "bg-[#4ade80]/5 border-[#4ade80] shadow-[0_0_12px_rgba(74,222,128,0.06)]"
                        : "bg-black/30 border-white/5 hover:bg-neutral-900/60 hover:border-white/10"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <span className={`w-1.5 h-1.5 rounded-full mt-1.5 ${stateColor}`} />

                      {/* Name or Rename block */}
                      {isRenaming ? (
                        <div className="flex-1 flex gap-1 items-center" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="text"
                            value={renameText}
                            onChange={(e) => setRenameText(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && saveRename(session.id)}
                            className="bg-black border border-white/10 rounded px-1.5 py-0.5 text-xs text-neutral-100 font-mono w-full focus:outline-none focus:border-[#4ade80]"
                          />
                          <button
                            onClick={() => saveRename(session.id)}
                            className="p-1 rounded bg-neutral-900 text-[#4ade80]"
                          >
                            <Check className="w-3 h-3" />
                          </button>
                        </div>
                      ) : (
                        <div className="flex-1 font-mono text-xs font-semibold text-neutral-300 truncate flex items-center gap-2">
                          {session.id.startsWith("live-")
                            ? (language === "ru" ? "◉ Машинный Агент Live" : "◉ Live Agent Monitor")
                            : session.name
                          }
                          {session.id.startsWith("live-") && (
                            <span className="text-[8px] font-extrabold uppercase tracking-widest text-[#4ade80] bg-[#4ade80]/10 border border-[#4ade80]/20 px-1.5 py-0.5 rounded leading-none">
                              LIVE
                            </span>
                          )}
                        </div>
                      )}

                      {/* Three-dot menu */}
                      {!isRenaming && (
                        <div className="relative">
                          <button
                            data-menu-root="true"
                            onClick={(e) => {
                              e.stopPropagation();
                              setMenuOpenId(menuOpenId === session.id ? null : session.id);
                            }}
                            className="p-0.5 rounded text-neutral-500 hover:text-white transition-colors"
                          >
                            <MoreVertical className="w-3.5 h-3.5" />
                          </button>

                          {menuOpenId === session.id && (
                            <div data-menu-root="true" className="absolute right-0 top-6 z-50 w-44 bg-neutral-900 border border-white/10 rounded-lg shadow-2xl py-1 overflow-hidden" onClick={(e) => e.stopPropagation()}>
                              <button
                                onClick={(e) => saveArchive(session, e)}
                                className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-neutral-300 hover:bg-white/5 hover:text-white transition-colors text-left"
                              >
                                <Download className="w-3.5 h-3.5 text-neutral-500" />
                                {language === "ru" ? "Сохранить архив" : "Save Archive"}
                              </button>
                              <button
                                id={`delete-session-${session.id}`}
                                onClick={(e) => deleteSession(session.id, e)}
                                className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 transition-colors text-left"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                                {language === "ru" ? "Удалить" : "Delete"}
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between text-[10px] text-neutral-500 font-mono uppercase">
                      <span>{session.id.startsWith("live-") ? "REAL-TIME" : new Date(session.timestamp).toLocaleDateString()}</span>
                      <span className="flex items-center gap-1.5">
                        {session.copyCount !== undefined && session.copyCount > 0 && (
                          <span
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[9px] font-semibold uppercase tracking-wider
                              bg-emerald-400/10 text-emerald-400/80 border border-emerald-400/20
                              hover:bg-emerald-400/20 hover:text-emerald-300 transition-colors cursor-default"
                            title={language === "ru" ? "Загружено как бэкап раз" : "Uploaded as backup this many times"}
                          >
                            <Copy className="w-2.5 h-2.5" />
                            <span>{session.copyCount} {language === "ru" ? "копия" : "copy"}</span>
                          </span>
                        )}
                        <span>{session.status}</span>
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Upload session backup JSON or .agent folder */}
            <div className="mt-4 pt-4 border-t border-white/5">
              <label
                htmlFor="backup-upload"
                className="w-full border border-dashed border-white/10 hover:border-[#4ade80]/50 bg-black/40 hover:bg-[#4ade80]/5 rounded-lg py-2.5 px-3 flex items-center justify-center gap-2 cursor-pointer transition-all group text-center"
              >
                <FolderOpen className="w-4 h-4 text-neutral-500 group-hover:text-[#4ade80]" />
                <span className="text-[10px] font-bold text-neutral-400 uppercase tracking-wider">{language === "ru" ? "Импортировать архив" : "Upload Backup"}</span>
              </label>
              <input
                type="file"
                id="backup-upload"
                multiple
                className="hidden"
                accept=".json"
                onChange={handleBackupUpload}
              />
            </div>
          </div>
        </aside>

        {/* Primary Dashboard Space */}
        <main className="flex-1 flex flex-col gap-6 min-w-0 relative z-10 order-1 lg:order-none">
          {/* Handle Live-Session Error / Setup Guidelines */}
          {activeSessionId.startsWith("live-") && liveError ? (
            <div className="bg-[#0c0c0e]/95 border border-amber-500/20 rounded-lg p-6 md:p-8 shadow-2xl flex flex-col justify-center items-center text-center space-y-6">
              <div className="w-12 h-12 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-500">
                <AlertCircle className="w-6 h-6" />
              </div>
              <div className="space-y-2 max-w-lg">
                <h2 className="text-base font-bold font-mono text-white uppercase tracking-wide">
                  {language === "ru" ? "Ожидание запуска JINX" : "Waiting for JINX Agent Session"}
                </h2>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  {(() => {
                    const isAuthError = liveError.toLowerCase().includes("unauthorized");
                    if (isAuthError) {
                      return language === "ru"
                        ? "Требуется корректный DASHBOARD_API_TOKEN. Обновите токен и повторите запрос."
                        : "A valid DASHBOARD_API_TOKEN is required. Update the token and retry.";
                    }
                    return language === "ru"
                      ? "Директория .agent не обнаружена. JINX создаёт её автоматически при первом запуске. Запустите агента через Cloud Code или OpenCode — дашборд сам подхватит состояние."
                      : "The .agent runtime directory has not been detected yet. JINX provisions it automatically on first launch. Start an agent session via Cloud Code or OpenCode — the dashboard will sync in real-time.";
                  })()}
                </p>
              </div>

              {/* Searched paths log */}
              {searchedPaths.length > 0 && (
                <div className="w-full max-w-xl bg-black/55 rounded border border-white/5 p-4 text-left font-mono text-[10px] text-neutral-500 space-y-1">
                  <div className="text-neutral-400 uppercase font-bold mb-1 tracking-wider">
                    {language === "ru" ? "Проверенные пути на сервере:" : "Searched Paths on Backend Server:"}
                  </div>
                  {searchedPaths.map((p, idx) => (
                    <div key={idx} className="truncate">
                      • <span className="text-[#4ade80]/60">{p}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Steps to launch agent */}
              <div className="w-full max-w-xl bg-neutral-950 border border-white/10 rounded-lg p-5 text-left font-mono text-xs space-y-3">
                <div className="flex items-center gap-2 text-white font-bold uppercase tracking-wider text-[11px] pb-2 border-b border-white/5">
                  <HelpCircle className="w-4 h-4 text-[#4ade80]" />
                  {language === "ru" ? "Запуск агента" : "Launching an Agent Session"}
                </div>
                <div className="space-y-2.5 text-neutral-300">
                  <p>
                    {language === "ru"
                      ? "JINX — это когнитивный рантайм, который управляется через одну из двух точек входа:"
                      : "JINX is a cognitive runtime orchestrated through one of two entry points:"}
                  </p>
                  <div className="bg-black/40 border border-white/5 rounded-lg p-3 space-y-3">
                    <div>
                      <span className="text-[#4ade80] font-bold text-[11px] uppercase tracking-wider">
                        {language === "ru" ? "Вариант A: Cloud Code" : "Option A: Cloud Code"}
                      </span>
                      <p className="text-neutral-400 mt-1 leading-relaxed">
                        {language === "ru"
                          ? "Откройте репозиторий в Cloud Code editor и отправьте задачу агенту через встроенный чат. JINX запустится автоматически."
                          : "Open the repository in Cloud Code editor and send a task to the agent via the built-in chat interface. JINX will start automatically."}
                      </p>
                    </div>
                    <div className="border-t border-white/5 pt-3">
                      <span className="text-[#4ade80] font-bold text-[11px] uppercase tracking-wider">
                        {language === "ru" ? "Вариант B: OpenCode CLI" : "Option B: OpenCode CLI"}
                      </span>
                      <p className="text-neutral-400 mt-1 leading-relaxed">
                        {language === "ru"
                          ? "Запустите OpenCode в терминале, укажите задачу — JINX выполнит её в подпроцессе:"
                          : "Run OpenCode in your terminal with a task description — JINX handles execution as a subprocess:"}
                      </p>
                      <pre className="bg-black p-2.5 rounded text-[10px] text-amber-400 font-semibold overflow-x-auto mt-2">
                        opencode
                      </pre>
                    </div>
                  </div>
                  <p className="text-neutral-500 text-[10px] leading-relaxed">
                    {language === "ru"
                      ? "После запуска JINX инициализирует директорию .agent с конфигурацией, состоянием и логами. Дашборд автоматически обнаружит её и отобразит ход выполнения в реальном времени."
                      : "Once started, JINX bootstraps the .agent directory with configuration, state, and logs. The dashboard automatically detects it and streams execution progress in real-time."}
                  </p>
                </div>
              </div>

              {/* Refresh buttons */}
              <div className="flex gap-4">
                <button
                  onClick={() => fetchLiveSession()}
                  className="bg-white hover:bg-neutral-200 text-black px-6 py-2 rounded-lg font-mono text-xs font-bold uppercase tracking-wider flex items-center gap-2 transition-all cursor-pointer"
                >
                  <RefreshCw className="w-4 h-4" />
                  {language === "ru" ? "Проверить снова" : "Check Folder Now"}
                </button>
                {sessions.length > 1 && (
                  <button
                    onClick={() => {
                      const backupSession = sessions.find(s => !s.id.startsWith("live-"));
                      if (backupSession) setActiveSessionId(backupSession.id);
                    }}
                    className="bg-transparent hover:bg-white/5 text-neutral-400 hover:text-white border border-white/10 px-4 py-2 rounded-lg font-mono text-xs transition-all cursor-pointer"
                  >
                    {language === "ru" ? "Посмотреть импортированные сессии" : "View Imported Backup Runs"}
                  </button>
                )}
              </div>
            </div>
          ) : (
            <>
              {/* Active Session Info Panel */}
              <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-5 shadow-xl relative overflow-hidden flex flex-col sm:flex-row justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-[#4ade80] bg-[#4ade80]/10 border border-[#4ade80]/20 px-2 py-0.5 rounded">
                      {currentSession.status === "perceive" ? t.phases.perceive :
                       currentSession.status === "analyze" ? t.phases.analyze :
                       currentSession.status === "plan" ? t.phases.plan :
                       currentSession.status === "execute" ? t.phases.execute :
                       currentSession.status === "verify" ? t.phases.verify :
                       currentSession.status === "commit" ? t.phases.commit :
                       currentSession.status === "completed" ? t.phases.completed :
                       currentSession.status === "error" ? t.phases.error :
                       currentSession.status === "idle" ? t.phases.idle :
                       currentSession.status}
                    </span>
                    <span className="text-[10px] font-mono text-neutral-500 uppercase">PID: {currentSession.stats.pid || "N/A"}</span>
                  </div>
                  <h2 className="text-sm md:text-base font-bold text-white font-mono tracking-tight truncate max-w-lg">
                  {currentSession.id.startsWith("live-")
                    ? (language === "ru" ? "Сеанс JINX Agent" : "JINX Agent Session")
                    : currentSession.name
                  }
                  </h2>
                  <p className="text-[10px] text-neutral-500 font-mono mt-1 uppercase">
                    {t.session_info.launched_at}{new Date(currentSession.timestamp).toLocaleString()}
                  </p>
                </div>

                {/* Quick Metrics Summary */}
                <div className="flex gap-4 border-t sm:border-t-0 border-white/10 pt-3 sm:pt-0 font-mono">
                  <div className="border-r border-white/10 pr-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">{t.session_info.duration}</div>
                    <div className="text-sm md:text-base font-bold text-neutral-200 mt-0.5">{currentSession.elapsedTime}s</div>
                  </div>
                  <div className="border-r border-white/10 pr-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">
                      {language === "ru" ? "СИНХРОНИЗАЦИЯ" : "SYNCED TIME"}
                    </div>
                    <div className="text-sm md:text-base font-bold text-[#4ade80] mt-0.5">
                      {currentSession.id.startsWith("live-") ? (lastSyncedAt || "LIVE") : "BACKUP"}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">{t.session_info.workspace_files}</div>
                    <div className="text-sm md:text-base font-bold text-neutral-200 mt-0.5">{Object.keys(currentSession.files).length}</div>
                  </div>
                </div>
              </div>

              {/* Cognitive Loop Stage Visualization */}
              <CognitiveLoop currentStatus={currentSession.status} />

              {/* Workspace Tabs Navigation */}
              <div className="flex overflow-x-auto whitespace-nowrap flex-nowrap scrollbar-none border border-white/10 bg-neutral-950/60 p-1 rounded-lg gap-1 md:overflow-visible md:whitespace-normal md:grid md:grid-cols-5 md:gap-0">
                <button
                  id="tab-summary-btn"
                  onClick={() => setActiveTab("summary")}
                  className={`flex-shrink-0 min-w-[100px] md:min-w-0 flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded text-[11px] md:text-xs font-mono font-bold uppercase tracking-wider md:tracking-widest transition-all cursor-pointer ${
                    activeTab === "summary"
                      ? "bg-white text-black font-extrabold"
                      : "text-neutral-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <LayoutDashboard className="w-3.5 h-3.5" />
                  {t.tabs.summary}
                </button>
                <button
                  id="tab-thoughts-btn"
                  onClick={() => setActiveTab("thoughts")}
                  className={`flex-shrink-0 min-w-[100px] md:min-w-0 flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded text-[11px] md:text-xs font-mono font-bold uppercase tracking-wider md:tracking-widest transition-all cursor-pointer ${
                    activeTab === "thoughts"
                      ? "bg-white text-black font-extrabold"
                      : "text-neutral-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <Activity className="w-3.5 h-3.5" />
                  {t.tabs.thoughts}
                </button>
                <button
                  id="tab-files-btn"
                  onClick={() => setActiveTab("files")}
                  className={`flex-shrink-0 min-w-[100px] md:min-w-0 flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded text-[11px] md:text-xs font-mono font-bold uppercase tracking-wider md:tracking-widest transition-all cursor-pointer ${
                    activeTab === "files"
                      ? "bg-white text-black font-extrabold"
                      : "text-neutral-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <FileCode className="w-3.5 h-3.5" />
                  {t.tabs.files}
                </button>
                <button
                  id="tab-console-btn"
                  onClick={() => setActiveTab("console")}
                  className={`flex-shrink-0 min-w-[100px] md:min-w-0 flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded text-[11px] md:text-xs font-mono font-bold uppercase tracking-wider md:tracking-widest transition-all cursor-pointer ${
                    activeTab === "console"
                      ? "bg-white text-black font-extrabold"
                      : "text-neutral-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <Terminal className="w-3.5 h-3.5" />
                  {t.tabs.console}
                </button>
                <button
                  id="tab-diffs-btn"
                  onClick={() => setActiveTab("diffs")}
                  className={`flex-shrink-0 min-w-[100px] md:min-w-0 flex-1 flex items-center justify-center gap-1.5 py-2.5 rounded text-[11px] md:text-xs font-mono font-bold uppercase tracking-wider md:tracking-widest transition-all cursor-pointer ${
                    activeTab === "diffs"
                      ? "bg-white text-black font-extrabold"
                      : "text-neutral-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  <GitPullRequest className="w-3.5 h-3.5" />
                  {t.tabs.diffs}
                </button>
              </div>

              {/* Active Tab View Panels with Fade transition */}
              <div className="flex-1 min-h-[500px]">
                <AnimatePresence mode="wait">
                  {activeTab === "summary" && (
                    <motion.div
                      key="summary"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <RunSummary session={currentSession} />
                    </motion.div>
                  )}
                  {activeTab === "thoughts" && (
                    <motion.div
                      key="thoughts"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <ThoughtStream thoughts={currentSession.thoughts} />
                    </motion.div>
                  )}
                  {activeTab === "files" && (
                    <motion.div
                      key="files"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <FileExplorer files={currentSession.files} />
                    </motion.div>
                  )}
                  {activeTab === "console" && (
                    <motion.div
                      key="console"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <TerminalConsole terminalLog={currentSession.terminalLog} rpcLog={currentSession.rpcLog} />
                    </motion.div>
                  )}
                  {activeTab === "diffs" && (
                    <motion.div
                      key="diffs"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.2 }}
                    >
                      <DiffViewer diffs={currentSession.diffs} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </>
          )}
        </main>
      </div>

      {/* High-tech pixel status footer */}
      <footer className="border-t border-neutral-900 bg-neutral-950/80 px-6 py-4 flex flex-col sm:flex-row items-center justify-between text-[11px] font-mono text-neutral-500 mt-12">
        <div className="flex items-center gap-2">
          <span>WORKSPACE MONITOR v1.2.3</span>
          <span className="text-neutral-800">|</span>
          <span>COMPATIBLE WITH JINX RUNTIME SPEC 1.0.0</span>
        </div>
        <div className="mt-2 sm:mt-0">
          {language === "ru" ? "ЛОКАЛЬНЫЙ МОНИТОРИНГ ДАННЫХ .AGENT" : "LOCAL .AGENT REPOSITORY MONITOR"}
        </div>
      </footer>
    </div>
  );
}

function LiveSessionBanner({
  sessions,
  activeSessionId,
  onSwitch,
}: {
  sessions: AgentSession[];
  activeSessionId: string;
  onSwitch: (id: string) => void;
}) {
  const { t } = useLanguage();
  const liveSession = sessions.find(s => s.id.startsWith("live-") && s.status !== "idle");
  if (!liveSession || activeSessionId.startsWith("live-")) return null;
  return (
    <button
      onClick={() => onSwitch(liveSession.id)}
      className="w-full mb-3 px-3 py-2 rounded border border-[#4ade80]/20 bg-[#4ade80]/5 text-left flex items-center justify-between gap-2 transition-all hover:bg-[#4ade80]/10 cursor-pointer group"
    >
      <span className="text-[10px] font-mono font-bold text-[#4ade80] flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-[#4ade80] animate-pulse" />
        {t.sidebar.newer_live_available}
      </span>
      <span className="text-[9px] font-mono font-bold uppercase tracking-wider text-[#4ade80]/60 group-hover:text-[#4ade80] transition-colors">
        {t.sidebar.switch_to_live} →
      </span>
    </button>
  );
}
