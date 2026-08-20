/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { RPCMessage } from "../types";
import { ArrowUpRight, ArrowDownLeft, Terminal, ShieldAlert } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

interface TerminalConsoleProps {
  terminalLog: string[];
  rpcLog: RPCMessage[];
}

export default function TerminalConsole({ terminalLog, rpcLog }: TerminalConsoleProps) {
  const { t } = useLanguage();
  const [viewMode, setViewMode] = useState<"terminal" | "rpc">("terminal");

  return (
    <div id="terminal-console-card" className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg shadow-2xl flex flex-col h-[500px] overflow-hidden">
      {/* Console Tab Header */}
      <div className="bg-neutral-950/80 border-b border-white/10 p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-[#4ade80]" />
          <span className="text-xs font-mono font-bold uppercase tracking-wider text-neutral-200">
            {viewMode === "terminal" ? t.terminal.terminal_io : t.terminal.rpc_ipc}
          </span>
        </div>

        <div className="flex gap-1.5 bg-neutral-900 p-1 rounded border border-white/5">
          <button
            id="toggle-terminal-log-btn"
            onClick={() => setViewMode("terminal")}
            className={`px-3 py-1 rounded text-[10px] font-mono uppercase font-bold tracking-widest transition-colors cursor-pointer ${
              viewMode === "terminal" ? "bg-white text-black font-extrabold" : "text-neutral-400 hover:text-white"
            }`}
          >
            {t.terminal.terminal_tab}
          </button>
          <button
            id="toggle-rpc-log-btn"
            onClick={() => setViewMode("rpc")}
            className={`px-3 py-1 rounded text-[10px] font-mono uppercase font-bold tracking-widest transition-colors cursor-pointer ${
              viewMode === "rpc" ? "bg-white text-black font-extrabold" : "text-neutral-400 hover:text-white"
            }`}
          >
            {t.terminal.rpc_tab}
          </button>
        </div>
      </div>

      {/* View Console Logs */}
      {viewMode === "terminal" ? (
        <div id="terminal-text-area" className="flex-1 overflow-y-auto p-6 font-mono text-xs text-neutral-300 space-y-2 leading-relaxed selection:bg-[#4ade80]/30 bg-black">
          {terminalLog.length === 0 ? (
            <div className="text-neutral-600 text-center py-20 uppercase tracking-widest font-mono animate-pulse">
              {t.terminal.no_terminal}
            </div>
          ) : (
            terminalLog.map((line, idx) => {
              const isCommand = line.startsWith("$");
              const isError = line.toLowerCase().includes("error") || line.toLowerCase().includes("failed");
              const isSpread = line.toLowerCase().includes("pass") || line.toLowerCase().includes("success");

              let textStyle = "text-neutral-300";
              if (isCommand) textStyle = "text-[#4ade80] font-bold border-l-2 border-[#4ade80]/50 pl-2 bg-[#4ade80]/5 py-0.5";
              else if (isError) textStyle = "text-rose-400 bg-rose-950/10 border-l-2 border-rose-500 pl-2";
              else if (isSpread) textStyle = "text-[#4ade80] bg-[#4ade80]/10 border-l-2 border-[#4ade80] pl-2";

              return (
                <div key={idx} className={textStyle}>
                  {line}
                </div>
              );
            })
          )}
        </div>
      ) : (
        /* View JSON-RPC Logs */
        <div id="rpc-logs-list" className="flex-1 overflow-y-auto p-4 space-y-2.5 bg-black">
          {rpcLog.length === 0 ? (
            <div className="text-neutral-600 text-center py-20 font-mono text-xs uppercase tracking-widest animate-pulse">
              {t.terminal.no_ipc}
            </div>
          ) : (
            rpcLog.map((msg) => {
              const isSent = msg.direction === "sent";
              const hasError = !!msg.error;
              const formattedTime = new Date(msg.timestamp).toLocaleTimeString();

              return (
                <div
                  key={msg.id}
                  id={`rpc-log-item-${msg.id}`}
                  className={`border rounded p-3 bg-[#0c0c0e]/60 relative font-mono text-xs transition-all ${
                    hasError ? "border-red-900/40 hover:border-red-900" : "border-white/5 hover:border-white/10"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-1.5 py-0.5 rounded text-[9px] uppercase font-bold flex items-center gap-1 ${
                          isSent
                            ? "bg-blue-950/40 text-blue-400 border border-blue-800/25"
                            : "bg-[#4ade80]/10 text-[#4ade80] border border-[#4ade80]/20"
                        }`}
                      >
                        {isSent ? (
                          <>
                            <ArrowUpRight className="w-3 h-3" /> {t.terminal.call_sent}
                          </>
                        ) : (
                          <>
                            <ArrowDownLeft className="w-3 h-3" /> {t.terminal.reply_rcvd}
                          </>
                        )}
                      </span>
                      <span className="font-bold text-neutral-300">method: {msg.method}</span>
                    </div>
                    <span className="text-[10px] text-neutral-500">{formattedTime}</span>
                  </div>

                  {msg.params && (
                    <div className="mb-1.5 pl-2 border-l border-white/10">
                      <div className="text-[10px] text-neutral-500 uppercase font-bold tracking-wider mb-0.5">{t.terminal.parameters}</div>
                      <pre className="text-[11px] text-[#4ade80]/80 overflow-x-auto whitespace-pre">
                        {JSON.stringify(msg.params, null, 2)}
                      </pre>
                    </div>
                  )}

                  {msg.result && (
                    <div className="pl-2 border-l border-[#4ade80]/30">
                      <div className="text-[10px] text-neutral-500 uppercase font-bold tracking-wider mb-0.5">{t.terminal.response_result}</div>
                      <pre className="text-[11px] text-neutral-300 overflow-x-auto whitespace-pre">
                        {JSON.stringify(msg.result, null, 2)}
                      </pre>
                    </div>
                  )}

                  {msg.error && (
                    <div className="pl-2 border-l border-red-500 bg-red-950/10 p-1.5 rounded">
                      <div className="text-[10px] text-red-400 uppercase font-bold tracking-wider mb-0.5 flex items-center gap-1">
                        <ShieldAlert className="w-3 h-3" /> {t.terminal.response_error}
                      </div>
                      <pre className="text-[11px] text-rose-400 overflow-x-auto whitespace-pre">
                        {JSON.stringify(msg.error, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}
