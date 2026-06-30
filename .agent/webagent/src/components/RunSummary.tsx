/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgentSession } from "../types";
import { useLanguage } from "../context/LanguageContext";
import {
  CheckCircle2,
  Clock,
  Cpu,
  Monitor,
  AlertCircle,
  Loader2,
  Circle,
  Terminal,
} from "lucide-react";

interface RunSummaryProps {
  session: AgentSession;
}

export default function RunSummary({ session }: RunSummaryProps) {
  const { language, t } = useLanguage();

  const getStepStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="w-5 h-5 text-[#4ade80] flex-shrink-0" />;
      case "running":
        return <Loader2 className="w-5 h-5 text-amber-400 animate-spin flex-shrink-0" />;
      case "failed":
        return <AlertCircle className="w-5 h-5 text-rose-500 flex-shrink-0" />;
      default:
        return <Circle className="w-5 h-5 text-neutral-700 flex-shrink-0" />;
    }
  };

  const getStepStatusClass = (status: string) => {
    switch (status) {
      case "completed":
        return "border-[#4ade80]/20 bg-[#4ade80]/5 text-neutral-200";
      case "running":
        return "border-amber-500/20 bg-amber-500/5 text-amber-200 shadow-[0_0_15px_rgba(245,158,11,0.05)]";
      case "failed":
        return "border-rose-950/20 bg-rose-950/5 text-rose-200";
      default:
        return "border-white/5 bg-black/40 text-neutral-400";
    }
  };

  return (
    <div id="run-summary-container" className="space-y-6">
      {/* Step Checklist Card */}
      <div id="checklist-card" className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-6 shadow-xl">
        <h3 className="text-[11px] font-extrabold uppercase tracking-[0.2em] text-neutral-400 mb-4 flex items-center gap-2">
          <span className="w-1.5 h-3 bg-[#4ade80] rounded-full"></span>
          {t.run_summary.multi_step_plan}
        </h3>

        <div id="plan-steps-list" className="space-y-3">
          {session.plan.length === 0 ? (
            <div className="text-center py-12 text-neutral-600 font-mono text-xs uppercase">
              [ {t.run_summary.no_plan}. {language === "ru" ? "Агент находится в фазе" : "Agent is in"} {session.status.toUpperCase()} ]
            </div>
          ) : (
            session.plan.map((step) => (
              <div
                key={step.id}
                id={`plan-step-item-${step.id}`}
                className={`border rounded p-4 transition-all flex items-start gap-4 ${getStepStatusClass(
                  step.status
                )}`}
              >
                <div className="mt-0.5">{getStepStatusIcon(step.status)}</div>
                <div className="flex-1">
                  <h4 className="text-xs font-bold font-mono uppercase tracking-wide">{step.title}</h4>
                  <p className="text-xs text-neutral-500 mt-1 leading-relaxed">{step.description}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* JINX Native State Overview */}
      {((session.facts && session.facts.length > 0) || (session.open && session.open.length > 0) || (session.debt && session.debt.length > 0)) && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Scope Facts */}
          {session.facts && session.facts.length > 0 && (
            <div className="bg-[#0c0c0e]/95 border border-emerald-500/10 rounded-lg p-5 shadow-xl flex flex-col">
              <h4 className="text-[10px] font-extrabold uppercase tracking-[0.2em] text-emerald-400 mb-3 flex items-center gap-1.5 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                {language === "ru" ? "Выявленные Факты" : "Scope Facts"}
              </h4>
              <ul className="space-y-2 flex-1 text-xs text-neutral-300 leading-normal font-mono list-inside list-disc">
                {session.facts.map((fact, idx) => (
                  <li key={idx} className="marker:text-emerald-500 pl-1">{fact}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Open Requirements */}
          {session.open && session.open.length > 0 && (
            <div className="bg-[#0c0c0e]/95 border border-amber-500/10 rounded-lg p-5 shadow-xl flex flex-col">
              <h4 className="text-[10px] font-extrabold uppercase tracking-[0.2em] text-amber-400 mb-3 flex items-center gap-1.5 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500"></span>
                {language === "ru" ? "Открытые Задачи" : "Open Requirements"}
              </h4>
              <ul className="space-y-2 flex-1 text-xs text-neutral-300 leading-normal font-mono list-inside list-disc">
                {session.open.map((req, idx) => (
                  <li key={idx} className="marker:text-amber-500 pl-1">{req}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Design Debt */}
          {session.debt && session.debt.length > 0 && (
            <div className="bg-[#0c0c0e]/95 border border-rose-500/10 rounded-lg p-5 shadow-xl flex flex-col">
              <h4 className="text-[10px] font-extrabold uppercase tracking-[0.2em] text-rose-400 mb-3 flex items-center gap-1.5 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse"></span>
                {language === "ru" ? "Технический Долг" : "Unresolved Debt"}
              </h4>
              <ul className="space-y-2 flex-1 text-xs text-neutral-300 leading-normal font-mono list-inside list-disc">
                {session.debt.map((debtItem, idx) => (
                  <li key={idx} className="marker:text-rose-500 pl-1">{debtItem}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Technical Stats Card */}
      <div id="tech-stats-grid" className="grid grid-cols-2 md:grid-cols-4 gap-4 font-mono">
        <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-1">
            <Clock className="w-4 h-4" />
            <span className="text-[9px] uppercase tracking-wider font-bold">{t.run_summary.elapsed_time}</span>
          </div>
          <div className="text-lg font-bold text-neutral-100">{session.elapsedTime}s</div>
        </div>

        <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-1">
            <Cpu className="w-4 h-4" />
            <span className="text-[9px] uppercase tracking-wider font-bold">{t.run_summary.subprocess_pid}</span>
          </div>
          <div className="text-lg font-bold text-neutral-100">{session.stats.pid || "N/A"}</div>
        </div>

        <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-1">
            <Terminal className="w-4 h-4" />
            <span className="text-[9px] uppercase tracking-wider font-bold">{language === "ru" ? "ПЛАТФОРМА" : "PLATFORM OS"}</span>
          </div>
          <div className="text-sm font-bold text-[#4ade80] uppercase truncate mt-0.5">
            {session.stats.os || "local"}
          </div>
        </div>

        <div className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-4">
          <div className="flex items-center gap-2 text-neutral-500 mb-1">
            <Monitor className="w-4 h-4" />
            <span className="text-[9px] uppercase tracking-wider font-bold">{t.run_summary.host_node}</span>
          </div>
          <div className="text-xs font-bold text-neutral-400 truncate mt-0.5">{session.stats.hostname || "localhost"}</div>
        </div>
      </div>
    </div>
  );
}
