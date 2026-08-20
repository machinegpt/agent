/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { motion } from "motion/react";
import { SessionStatus } from "../types";
import { Eye, Search, ClipboardList, Play, ShieldCheck, CheckCircle, Database } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

interface CognitiveLoopProps {
  currentStatus: SessionStatus;
}

interface PhaseDetail {
  status: SessionStatus;
  label: string;
  desc: string;
  color: string;
  icon: any;
}

const statusIndexMap: Record<SessionStatus, number> = {
  idle: -1,
  perceive: 0,
  analyze: 1,
  plan: 2,
  execute: 3,
  verify: 4,
  commit: 5,
  completed: 6,
  error: 6,
};

function getStatusIndex(status: SessionStatus) {
  return statusIndexMap[status] ?? -1;
}

export default function CognitiveLoop({ currentStatus }: CognitiveLoopProps) {
  const { language, t } = useLanguage();

  const phases: PhaseDetail[] = [
    {
      status: "perceive",
      label: t.cognitive_loop_phases.perceive.label,
      desc: t.cognitive_loop_phases.perceive.desc,
      color: "from-cyan-500 to-blue-600",
      icon: Eye,
    },
    {
      status: "analyze",
      label: t.cognitive_loop_phases.analyze.label,
      desc: t.cognitive_loop_phases.analyze.desc,
      color: "from-purple-500 to-indigo-600",
      icon: Search,
    },
    {
      status: "plan",
      label: t.cognitive_loop_phases.plan.label,
      desc: t.cognitive_loop_phases.plan.desc,
      color: "from-amber-500 to-orange-600",
      icon: ClipboardList,
    },
    {
      status: "execute",
      label: t.cognitive_loop_phases.execute.label,
      desc: t.cognitive_loop_phases.execute.desc,
      color: "from-rose-500 to-pink-600",
      icon: Play,
    },
    {
      status: "verify",
      label: t.cognitive_loop_phases.verify.label,
      desc: t.cognitive_loop_phases.verify.desc,
      color: "from-emerald-500 to-teal-600",
      icon: ShieldCheck,
    },
    {
      status: "commit",
      label: t.cognitive_loop_phases.commit.label,
      desc: t.cognitive_loop_phases.commit.desc,
      color: "from-indigo-500 to-blue-600",
      icon: Database,
    },
    {
      status: "completed",
      label: t.cognitive_loop_phases.completed.label,
      desc: t.cognitive_loop_phases.completed.desc,
      color: "from-emerald-400 to-green-500",
      icon: CheckCircle,
    },
  ];

  const currentIndex = getStatusIndex(currentStatus);
  const isErrorState = currentStatus === "error";

  return (
    <div id="cognitive-loop-container" className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-6 shadow-xl relative overflow-hidden">
      {/* Decorative cyber grid background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:1.5rem_1.5rem] opacity-30" />
      
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-[11px] uppercase tracking-[0.2em] font-extrabold text-neutral-400">{t.cognitive_loop.title}</h3>
            <p className="text-xs text-neutral-500 font-mono uppercase mt-0.5">{t.cognitive_loop.status}: <span className={`font-bold ${
              currentStatus === "completed" ? "text-[#4ade80]" :
              currentStatus === "error" ? "text-red-400" :
              "text-[#4ade80]"
            }`}>
              {currentStatus === "perceive" ? t.phases.perceive :
               currentStatus === "analyze" ? t.phases.analyze :
               currentStatus === "plan" ? t.phases.plan :
               currentStatus === "execute" ? t.phases.execute :
               currentStatus === "verify" ? t.phases.verify :
               currentStatus === "commit" ? t.phases.commit :
               currentStatus === "completed" ? t.phases.completed :
               currentStatus === "error" ? t.phases.error :
               currentStatus === "idle" ? t.phases.idle :
               currentStatus}
            </span></p>
          </div>
          {currentStatus === "error" && (
            <div className="bg-red-500/10 border border-red-500/30 text-red-400 text-xs px-3 py-1 rounded font-mono animate-pulse">
              LOOP EXCEPTION DETECTED
            </div>
          )}
        </div>

        {/* Desktop View */}
        <div className="hidden lg:grid grid-cols-7 gap-3 relative items-stretch">
          {currentStatus === "idle" && (
            <div className="col-span-7 bg-neutral-950/40 border border-white/5 rounded-xl p-4 flex gap-3.5 items-center">
              <div className="p-2 bg-neutral-900 text-neutral-500 rounded-lg">
                <Database className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <span className="text-[9px] font-mono uppercase tracking-wider text-[#4ade80]/60 font-extrabold">SYSTEM_STATUS // IDLE</span>
                <h4 className="text-xs font-mono font-bold text-white uppercase mt-0.5">
                  {language === "ru" ? "Ожидание запуска" : "Waiting for JINX Agent"}
                </h4>
                <p className="text-[11px] text-neutral-400 mt-1 leading-relaxed">
                  {language === "ru"
                    ? "Запустите локального Python-агента JINX на вашем компьютере для начала трансляции и обработки когнитивных циклов."
                    : "Initialize the local JINX Python agent to begin streaming telemetry and cognitive decision loops."}
                </p>
              </div>
            </div>
          )}
          {phases.map((phase, idx) => {
            const Icon = phase.icon;
            const isCompleted = isErrorState ? idx < phases.length - 1 : currentIndex >= idx;
            const isActive = !isErrorState && currentIndex === idx && currentStatus !== "completed";
            const isError = isErrorState && idx === (phases.length - 1);
            const isAllDone = currentStatus === "completed" && idx === (phases.length - 1);

            let borderClass = "border-white/5 bg-black/40";
            let glowClass = "";

            if (isActive) {
              borderClass = "border-[#4ade80] bg-[#4ade80]/10";
              glowClass = "shadow-[0_0_15px_rgba(74,222,128,0.15)]";
            } else if (isAllDone) {
              borderClass = "border-[#4ade80]/60 bg-[#4ade80]/15";
              glowClass = "shadow-[0_0_20px_rgba(74,222,128,0.25)]";
            } else if (isCompleted) {
              borderClass = "border-[#4ade80]/30 bg-[#4ade80]/5";
            } else if (isError) {
              borderClass = "border-red-500 bg-red-950/20";
              glowClass = "shadow-[0_0_15px_rgba(239,68,68,0.2)]";
            }

            return (
              <motion.div
                key={phase.status}
                id={`loop-phase-${phase.status}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: idx * 0.05 }}
                className={`border rounded-lg p-3 flex flex-col items-center text-center relative transition-all duration-300 h-full ${borderClass} ${glowClass}`}
              >
                {/* Active pulse dot */}
                {isActive && (
                  <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#4ade80] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-[#4ade80]"></span>
                  </span>
                )}
                {isError && (
                  <span className="absolute -top-1 -right-1 flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                  </span>
                )}

                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 bg-gradient-to-br ${
                    isActive || isCompleted ? "from-[#4ade80]/20 to-[#22c55e]/20" : "from-neutral-800 to-neutral-900"
                  } ${isActive || isCompleted ? "text-[#4ade80]" : "text-neutral-500"}`}
                >
                  <Icon className="w-5 h-5" />
                </div>

                <div className="font-mono text-xs font-semibold mb-1 truncate w-full uppercase">{phase.label}</div>
                <div className="text-[10px] text-neutral-500 leading-snug hidden lg:block min-h-[5.5rem]">
                  {phase.desc}
                </div>

                {/* Connecting arrow for desktop */}
                {idx < 6 && (
                  <div className="absolute -right-2 top-1/2 -translate-y-1/2 translate-x-1/2 z-20">
                    <svg className="w-4 h-4 text-neutral-800" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>

        {/* Mobile View: High-fidelity Vertical Timeline */}
        <div className="block lg:hidden space-y-4">
          {currentStatus === "idle" && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-neutral-950/40 border border-white/5 rounded-xl p-4 flex gap-3.5 items-start"
            >
              <div className="p-2 bg-neutral-900 text-neutral-500 rounded-lg">
                <Database className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <span className="text-[9px] font-mono uppercase tracking-wider text-[#4ade80]/60 font-extrabold">SYSTEM_STATUS // IDLE</span>
                <h4 className="text-xs font-mono font-bold text-white uppercase mt-0.5">
                  {language === "ru" ? "Ожидание запуска" : "Waiting for JINX Agent"}
                </h4>
                <p className="text-[11px] text-neutral-400 mt-1 leading-relaxed">
                  {language === "ru" 
                    ? "Запустите локального Python-агента JINX на вашем компьютере для начала трансляции и обработки когнитивных циклов."
                    : "Initialize the local JINX Python agent to begin streaming telemetry and cognitive decision loops."}
                </p>
              </div>
            </motion.div>
          )}

          <div className="relative border-l border-white/10 ml-4 pl-5 space-y-4">
            {phases.map((phase, idx) => {
              const Icon = phase.icon;
              const isCompleted = isErrorState ? idx < phases.length - 1 : currentIndex >= idx;
              const isActive = !isErrorState && currentIndex === idx && currentStatus !== "completed";
              const isError = isErrorState && idx === (phases.length - 1);
              const isIdle = currentStatus === "idle";
              const isAllDone = currentStatus === "completed" && idx === (phases.length - 1);

              let bgClass = "bg-neutral-900 border-neutral-800 text-neutral-500";
              let textClass = "text-neutral-500";
              let titleClass = "text-neutral-400";
              let borderClass = "border-white/5 bg-black/20";
              let glowClass = "";

              if (isActive) {
                bgClass = "bg-[#4ade80] border-[#4ade80] text-black ring-4 ring-[#4ade80]/25 scale-105 z-10";
                titleClass = "text-[#4ade80] font-bold";
                textClass = "text-neutral-200";
                borderClass = "border-[#4ade80]/30 bg-[#4ade80]/10";
                glowClass = "shadow-[0_0_15px_rgba(74,222,128,0.1)]";
              } else if (isAllDone) {
                bgClass = "bg-[#4ade80]/30 border-[#4ade80] text-[#4ade80] ring-2 ring-[#4ade80]/30";
                titleClass = "text-[#4ade80] font-bold";
                textClass = "text-neutral-200";
                borderClass = "border-[#4ade80]/60 bg-[#4ade80]/15";
                glowClass = "shadow-[0_0_20px_rgba(74,222,128,0.25)]";
              } else if (isCompleted) {
                bgClass = "bg-[#4ade80]/20 border-[#4ade80]/40 text-[#4ade80]";
                titleClass = "text-white font-semibold";
                textClass = "text-neutral-300";
                borderClass = "border-[#4ade80]/10 bg-[#4ade80]/5";
              } else if (isError) {
                bgClass = "bg-red-500 border-red-500 text-white scale-105 z-10";
                titleClass = "text-red-400 font-bold";
                textClass = "text-neutral-200";
                borderClass = "border-red-500/30 bg-red-950/15";
                glowClass = "shadow-[0_0_15px_rgba(239,68,68,0.15)]";
              }

              return (
                <motion.div
                  key={phase.status}
                  initial={{ opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  className={`relative flex flex-col gap-1.5 p-3.5 rounded-xl border transition-all duration-300 ${borderClass} ${glowClass}`}
                >
                  {/* Timeline Node Icon */}
                  <div className="absolute -left-[38px] top-1/2 -translate-y-1/2 flex items-center justify-center">
                    <span className={`w-8 h-8 rounded-full border flex items-center justify-center transition-all duration-300 ${bgClass}`}>
                      <Icon className="w-4 h-4" />
                    </span>
                    {isActive && (
                      <span className="absolute -inset-1 rounded-full border border-[#4ade80] animate-ping opacity-30" />
                    )}
                    {isError && (
                      <span className="absolute -inset-1 rounded-full border border-red-500 animate-ping opacity-30" />
                    )}
                  </div>

                  {/* Header / State Row */}
                  <div className="flex items-center justify-between">
                    <span className={`text-[10px] font-mono tracking-wider font-extrabold uppercase ${titleClass}`}>
                      STEP 0{idx + 1} // {phase.label}
                    </span>
                    <span className={`text-[9px] font-mono font-bold uppercase px-2 py-0.5 rounded ${
                      isError 
                        ? "bg-red-500/15 text-red-400" 
                        : isActive 
                        ? "bg-[#4ade80]/20 text-[#4ade80]" 
                        : isCompleted 
                        ? "bg-[#4ade80]/10 text-[#4ade80]/80"
                        : "bg-neutral-800 text-neutral-500"
                    }`}>
                      {isError ? "ERROR" : isActive ? "ACTIVE" : isCompleted ? "DONE" : isIdle ? "IDLE" : "PENDING"}
                    </span>
                  </div>

                  {/* Step Description */}
                  <p className={`text-[11px] leading-relaxed font-sans ${textClass}`}>
                    {phase.desc}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
