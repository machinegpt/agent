/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useMemo } from "react";
import { motion } from "motion/react";
import { ThoughtLog } from "../types";
import { MessageSquare, HelpCircle, AlertTriangle, CheckSquare, Settings, Search } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

interface ThoughtStreamProps {
  thoughts: ThoughtLog[];
}

export default function ThoughtStream({ thoughts }: ThoughtStreamProps) {
  const { t } = useLanguage();
  const [searchTerm, setSearchTerm] = useState(() => {
    return localStorage.getItem("jinx_thought_search_term") || "";
  });
  const [selectedCategory, setSelectedCategory] = useState<string>(() => {
    return localStorage.getItem("jinx_thought_selected_category") || "all";
  });
  const [selectedPhase, setSelectedPhase] = useState<string>(() => {
    return localStorage.getItem("jinx_thought_selected_phase") || "all";
  });

  useEffect(() => {
    localStorage.setItem("jinx_thought_search_term", searchTerm);
  }, [searchTerm]);

  useEffect(() => {
    localStorage.setItem("jinx_thought_selected_category", selectedCategory);
  }, [selectedCategory]);

  useEffect(() => {
    localStorage.setItem("jinx_thought_selected_phase", selectedPhase);
  }, [selectedPhase]);

  const categories = [
    { value: "all", label: t.thought_stream.category_all },
    { value: "monologue", label: t.categories.monologue },
    { value: "question", label: t.categories.question },
    { value: "decision", label: t.categories.decision },
    { value: "check", label: t.categories.check },
    { value: "system", label: t.categories.system },
  ];

  const phases = [
    { value: "all", label: t.thought_stream.phase_all },
    { value: "perceive", label: t.phases.perceive },
    { value: "analyze", label: t.phases.analyze },
    { value: "plan", label: t.phases.plan },
    { value: "execute", label: t.phases.execute },
    { value: "verify", label: t.phases.verify },
    { value: "commit", label: t.phases.commit },
    { value: "completed", label: t.phases.completed },
  ];

  const getCategoryIcon = (category: ThoughtLog["category"]) => {
    switch (category) {
      case "monologue": return MessageSquare;
      case "question": return HelpCircle;
      case "decision": return Settings;
      case "check": return AlertTriangle;
      case "system": return CheckSquare;
      default: return MessageSquare;
    }
  };

  const getCategoryColor = (category: ThoughtLog["category"]) => {
    switch (category) {
      case "monologue": return "border-white/10 text-neutral-300 bg-white/[0.02]";
      case "question": return "border-amber-500/20 text-amber-400 bg-amber-500/[0.02]";
      case "decision": return "border-purple-500/20 text-purple-400 bg-purple-500/[0.02]";
      case "check": return "border-red-500/20 text-red-400 bg-red-500/[0.02]";
      case "system": return "border-[#4ade80]/20 text-[#4ade80] bg-[#4ade80]/[0.02]";
      default: return "border-white/10 text-neutral-400 bg-black/40";
    }
  };

  const filteredThoughts = useMemo(() => thoughts.filter((t) => {
    const matchesSearch = t.text.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          t.phase.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === "all" || t.category === selectedCategory;
    const matchesPhase = selectedPhase === "all" || t.phase === selectedPhase;
    return matchesSearch && matchesCategory && matchesPhase;
  }), [thoughts, searchTerm, selectedCategory, selectedPhase]);

  return (
    <div id="thought-stream-card" className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-6 shadow-xl flex flex-col h-full">
      {/* Search & Filter Headers */}
      <div className="flex flex-col md:flex-row gap-3 mb-6 items-center justify-between border-b border-white/10 pb-4">
        <div>
          <h3 className="text-xs md:text-sm font-extrabold uppercase tracking-widest text-white flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#4ade80] animate-pulse"></span>
            {t.thought_stream.title}
          </h3>
          <p className="text-[10px] text-neutral-500 font-mono mt-0.5 uppercase">{t.thought_stream.desc}</p>
        </div>

        <div className="flex flex-wrap gap-2 w-full md:w-auto font-mono">
          {/* Search Box */}
          <div className="relative flex-1 md:flex-initial">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-500" />
            <input
              type="text"
              id="thought-search-input"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder={t.thought_stream.search_placeholder}
              className="bg-black border border-white/10 rounded py-1.5 pl-9 pr-3 text-xs text-neutral-100 placeholder-neutral-600 focus:outline-none focus:border-[#4ade80] w-full md:w-56 font-mono uppercase"
            />
          </div>

          {/* Category Select */}
          <select
            id="category-filter-select"
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="bg-black border border-white/10 rounded py-1.5 px-3 text-xs text-neutral-300 focus:outline-none focus:border-[#4ade80] uppercase"
          >
            {categories.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>

          {/* Phase Select */}
          <select
            id="phase-filter-select"
            value={selectedPhase}
            onChange={(e) => setSelectedPhase(e.target.value)}
            className="bg-black border border-white/10 rounded py-1.5 px-3 text-xs text-neutral-300 focus:outline-none focus:border-[#4ade80] uppercase"
          >
            {phases.map((p) => (
              <option key={p.value} value={p.value}>{p.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Stream Area */}
      <div id="thought-logs-scroll-area" className="flex-1 overflow-y-auto space-y-4 max-h-[500px] pr-2">
        {filteredThoughts.length === 0 ? (
          <div className="text-center py-12 text-neutral-600 font-mono text-xs uppercase">
            {t.thought_stream.no_thoughts}
          </div>
        ) : (
          filteredThoughts.map((thought, idx) => {
            const Icon = getCategoryIcon(thought.category);
            const styleClass = getCategoryColor(thought.category);
            const formattedTime = new Date(thought.timestamp).toLocaleTimeString();

            return (
              <motion.div
                key={thought.id}
                id={`thought-log-item-${thought.id}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.25 }}
                className={`border rounded-lg p-4 relative overflow-hidden transition-all duration-200 ${styleClass}`}
              >
                {/* Visual marker */}
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-current opacity-70" />

                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded bg-neutral-950">
                      <Icon className="w-4 h-4" />
                    </div>
                    <div>
                      <span className="text-xs font-mono font-bold uppercase tracking-wider">
                        {thought.category === "monologue" ? t.categories.monologue :
                         thought.category === "question" ? t.categories.question :
                         thought.category === "decision" ? t.categories.decision :
                         thought.category === "check" ? t.categories.check :
                         thought.category === "system" ? t.categories.system :
                         thought.category}
                      </span>
                      <span className="text-[10px] text-neutral-500 font-mono ml-2">
                        Phase: {
                          thought.phase === "perceive" ? t.phases.perceive :
                          thought.phase === "analyze" ? t.phases.analyze :
                          thought.phase === "plan" ? t.phases.plan :
                          thought.phase === "execute" ? t.phases.execute :
                          thought.phase === "verify" ? t.phases.verify :
                          thought.phase === "commit" ? t.phases.commit :
                          thought.phase === "completed" ? t.phases.completed :
                          thought.phase
                        }
                      </span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono text-neutral-500">{formattedTime}</span>
                </div>

                <div className="text-sm leading-relaxed text-neutral-200 font-mono whitespace-pre-wrap pl-1">
                  {thought.text}
                </div>
              </motion.div>
            );
          })
        )}
      </div>
    </div>
  );
}
