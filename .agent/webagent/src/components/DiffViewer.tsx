/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { CodeDiff } from "../types";
import { GitPullRequest, FileEdit, Plus, Minus } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

interface DiffViewerProps {
  diffs: CodeDiff[];
}

export default function DiffViewer({ diffs }: DiffViewerProps) {
  const { t } = useLanguage();

  if (diffs.length === 0) {
    return (
      <div id="diffs-empty-card" className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg p-8 text-center text-neutral-500 font-mono text-xs uppercase">
        <GitPullRequest className="w-8 h-8 text-neutral-850 mx-auto mb-2" />
        {t.diff_viewer.no_diffs}
      </div>
    );
  }

  return (
    <div id="diff-viewer-container" className="space-y-6">
      {diffs.map((diff, index) => {
        const lines = diff.diffText.split("\n");

        return (
          <div key={diff.filepath} className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg shadow-xl overflow-hidden">
            {/* Header */}
            <div className="bg-neutral-950/80 px-4 py-3 border-b border-white/10 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileEdit className="w-4 h-4 text-[#4ade80]" />
                <span className="font-mono text-xs font-bold text-neutral-200 uppercase tracking-wide">{diff.filepath}</span>
              </div>

              <div className="flex items-center gap-2 font-mono text-xs">
                <span className="text-[#4ade80] flex items-center gap-0.5 font-bold">
                  <Plus className="w-3.5 h-3.5" /> {diff.additions}
                </span>
                <span className="text-rose-400 flex items-center gap-0.5 font-bold">
                  <Minus className="w-3.5 h-3.5" /> {diff.deletions}
                </span>
              </div>
            </div>

            {/* Diff Body */}
            <div className="overflow-x-auto">
              <table className="w-full border-collapse font-mono text-xs text-neutral-300">
                <tbody>
                  {lines.map((line, lineIdx) => {
                    const isAddition = line.startsWith("+") && !line.startsWith("+++");
                    const isDeletion = line.startsWith("-") && !line.startsWith("---");
                    const isHeader = line.startsWith("@@");

                    let rowBg = "hover:bg-white/[0.02]";
                    let numColor = "text-neutral-600";
                    let codeColor = "text-neutral-300";

                    if (isAddition) {
                      rowBg = "bg-[#4ade80]/5 hover:bg-[#4ade80]/10";
                      numColor = "text-[#4ade80] bg-[#4ade80]/5";
                      codeColor = "text-[#4ade80]";
                    } else if (isDeletion) {
                      rowBg = "bg-rose-950/20 hover:bg-rose-950/30";
                      numColor = "text-rose-600 bg-rose-950/10";
                      codeColor = "text-rose-400";
                    } else if (isHeader) {
                      rowBg = "bg-white/5";
                      numColor = "text-neutral-400 bg-white/5";
                      codeColor = "text-neutral-200 italic opacity-80";
                    }

                    return (
                      <tr key={lineIdx} className={`transition-colors duration-150 ${rowBg}`}>
                        {/* Line indices placeholder */}
                        <td className={`w-12 text-right pr-3 select-none border-r border-white/10 font-semibold py-0.5 ${numColor}`}>
                          {lineIdx + 1}
                        </td>
                        {/* Source code */}
                        <td className={`pl-4 whitespace-pre select-all align-middle py-0.5 ${codeColor}`}>
                          {line}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}
