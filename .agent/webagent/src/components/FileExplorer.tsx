/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from "react";
import { File, Folder, Code, Terminal, FileCode, CheckCircle2, Copy, Check } from "lucide-react";
import { useLanguage } from "../context/LanguageContext";

interface FileExplorerProps {
  files: Record<string, string>;
}

export default function FileExplorer({ files }: FileExplorerProps) {
  const { language, t } = useLanguage();
  const fileNames = Object.keys(files);
  const [selectedFile, setSelectedFile] = useState<string>(() => {
    const saved = localStorage.getItem("jinx_selected_file");
    if (saved && fileNames.includes(saved)) {
      return saved;
    }
    return fileNames[0] || "";
  });
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  const copyTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (selectedFile) {
      localStorage.setItem("jinx_selected_file", selectedFile);
    }
  }, [selectedFile]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    };
  }, []);

  const handleCopy = async (text: string) => {
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    try {
      await navigator.clipboard.writeText(text);
      setCopyFailed(false);
      setCopied(true);
      copyTimerRef.current = setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy to clipboard", err);
      setCopied(false);
      setCopyFailed(true);
      copyTimerRef.current = setTimeout(() => setCopyFailed(false), 2000);
    }
  };

  const getFileIcon = (name: string) => {
    if (name.endsWith(".yaml") || name.endsWith(".yml")) {
      return <FileCode className="w-4 h-4 text-[#4ade80]" />;
    }
    if (name.endsWith(".json")) {
      return <Code className="w-4 h-4 text-neutral-300" />;
    }
    if (name.endsWith(".patch") || name.endsWith(".diff")) {
      return <Terminal className="w-4 h-4 text-[#4ade80]" />;
    }
    return <File className="w-4 h-4 text-neutral-400" />;
  };

  const renderFileContent = (name: string, rawContent: string) => {
    if (!rawContent) {
      return (
        <div className="text-neutral-600 font-mono text-xs uppercase">
          {language === "ru" ? "ФАЙЛ ПУСТ." : "FILE IS EMPTY."}
        </div>
      );
    }

    // Check if it's JSON
    if (name.endsWith(".json")) {
      try {
        const parsed = JSON.parse(rawContent);
        return (
          <pre className="text-xs text-neutral-300 font-mono overflow-x-auto whitespace-pre h-full">
            {JSON.stringify(parsed, null, 2)}
          </pre>
        );
      } catch (e) {
        // Fallback to normal text if not valid json
      }
    }

    return (
      <pre className="text-xs text-neutral-300 font-mono overflow-x-auto whitespace-pre-wrap h-full leading-relaxed">
        {rawContent}
      </pre>
    );
  };

  return (
    <div id="file-explorer-card" className="bg-[#0c0c0e]/90 border border-white/10 rounded-lg shadow-xl flex flex-col md:flex-row h-[600px] md:h-[500px] overflow-hidden">
      {/* File Tree Sidebar */}
      <div className="w-full md:w-64 border-b md:border-b-0 md:border-r border-white/10 flex flex-col bg-neutral-950/80 max-h-[160px] md:max-h-none flex-shrink-0">
        <div className="p-4 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Folder className="w-4 h-4 text-[#4ade80]" />
            <span className="text-[11px] font-mono font-extrabold uppercase tracking-widest text-neutral-300">.agent /</span>
          </div>
          <span className="text-[10px] font-mono text-neutral-500 uppercase">{fileNames.length} {t.file_explorer.files_count}</span>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {fileNames.length === 0 ? (
            <div className="text-center py-8 text-neutral-600 font-mono text-xs uppercase">
              {t.file_explorer.no_files}
            </div>
          ) : (
            fileNames.map((name) => {
              const isSelected = selectedFile === name;
              return (
                <button
                  key={name}
                  id={`file-tree-item-${name}`}
                  onClick={() => setSelectedFile(name)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded font-mono text-xs text-left transition-all duration-150 cursor-pointer ${
                    isSelected
                      ? "bg-[#4ade80]/10 border-l-2 border-[#4ade80] text-[#4ade80] font-bold"
                      : "text-neutral-400 hover:bg-white/5 hover:text-neutral-200"
                  }`}
                >
                  {getFileIcon(name)}
                  <span className="truncate flex-1">{name}</span>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Code Editor View */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0c0c0e]/30">
        {selectedFile && files[selectedFile] ? (
          <>
            {/* Header / Actions */}
            <div className="p-3 bg-neutral-950/80 border-b border-white/10 flex items-center justify-between px-6">
              <div className="flex items-center gap-2 min-w-0">
                <File className="w-4 h-4 text-neutral-500 flex-shrink-0" />
                <span className="text-xs font-mono font-bold text-neutral-300 uppercase tracking-wide truncate">{selectedFile}</span>
              </div>
              <button
                id="copy-file-btn"
                onClick={() => handleCopy(files[selectedFile])}
                className="p-1.5 rounded border border-white/10 hover:bg-white/5 text-neutral-400 hover:text-[#4ade80] transition-colors flex items-center gap-1.5 text-xs font-mono uppercase cursor-pointer"
              >
                {copyFailed ? (
                  <span className="text-red-400 font-bold">
                    {t.file_explorer.copy_failed}
                  </span>
                ) : copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-[#4ade80]" />
                    <span className="text-[#4ade80] font-bold">{t.file_explorer.copied}</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>{t.file_explorer.copy_to_clipboard}</span>
                  </>
                )}
              </button>
            </div>

            {/* Code Body */}
            <div id="file-code-viewer" className="flex-1 overflow-auto p-6 bg-black/40">
              {renderFileContent(selectedFile, files[selectedFile])}
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-neutral-600 font-mono text-xs uppercase text-center px-4">
            <CheckCircle2 className="w-8 h-8 text-neutral-800 mb-2" />
            {t.file_explorer.select_file_placeholder}
          </div>
        )}
      </div>
    </div>
  );
}
