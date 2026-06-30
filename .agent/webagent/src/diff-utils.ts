import { CodeDiff } from "./types";

export function parseDiffText(filename: string, text: string): CodeDiff[] {
  if (!text) return [];

  const diffs: CodeDiff[] = [];
  const lines = text.split("\n");
  let currentFile = filename;
  let currentDiffLines: string[] = [];
  let additions = 0;
  let deletions = 0;

  for (const line of lines) {
    if (line.startsWith("diff --git")) {
      if (currentDiffLines.length > 0) {
        diffs.push({
          filepath: currentFile,
          filename: currentFile.split("/").pop() || currentFile,
          additions,
          deletions,
          diffText: currentDiffLines.join("\n"),
        });
      }
      currentDiffLines = [];
      additions = 0;
      deletions = 0;
      const match = line.match(/b\/(.+)$/);
      if (match) currentFile = match[1];
    }
    currentDiffLines.push(line);
    if (line.startsWith("+") && !line.startsWith("+++")) additions++;
    if (line.startsWith("-") && !line.startsWith("---")) deletions++;
  }

  if (currentDiffLines.length > 0) {
    diffs.push({
      filepath: currentFile,
      filename: currentFile.split("/").pop() || currentFile,
      additions,
      deletions,
      diffText: currentDiffLines.join("\n"),
    });
  }

  if (diffs.length === 0 && text.trim().length > 0) {
    return [{
      filepath: filename,
      filename: filename.split("/").pop() || filename,
      additions: text.split("\n").filter(l => l.startsWith("+")).length,
      deletions: text.split("\n").filter(l => l.startsWith("-")).length,
      diffText: text,
    }];
  }

  return diffs;
}
