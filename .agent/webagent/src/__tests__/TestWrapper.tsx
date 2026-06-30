import type { ReactNode } from "react";
import { LanguageProvider } from "../context/LanguageContext";

export function TestWrapper({ children }: { children: ReactNode }) {
  return <LanguageProvider>{children}</LanguageProvider>;
}
