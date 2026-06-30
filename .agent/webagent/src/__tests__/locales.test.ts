import { en } from "../locales/en";
import { ru } from "../locales/ru";
import type { TranslationDict } from "../locales/types";

type NestedKeyOf<T> = T extends object
  ? { [K in keyof T]: K extends string ? T[K] extends object ? `${K}.${NestedKeyOf<T[K]>}` : K : never }[keyof T]
  : never;

type LeafKeys = NestedKeyOf<TranslationDict>;

function flattenKeys(obj: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(obj).flatMap(([key, val]) =>
    val && typeof val === "object" && !Array.isArray(val)
      ? flattenKeys(val as Record<string, unknown>, `${prefix}${key}.`)
      : [`${prefix}${key}`]
  );
}

const typeKeys = new Set(flattenKeys(
  Object.fromEntries(
    Object.entries({
      header: {} as TranslationDict["header"],
      sidebar: {} as TranslationDict["sidebar"],
      tabs: {} as TranslationDict["tabs"],
      session_info: {} as TranslationDict["session_info"],
      cognitive_loop: {} as TranslationDict["cognitive_loop"],
      thought_stream: {} as TranslationDict["thought_stream"],
      file_explorer: {} as TranslationDict["file_explorer"],
      terminal: {} as TranslationDict["terminal"],
      diff_viewer: {} as TranslationDict["diff_viewer"],
      run_summary: {} as TranslationDict["run_summary"],
      phases: {} as TranslationDict["phases"],
      categories: {} as TranslationDict["categories"],
      cognitive_loop_phases: {} as TranslationDict["cognitive_loop_phases"],
      loading: {} as TranslationDict["loading"],
    }).map(([k, v]) => [k, v])
  )
));

const enKeys = new Set(flattenKeys(en as unknown as Record<string, unknown>));
const ruKeys = new Set(flattenKeys(ru as unknown as Record<string, unknown>));

describe("locales", () => {
  it("en has all keys from types.ts", () => {
    const missing = [...typeKeys].filter(k => !enKeys.has(k));
    expect(missing).toEqual([]);
  });

  it("ru has all keys from types.ts", () => {
    const missing = [...typeKeys].filter(k => !ruKeys.has(k));
    expect(missing).toEqual([]);
  });

  it("en and ru have the same keys", () => {
    const extraInRu = [...ruKeys].filter(k => !enKeys.has(k));
    const missingInRu = [...enKeys].filter(k => !ruKeys.has(k));
    expect({ extraInRu, missingInRu }).toEqual({ extraInRu: [], missingInRu: [] });
  });

  it("loading key exists in both locales", () => {
    expect(en.loading).toBeTruthy();
    expect(ru.loading).toBeTruthy();
  });
});
