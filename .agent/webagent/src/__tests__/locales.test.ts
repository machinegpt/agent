import { en } from "../locales/en";
import { ru } from "../locales/ru";

function flattenKeys(obj: Record<string, unknown>, prefix = ""): string[] {
  return Object.entries(obj).flatMap(([key, val]) =>
    val && typeof val === "object" && !Array.isArray(val)
      ? flattenKeys(val as Record<string, unknown>, `${prefix}${key}.`)
      : [`${prefix}${key}`]
  );
}

const enKeys = new Set(flattenKeys(en as unknown as Record<string, unknown>));
const ruKeys = new Set(flattenKeys(ru as unknown as Record<string, unknown>));

describe("locales", () => {
  it("en and ru have the same keys", () => {
    const extraInRu = [...ruKeys].filter(k => !enKeys.has(k));
    const missingInRu = [...enKeys].filter(k => !ruKeys.has(k));
    expect({ extraInRu, missingInRu }).toEqual({ extraInRu: [], missingInRu: [] });
  });
});
