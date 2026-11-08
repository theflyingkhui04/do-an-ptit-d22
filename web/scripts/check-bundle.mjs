// Fails when the built browser bundle mentions a backend credential name or a test secret.
// Run after `npm run build` (CHECK_SECRET may hold an extra literal to look for).
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const root = new URL("../.next/static", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const patterns = ["MODAL_PROXY_TOKEN", "SUPABASE_SERVICE_KEY", "DEMO_DATABASE_URL", "DEMO_READER_DATABASE_URL",
  "postgresql://", process.env.CHECK_SECRET].filter(Boolean);
const files = (dir) => readdirSync(dir).flatMap((name) => {
  const path = join(dir, name);
  return statSync(path).isDirectory() ? files(path) : [path];
});
const offenders = files(root).filter((path) => {
  const text = readFileSync(path, "utf-8");
  return patterns.some((p) => text.includes(p));
});
if (offenders.length) {
  console.error("Secrets or credential names in the bundle:", offenders);
  process.exit(1);
}
console.log(`check-bundle: ${files(root).length} files, no credential found`);
