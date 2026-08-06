/**
 * Tag every question in the untagged subjects with a syllabus topic.
 *
 * Strategy the user asked for: match each question to the closest topic from the
 * standard (CAIE) topical taxonomy, using the LLM to do the matching accurately.
 * Questions are de-duplicated by normalized text first, so shared MCQs/repeated
 * structured questions cost one classification, not many.
 *
 *   node scripts/tag-topics.js --subject="Biology"
 *   node scripts/tag-topics.js                     # every untagged subject
 *   node scripts/tag-topics.js --subject="History" --dry-run
 *
 * Writes the `topic` field back into <data-root>/<Subject>/.../*.json and keeps a
 * resumable checkpoint at scripts/.topic-cache/<Subject>.json (text hash -> topic),
 * so a re-run after a rate-limit stop continues instead of re-classifying.
 */
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const dotenv = require("dotenv");
const Groq = require("groq-sdk");
const TAXONOMIES = require("./topic-taxonomies");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
dotenv.config({ path: path.resolve(__dirname, "..", ".env.local") });

const groqKey = (process.env.GROQ_API_KEY || "").trim();
if (!groqKey) { console.error("GROQ_API_KEY missing in OA-backend/.env"); process.exit(1); }
const groq = new Groq({ apiKey: groqKey });

const args = new Map(process.argv.slice(2).map((a) => {
  const [k, v = "true"] = a.replace(/^--/, "").split("=");
  return [k, v];
}));
// Default to the fast, high-rate-limit model: topic classification against a
// fixed list is easy, and the 70B model's free-tier limit makes a job this size
// take hours. Override with --model= if needed.
const groqModel = args.get("model") || "llama-3.1-8b-instant";
const DATA_ROOT = path.resolve(args.get("data-root") || "D:/office/O_LEVEL_INGEST");
const onlySubject = args.get("subject");
const dryRun = args.has("dry-run");
const BATCH = Math.max(1, Number.parseInt(args.get("batch") || "12", 10) || 12);
const CACHE_DIR = path.resolve(__dirname, ".topic-cache");
fs.mkdirSync(CACHE_DIR, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function listJson(dir) {
  const out = [];
  for (const sub of ["mcqs_by_year", "questions_by_year"]) {
    const d = path.join(dir, sub);
    if (fs.existsSync(d)) for (const f of fs.readdirSync(d)) if (f.endsWith(".json")) out.push(path.join(d, f));
  }
  return out;
}

// Everything a reader sees, so classification has the full stem/context.
function questionText(q) {
  const bits = [q.question_text, q.stem, q.intro_text, q.preview_text, q.prompt,
    q.context && typeof q.context === "object" ? q.context.text : null];
  for (const p of q.parts || []) {
    if (p && typeof p === "object") bits.push(p.question_text || p.text || p.body);
  }
  if (q.options && typeof q.options === "object") {
    bits.push(Array.isArray(q.options) ? q.options.map((o) => o && o.text).join(" ")
      : Object.values(q.options).join(" "));
  }
  return bits.filter((b) => typeof b === "string" && b.trim()).join(" ").replace(/\s+/g, " ").trim();
}

const keyOf = (text) => crypto.createHash("sha1").update(text.slice(0, 600).toLowerCase()).digest("hex").slice(0, 16);

function canonicalize(raw, topics) {
  if (!raw) return null;
  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
  const r = norm(raw);
  let hit = topics.find((t) => norm(t) === r);
  if (hit) return hit;
  hit = topics.find((t) => norm(t).includes(r) || r.includes(norm(t)));
  return hit || null;
}

async function classifyBatch(subject, topics, items) {
  const list = topics.map((t, i) => `${i + 1}. ${t}`).join("\n");
  // The opening of a question reveals its topic; sending only ~180 chars keeps us
  // well under the free-tier tokens-per-minute limit, which is the real throughput
  // bottleneck for a job this size.
  const qs = items.map((it, i) => `[${i + 1}] ${it.text.slice(0, 180)}`).join("\n");
  const sys = `You are a Cambridge O Level ${subject} examiner. Classify each question into exactly ONE topic from the allowed list, choosing the single closest match. Never invent a topic. Reply ONLY with compact json {"results":[{"n":1,"topic":"<exact topic text>"}, ...]} — ${items.length} objects, one per question, in order. Each object has EXACTLY the keys "n" and "topic". Do NOT add descriptions, explanations, or any other key.`;
  const user = `Allowed topics:\n${list}\n\nQuestions:\n${qs}`;

  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      const res = await groq.chat.completions.create({
        model: groqModel,
        temperature: 0,
        max_tokens: items.length * 32 + 64,   // just {n,topic} per item; blocks runaway prose
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: sys },
          { role: "user", content: user },
        ],
      });
      const raw = res.choices?.[0]?.message?.content || "{}";
      const parsed = JSON.parse(raw);
      const arr = Array.isArray(parsed) ? parsed : parsed.results || parsed.items || [];
      const byN = new Map(arr.map((o) => [Number(o.n), o.topic]));
      return items.map((_, i) => canonicalize(byN.get(i + 1), topics));
    } catch (e) {
      const msg = String(e && e.message || e);
      const rate = /rate|429|quota|tokens per/i.test(msg);
      if (attempt === 5) { console.error("  batch failed:", msg.slice(0, 120)); return items.map(() => null); }
      await sleep(rate ? 4000 * (attempt + 1) : 1500 * (attempt + 1));
    }
  }
  return items.map(() => null);
}

async function tagSubject(subject) {
  const topics = TAXONOMIES[subject];
  if (!topics) { console.log(`(${subject}: no taxonomy, skipped)`); return; }
  const dir = path.join(DATA_ROOT, subject);
  if (!fs.existsSync(dir)) { console.log(`(${subject}: not found under data root)`); return; }

  const cacheFile = path.join(CACHE_DIR, `${subject}.json`);
  const cache = fs.existsSync(cacheFile) ? JSON.parse(fs.readFileSync(cacheFile, "utf8")) : {};

  const files = listJson(dir);
  // gather unique texts needing a topic
  const uniq = new Map();               // key -> text
  for (const f of files) {
    const doc = JSON.parse(fs.readFileSync(f, "utf8"));
    for (const q of doc.questions || doc.mcqs || []) {
      const text = questionText(q);
      if (!text || text.length < 8) continue;
      const k = keyOf(text);
      if (!cache[k] && !uniq.has(k)) uniq.set(k, text);
    }
  }
  const pending = [...uniq.entries()].map(([k, text]) => ({ k, text }));
  console.log(`${subject}: ${pending.length} unique texts to classify (${Object.keys(cache).length} cached)`);

  // Split into batches and run a small pool of them concurrently. Each ~15-item
  // call takes several seconds, so serial classification is the bottleneck;
  // CONC parallel calls cut wall-time to a fraction. classifyBatch backs off on
  // 429, so the pool self-throttles under the rate limit. Single process only,
  // so the in-memory cache can never be clobbered by a rival run.
  const batches = [];
  for (let i = 0; i < pending.length; i += BATCH) batches.push(pending.slice(i, i + BATCH));
  const CONC = Math.max(1, Number.parseInt(args.get("conc") || "4", 10) || 4);
  let next = 0, done = 0, sinceFlush = 0;
  const flush = () => { if (!dryRun) fs.writeFileSync(cacheFile, JSON.stringify(cache)); };

  async function worker() {
    while (next < batches.length) {
      const batch = batches[next++];
      const out = await classifyBatch(subject, topics, batch);
      // only cache a real classification; a failed/blank one is left uncached so a
      // re-run retries it rather than writing a bogus "Other"
      batch.forEach((it, j) => { if (out[j]) cache[it.k] = out[j]; });
      done += batch.length;
      if (++sinceFlush >= 4) { flush(); sinceFlush = 0; }
      process.stdout.write(`\r  ${subject}: classified ${done}/${pending.length}   `);
    }
  }
  await Promise.all(Array.from({ length: CONC }, worker));
  flush();
  process.stdout.write("\n");

  // write topics back into every question
  let written = 0, filesChanged = 0;
  for (const f of files) {
    const doc = JSON.parse(fs.readFileSync(f, "utf8"));
    let changed = false;
    for (const q of doc.questions || doc.mcqs || []) {
      const text = questionText(q);
      if (!text || text.length < 8) continue;
      const topic = cache[keyOf(text)];
      if (topic && q.topic !== topic) { q.topic = topic; changed = true; written++; }
    }
    if (changed && !dryRun) { fs.writeFileSync(f, JSON.stringify(doc, null, 0)); filesChanged++; }
  }
  const dist = {};
  for (const v of Object.values(cache)) dist[v] = (dist[v] || 0) + 1;
  console.log(`  ${subject}: wrote topic on ${written} questions across ${filesChanged} files${dryRun ? " (dry-run)" : ""}`);
  console.log(`  distribution:`, Object.fromEntries(Object.entries(dist).sort((a, b) => b[1] - a[1]).slice(0, 8)));
}

async function main() {
  const subjects = onlySubject ? [onlySubject] : Object.keys(TAXONOMIES);
  for (const s of subjects) {
    try { await tagSubject(s); } catch (e) { console.error(`${s} FAILED:`, e.message); }
  }
  console.log("DONE");
}
main();
