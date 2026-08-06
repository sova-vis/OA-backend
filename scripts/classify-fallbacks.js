/**
 * Re-classify only the keyword-tagger's fallbacks (short MCQ stems that matched
 * no rule and got the subject default) with the LLM, then write the better topic
 * into the JSON. Small, short-text set -> flows under the free-tier rate limit.
 *
 *   node scripts/classify-fallbacks.js Commerce Economics
 */
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const dotenv = require("dotenv");
const Groq = require("groq-sdk");
const TAX = require("./topic-taxonomies");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
const groq = new Groq({ apiKey: (process.env.GROQ_API_KEY || "").trim() });
const MODEL = "llama-3.1-8b-instant";
const DATA = "D:/office/O_LEVEL_INGEST";
const SCRATCH = "C:/Users/SHEKHA~1/AppData/Local/Temp/claude/D--office-css-proj/c70abeb6-f453-4171-a05f-3ad21fc07057/scratchpad";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "");
const hkey = (t) => crypto.createHash("sha1").update(t.slice(0, 600)).digest("hex").slice(0, 12);

function qtext(q) {
  const b = [q.question_text, q.stem, q.intro_text, q.preview_text, q.prompt,
    q.context && typeof q.context === "object" ? q.context.text : null];
  for (const p of q.parts || []) if (p && typeof p === "object") b.push(p.question_text || p.text || p.body);
  if (q.options && typeof q.options === "object") b.push(Array.isArray(q.options) ? q.options.map((o) => o && o.text).join(" ") : Object.values(q.options).join(" "));
  return b.filter((x) => typeof x === "string").join(" ");
}
const canon = (raw, topics) => {
  if (!raw) return null;
  const r = norm(raw);
  return topics.find((t) => norm(t) === r) || topics.find((t) => norm(t).includes(r) || r.includes(norm(t))) || null;
};

async function classify(subject, topics, items) {
  const list = topics.map((t, i) => `${i + 1}. ${t}`).join("\n");
  const qs = items.map((it, i) => `[${i + 1}] ${it[1].slice(0, 160)}`).join("\n");
  const sys = `You are a Cambridge O Level ${subject} examiner. Classify each question into exactly ONE topic from the list, closest match. Reply ONLY compact json {"results":[{"n":1,"topic":"..."}]}, keys n and topic only.`;
  for (let a = 0; a < 6; a++) {
    try {
      const r = await groq.chat.completions.create({ model: MODEL, temperature: 0, max_tokens: items.length * 32 + 64, response_format: { type: "json_object" }, messages: [{ role: "system", content: `${sys}\nTopics:\n${list}` }, { role: "user", content: qs }] });
      const arr = (JSON.parse(r.choices[0].message.content).results) || [];
      const byN = new Map(arr.map((o) => [Number(o.n), o.topic]));
      return items.map((_, i) => canon(byN.get(i + 1), topics));
    } catch (e) {
      if (a === 5) return items.map(() => null);
      await sleep((/429|rate/i.test(String(e.message)) ? 4000 : 1200) * (a + 1));
    }
  }
}

async function run(subject) {
  const topics = TAX[subject];
  const fb = JSON.parse(fs.readFileSync(path.join(SCRATCH, `${subject}_fallbacks.json`), "utf8")); // [[key, stem]]
  const result = {};           // fallback-key -> topic
  const BATCH = 20;
  for (let i = 0; i < fb.length; i += BATCH) {
    const batch = fb.slice(i, i + BATCH);
    const out = await classify(subject, topics, batch);
    batch.forEach((it, j) => { if (out[j]) result[it[0]] = out[j]; });
    process.stdout.write(`\r  ${subject}: ${Math.min(i + BATCH, fb.length)}/${fb.length}   `);
    await sleep(500);
  }
  process.stdout.write("\n");
  // apply: recompute each question's fallback key and overwrite topic if we got one
  let written = 0;
  for (const sub of ["mcqs_by_year", "questions_by_year"]) {
    const dir = path.join(DATA, subject, sub);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir).filter((x) => x.endsWith(".json"))) {
      const fp = path.join(dir, f);
      const doc = JSON.parse(fs.readFileSync(fp, "utf8"));
      let ch = false;
      for (const q of doc.questions || doc.mcqs || []) {
        const t = qtext(q);
        if (t.length < 6) continue;
        const chosen = result[hkey(t)];   // fallback keys were hashed on qtext[:600]
        if (chosen && q.topic !== chosen) { q.topic = chosen; ch = true; written++; }
      }
      if (ch) fs.writeFileSync(fp, JSON.stringify(doc), "utf8");
    }
  }
  console.log(`  ${subject}: re-tagged ${written} questions from ${Object.keys(result).length} classified fallbacks`);
}

(async () => {
  for (const s of process.argv.slice(2)) { try { await run(s); } catch (e) { console.error(s, "FAIL", e.message); } }
  console.log("DONE");
})();
