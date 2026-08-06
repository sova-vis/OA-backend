/**
 * Push the topics tagged into the O_LEVEL_INGEST JSONs to the live database,
 * WITHOUT a full re-ingest (which would re-write all the base64 images). Every
 * question already has a stable question_id, so this only:
 *   1. upserts each subject's distinct topics into public.topics (-> topic_id)
 *   2. UPDATEs questions.topic + topic_id, grouped by topic and batched by id
 *
 *   node scripts/update-topics.js --subject="Biology"
 *   node scripts/update-topics.js                      # every tagged subject
 *   node scripts/update-topics.js --dry-run
 */
const fs = require("fs");
const path = require("path");
const dotenv = require("dotenv");
const { createClient } = require("@supabase/supabase-js");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
dotenv.config({ path: path.resolve(__dirname, "..", ".env.local") });

const url = process.env.SUPABASE_URL;
const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;
if (!url || !key) { console.error("Missing SUPABASE_URL / service key"); process.exit(1); }
const sb = createClient(url, key, { auth: { persistSession: false } });

const args = new Map(process.argv.slice(2).map((a) => {
  const [k, v = "true"] = a.replace(/^--/, "").split("=");
  return [k, v];
}));
const DATA_ROOT = path.resolve(args.get("data-root") || "D:/office/O_LEVEL_INGEST");
const onlySubject = args.get("subject");
const dryRun = args.has("dry-run");

// the 13 subjects that were topic-tagged
const SUBJECTS = ["Accounting", "Art and Design", "Biology", "Business Studies",
  "Commerce", "Computer Science", "Economics", "Environmental Management",
  "Geography", "History", "Religious Studies", "Sociology", "Statistics"];

const listJson = (dir) => {
  const out = [];
  for (const sub of ["mcqs_by_year", "questions_by_year"]) {
    const d = path.join(dir, sub);
    if (fs.existsSync(d)) for (const f of fs.readdirSync(d)) if (f.endsWith(".json")) out.push(path.join(d, f));
  }
  return out;
};

const chunk = (arr, n) => { const o = []; for (let i = 0; i < arr.length; i += n) o.push(arr.slice(i, i + n)); return o; };

async function upsertTopic(subject, name) {
  const { data, error } = await sb.from("topics")
    .upsert({ subject, name }, { onConflict: "subject,name" })
    .select("id").single();
  if (error) throw new Error(`topic upsert ${subject}/${name}: ${error.message}`);
  return data.id;
}

async function updateSubject(subject) {
  const dir = path.join(DATA_ROOT, subject);
  if (!fs.existsSync(dir)) { console.log(`(${subject}: not found)`); return; }

  // question_id -> topic, from the tagged JSON
  const byTopic = new Map();         // topic -> [question_id]
  let tagged = 0, untagged = 0;
  for (const f of listJson(dir)) {
    const doc = JSON.parse(fs.readFileSync(f, "utf8"));
    for (const q of doc.questions || doc.mcqs || []) {
      if (!q.question_id) continue;
      const t = (q.topic || "").trim();
      if (!t) { untagged++; continue; }
      tagged++;
      if (!byTopic.has(t)) byTopic.set(t, []);
      byTopic.get(t).push(q.question_id);
    }
  }
  console.log(`${subject}: ${tagged} tagged / ${untagged} untagged, ${byTopic.size} distinct topics`);
  if (byTopic.size === 0) return;
  if (dryRun) { console.log("  (dry-run)"); return; }

  let updated = 0;
  for (const [topic, ids] of byTopic) {
    const topicId = await upsertTopic(subject, topic);
    for (const batch of chunk(ids, 300)) {
      const { error, count } = await sb.from("questions")
        .update({ topic, topic_id: topicId }, { count: "exact" })
        .in("question_id", batch);
      if (error) { console.error(`  update failed (${topic}):`, error.message); continue; }
      updated += count || 0;
    }
  }
  console.log(`  ${subject}: updated ${updated} rows`);
}

async function main() {
  const subs = onlySubject ? [onlySubject] : SUBJECTS;
  for (const s of subs) {
    try { await updateSubject(s); } catch (e) { console.error(`${s} FAILED:`, e.message); }
  }
  console.log("DONE");
}
main();
