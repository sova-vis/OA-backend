/**
 * Import O-Level JSONs into the subject-agnostic schema v2:
 *   - public.topics          (one row per subject+topic, drives the Topic dropdown)
 *   - public.questions       (type = 'mcq' | 'structured')
 *   - public.question_parts  ((a)(i)/(b)(ii) parts of structured questions)
 *
 * Source layout (per subject folder under O_Level_jsons/):
 *   <Subject>/mcqs_by_year/<year>.json     -> MCQs  (type = 'mcq')
 *   <Subject>/<batch>.json                 -> papers (type = 'structured')
 *
 * Usage:
 *   node scripts/import-o-level-v2.js --subject=physics --replace
 *   node scripts/import-o-level-v2.js                       (all subject folders)
 *
 * Flags:
 *   --subject=<folder>    only import this folder (case-insensitive). Omit for all.
 *   --replace             delete this subject's existing rows first (parts cascade)
 *   --data-root=<path>    override O_Level_jsons location
 *   --batch-size=<n>      questions per upsert (default 20; base64 images are heavy)
 *   --dry-run             parse + report only, write nothing
 */

const fs = require("fs/promises");
const path = require("path");
const dotenv = require("dotenv");
const { createClient } = require("@supabase/supabase-js");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
dotenv.config({ path: path.resolve(__dirname, "..", ".env.local") });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;
const DEFAULT_DATA_ROOT = path.resolve(__dirname, "..", "..", "O_Level_jsons");

const args = new Map(
  process.argv.slice(2).map((arg) => {
    const [key, value = "true"] = arg.replace(/^--/, "").split("=");
    return [key, value];
  }),
);

const dataRoot = path.resolve(args.get("data-root") || DEFAULT_DATA_ROOT);
const onlySubject = args.get("subject");
const dryRun = args.has("dry-run");
const replaceExisting = args.has("replace");
const batchSize = Math.max(1, Number.parseInt(args.get("batch-size") || "20", 10) || 20);

if (!SUPABASE_URL || !SERVICE_KEY) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in OA-backend/.env");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

function cleanText(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/â€“/g, "-")
    .replace(/â€”/g, "-")
    .replace(/â€˜/g, "'")
    .replace(/â€™/g, "'")
    .replace(/â€œ/g, '"')
    .replace(/â€/g, '"')
    .replace(/â‰¤/g, "<=")
    .replace(/â‰¥/g, ">=")
    .replace(/â»/g, "-")
    .replace(/Â/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function intOrNull(...values) {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const n = Number.parseInt(String(value).replace(/[^\d-]/g, ""), 10);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function asJsonObjectOrNull(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

function paperRank(paper) {
  const match = String(paper || "").match(/paper[_\s-]*(\d+)/i);
  return match ? Number.parseInt(match[1], 10) : 99;
}

// ---------------------------------------------------------------------------
// Topic cache: upsert (subject, name) once, remember its uuid.
// ---------------------------------------------------------------------------
const topicCache = new Map();

async function resolveTopicId(subject, name, theme, syllabusRef) {
  const cleanName = cleanText(name) || "Uncategorised";
  const key = `${subject}|${cleanName.toLowerCase()}`;
  if (topicCache.has(key)) return topicCache.get(key);

  if (dryRun) {
    topicCache.set(key, null);
    return null;
  }

  const { data, error } = await supabase
    .from("topics")
    .upsert(
      { subject, name: cleanName, theme: cleanText(theme) || null, syllabus_ref: cleanText(syllabusRef) || null },
      { onConflict: "subject,name" },
    )
    .select("id")
    .single();

  if (error) throw new Error(`Topic upsert failed for ${subject}/${cleanName}: ${error.message}`);
  topicCache.set(key, data.id);
  return data.id;
}

// ---------------------------------------------------------------------------
// Upsert helpers with retry + split-on-failure (mirrors the legacy importer).
// ---------------------------------------------------------------------------
async function writeWithRetry(table, rows, { onConflict, returning } = {}) {
  let lastError = null;
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    let q = onConflict ? supabase.from(table).upsert(rows, { onConflict }) : supabase.from(table).insert(rows);
    if (returning) q = q.select(returning);
    const { data, error } = await q;
    if (!error) return data;
    lastError = error;
    if (attempt < 4) await sleep(attempt * 750);
  }

  if (rows.length > 1) {
    const mid = Math.ceil(rows.length / 2);
    const a = await writeWithRetry(table, rows.slice(0, mid), { onConflict, returning });
    const b = await writeWithRetry(table, rows.slice(mid), { onConflict, returning });
    return returning ? [...(a || []), ...(b || [])] : null;
  }

  throw new Error(`${table} write failed: ${lastError?.message || lastError}`);
}

// Flush a batch of { row, parts } items: upsert questions, then their parts.
async function flushBatch(items) {
  if (items.length === 0 || dryRun) return;

  const rows = items.map((item) => item.row);
  const inserted = await writeWithRetry("questions", rows, { onConflict: "question_id", returning: "id,question_id" });

  const idByQuestionId = new Map((inserted || []).map((r) => [r.question_id, r.id]));

  const partRows = [];
  const parentUids = [];
  for (const item of items) {
    if (!item.parts || item.parts.length === 0) continue;
    const uid = idByQuestionId.get(item.row.question_id);
    if (!uid) continue;
    parentUids.push(uid);
    item.parts.forEach((part, index) => {
      partRows.push({
        question_uid: uid,
        label: cleanText(part.part) || `(${index + 1})`,
        order_index: index,
        body: cleanText(part.question_text),
        marks: intOrNull(part.marks),
        answer: cleanText(part.answer) || null,
      });
    });
  }

  // Idempotent re-import: clear any existing parts for these questions first.
  if (parentUids.length > 0) {
    const { error: delErr } = await supabase.from("question_parts").delete().in("question_uid", parentUids);
    if (delErr) throw new Error(`Could not clear old parts: ${delErr.message}`);
  }

  if (partRows.length > 0) {
    await writeWithRetry("question_parts", partRows);
  }
}

// ---------------------------------------------------------------------------
// Flatten the nested papers.{year}.{session}.{paper}.{variant}[] structure.
// ---------------------------------------------------------------------------
function flattenStructured(batchData) {
  const rows = [];
  const papersByYear = asJsonObjectOrNull(batchData.papers) || {};

  for (const year of Object.keys(papersByYear).sort()) {
    if (!/^\d{4}$/.test(year)) continue; // skip metadata keys
    const sessions = papersByYear[year] || {};
    for (const session of Object.keys(sessions).sort()) {
      const papers = sessions[session] || {};
      for (const paper of Object.keys(papers).sort((a, b) => paperRank(a) - paperRank(b) || a.localeCompare(b))) {
        const variants = papers[paper] || {};
        for (const variant of Object.keys(variants).sort()) {
          const questions = Array.isArray(variants[variant]) ? variants[variant] : [];
          for (const question of questions) {
            rows.push({ ...question, year, session, paper, variant });
          }
        }
      }
    }
  }
  return rows;
}

async function clearSubject(subject) {
  // question_parts cascade-delete via FK when questions are removed.
  let deleted = 0;
  while (true) {
    const { data: rows, error } = await supabase
      .from("questions")
      .select("id")
      .eq("subject", subject)
      .limit(500);
    if (error) throw new Error(`Could not select existing ${subject} questions: ${error.message}`);
    if (!rows || rows.length === 0) break;
    const ids = rows.map((r) => r.id);
    const { error: delErr } = await supabase.from("questions").delete().in("id", ids);
    if (delErr) throw new Error(`Could not clear existing ${subject} questions: ${delErr.message}`);
    deleted += ids.length;
  }
  if (deleted > 0) console.log(`Cleared ${deleted} existing ${subject} question rows.`);
}

async function importSubjectFolder(folderName) {
  const subjectDir = path.join(dataRoot, folderName);
  const mcqDir = path.join(subjectDir, "mcqs_by_year");

  const mcqFiles = (await fs.readdir(mcqDir)).filter((f) => /^\d{4}\.json$/.test(f)).sort();
  const batchFiles = (await fs.readdir(subjectDir))
    .filter((f) => /^\d{4}-\d{4}\.json$|^\d{4}-onwards\.json$/.test(f))
    .sort();

  // Canonical subject name from the JSON itself (folder may be lower-cased).
  let subject = folderName;
  if (mcqFiles.length) {
    const first = await readJson(path.join(mcqDir, mcqFiles[0]));
    subject = cleanText(first.subject) || folderName;
  } else if (batchFiles.length) {
    const first = await readJson(path.join(subjectDir, batchFiles[0]));
    subject = cleanText(first.subject) || folderName;
  }

  if (replaceExisting && !dryRun) await clearSubject(subject);

  let mcqCount = 0;
  let structCount = 0;
  let partCount = 0;
  let batch = [];

  const flush = async () => {
    await flushBatch(batch);
    batch = [];
  };

  // ---- MCQs --------------------------------------------------------------
  for (const file of mcqFiles) {
    const data = await readJson(path.join(mcqDir, file));
    const fallbackYear = data.year || file.replace(".json", "");
    const questions = Array.isArray(data.mcqs) ? data.mcqs : [];

    for (const raw of questions) {
      const subj = cleanText(raw.subject) || subject;
      const topicId = await resolveTopicId(subj, raw.topic, raw.theme, raw.syllabus_ref);
      batch.push({
        row: {
          question_id: cleanText(raw.question_id),
          subject: subj,
          type: "mcq",
          exam_year: intOrNull(raw.year, fallbackYear),
          session: cleanText(raw.session),
          paper: cleanText(raw.paper),
          variant: cleanText(raw.variant),
          question_number: intOrNull(raw.question_number) ?? 0,
          topic: cleanText(raw.topic) || null,
          theme: cleanText(raw.theme) || null,
          topic_id: topicId,
          question_text: cleanText(raw.question_text),
          marks: intOrNull(raw.marks),
          options: asJsonObjectOrNull(raw.options),
          correct_option: cleanText(raw.correct_option) || null,
          marking_scheme: cleanText(raw.marking_scheme) || null,
          requires_diagram: Boolean(raw.requires_diagram),
          images: Array.isArray(raw.images) ? raw.images : [],
          reference: asJsonObjectOrNull(raw.reference),
        },
        parts: [],
      });
      mcqCount += 1;
      if (batch.length >= batchSize) await flush();
    }
  }

  // ---- Structured papers -------------------------------------------------
  for (const file of batchFiles) {
    const data = await readJson(path.join(subjectDir, file));
    const questions = flattenStructured(data);

    for (const raw of questions) {
      const subj = cleanText(raw.subject) || subject;
      const topicId = await resolveTopicId(subj, raw.topic, raw.theme, raw.syllabus_ref);
      const parts = Array.isArray(raw.parts) ? raw.parts : [];
      batch.push({
        row: {
          question_id: cleanText(raw.question_id),
          subject: subj,
          type: "structured",
          exam_year: intOrNull(raw.year),
          session: cleanText(raw.session),
          paper: cleanText(raw.paper),
          variant: cleanText(raw.variant),
          question_number: intOrNull(raw.question_number) ?? 0,
          topic: cleanText(raw.topic) || null,
          theme: cleanText(raw.theme) || null,
          topic_id: topicId,
          question_text: cleanText(raw.intro_text) || cleanText(raw.question_text),
          marks: intOrNull(raw.total_marks, raw.marks),
          options: null,
          correct_option: null,
          marking_scheme: cleanText(raw.marking_scheme) || null,
          requires_diagram: Boolean(raw.requires_diagram),
          images: Array.isArray(raw.images) ? raw.images : [],
          reference: asJsonObjectOrNull(raw.reference),
        },
        parts,
      });
      structCount += 1;
      partCount += parts.length;
      if (batch.length >= batchSize) await flush();
    }
  }

  await flush();
  console.log(
    `${dryRun ? "[dry-run] " : ""}${subject}: ${mcqCount} MCQs, ${structCount} structured questions, ${partCount} parts`,
  );
}

async function main() {
  const entries = await fs.readdir(dataRoot, { withFileTypes: true });
  const folders = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (onlySubject && entry.name.toLowerCase() !== onlySubject.toLowerCase()) continue;
    try {
      await fs.access(path.join(dataRoot, entry.name, "mcqs_by_year"));
      folders.push(entry.name);
    } catch {
      // not a subject folder
    }
  }

  if (folders.length === 0) throw new Error(`No subject folders found in ${dataRoot}`);

  console.log(`Importing ${folders.length} subject folder(s) from ${dataRoot}`);
  console.log(`Batch size: ${batchSize}${dryRun ? " (dry-run)" : ""}${replaceExisting ? " (replace)" : ""}`);

  for (const folder of folders.sort()) {
    await importSubjectFolder(folder);
  }

  console.log("O-Level v2 import complete.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
