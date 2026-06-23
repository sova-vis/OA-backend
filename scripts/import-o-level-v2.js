/**
 * Import O-Level JSONs into the subject-agnostic schema v2:
 *   - public.topics          (one row per subject+topic, drives the Topic dropdown)
 *   - public.questions       (type = 'mcq' | 'structured')
 *   - public.question_parts  ((a)(i)/(b)(ii) parts of structured questions)
 *
 * Source layout (per subject folder under O_Level_jsons/):
 *   <Subject>/mcqs_by_year/<year>.json        -> MCQs  (type = 'mcq', flat mcqs[])
 *   <Subject>/question per year/<year>.json   -> written questions (type = 'structured', flat questions[])
 *
 * dedup_group is taken from the source JSON when present, else computed from content.
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
const crypto = require("crypto");
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

async function readDirSafe(dir) {
  try {
    return await fs.readdir(dir);
  } catch {
    return [];
  }
}

// Content fingerprint so the SAME question (recurring across variants/years) shares
// a dedup_group. Hash is over normalized TEXT only — re-scanned diagrams differ
// byte-wise across years, so including image bytes would defeat cross-year dedup.
function normContent(value) {
  return String(value == null ? "" : value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function dedupGroup(raw, type) {
  let content;
  if (type === "mcq") {
    const opts = ["A", "B", "C", "D"].map((k) => normContent((raw.options || {})[k])).join("~");
    content = "mcq||" + normContent(raw.mcq_stem || raw.question_text) + "||" + opts;
  } else {
    const partsText = (Array.isArray(raw.parts) ? raw.parts : [])
      .map((part) => normContent(part.part) + ":" + normContent(part.question_text))
      .join("|");
    content = "structured||" + normContent(raw.intro_text || raw.question_text) + "||" + partsText;
  }
  return crypto.createHash("sha1").update(content).digest("hex");
}

// Collect a question's images. Handles three shapes:
//   - raw.images[]            (Physics/Chemistry: stem/figure images)
//   - raw.image               (Mathematics: the whole question as one image)
//   - raw.answer_image        (Mathematics: the mark-scheme answer image)
// answer_image is tagged role 'answer' so the UI keeps it behind the reveal toggle.
// Some parts (e.g. English vocabulary-in-context) carry MCQ options. The parts
// table has no options column, so render them into the body text as labelled lines.
function formatPartOptions(options) {
  if (!options) return "";
  let entries = [];
  if (Array.isArray(options)) entries = options.map((value, i) => [String.fromCharCode(65 + i), value]);
  else if (typeof options === "object") entries = Object.entries(options);
  const lines = entries
    .map(([label, text]) => [String(label).toUpperCase(), cleanText(text)])
    .filter(([, text]) => text)
    .map(([label, text]) => `${label}. ${text}`);
  return lines.length ? "\n" + lines.join("\n") : "";
}

function buildImages(raw) {
  const imgs = [];
  if (Array.isArray(raw.images)) imgs.push(...raw.images);
  if (raw.image && typeof raw.image === "object" && raw.image.data_url) {
    imgs.push({ ...raw.image, role: raw.image.role || "question" });
  }
  if (raw.answer_image && typeof raw.answer_image === "object" && raw.answer_image.data_url) {
    imgs.push({ ...raw.answer_image, role: "answer" });
  }
  return imgs;
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
        body: cleanText(part.question_text) + formatPartOptions(part.options),
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

const isYearFile = (f) => /^\d{4}\.json$/.test(f);

// Build a stable id for MCQ practice questions that ship without a question_id.
const sessionAbbrev = (s) => (/may|june|m.?j/i.test(s) ? "MJ" : /oct|nov|o.?n/i.test(s) ? "ON" : String(s || "").replace(/[^A-Za-z0-9]+/g, "") || "XX");
const paperAbbrev = (p) => {
  const m = String(p || "").match(/(\d+)/);
  return m ? "P" + m[1] : String(p || "").replace(/[^A-Za-z0-9]+/g, "") || "P";
};
const subjectAbbrev = (s) => String(s || "").replace(/[^A-Za-z0-9]+/g, "");

async function importSubjectFolder(folderName) {
  const subjectDir = path.join(dataRoot, folderName);
  const mcqDir = path.join(subjectDir, "mcqs_by_year");

  const mcqFiles = (await readDirSafe(mcqDir)).filter(isYearFile).sort();

  // Structured year files live in "question per year/", or directly in the
  // subject folder (e.g. Mathematics: <Subject>/<year>.json).
  let structDir = path.join(subjectDir, "question per year");
  let structFiles = (await readDirSafe(structDir)).filter(isYearFile).sort();
  if (structFiles.length === 0) {
    structDir = subjectDir;
    structFiles = (await readDirSafe(subjectDir)).filter(isYearFile).sort();
  }

  // Canonical subject name from the JSON itself (folder may be lower-cased).
  let subject = folderName;
  if (mcqFiles.length) {
    const first = await readJson(path.join(mcqDir, mcqFiles[0]));
    subject = cleanText(first.subject) || folderName;
  } else if (structFiles.length) {
    const first = await readJson(path.join(structDir, structFiles[0]));
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
    // Most MCQ files use `mcqs`; AI practice MCQ banks use `questions`.
    const questions = Array.isArray(data.mcqs) ? data.mcqs : Array.isArray(data.questions) ? data.questions : [];
    const seqByGroup = new Map();

    for (const raw of questions) {
      const subj = cleanText(raw.subject) || subject;
      const year = intOrNull(raw.year, fallbackYear);
      const session = cleanText(raw.session);
      const paper = cleanText(raw.paper);

      let questionId = cleanText(raw.question_id);
      let variant = cleanText(raw.variant);
      let questionNumber = intOrNull(raw.question_number) ?? 0;

      // Practice MCQs ship without a question_id/variant and reuse paper question
      // numbers across several derived MCQs. Renumber sequentially per
      // (year, session, paper) and synthesise a stable id so both unique keys hold.
      if (!questionId) {
        const groupKey = [year, session, paper].join("|");
        questionNumber = (seqByGroup.get(groupKey) || 0) + 1;
        seqByGroup.set(groupKey, questionNumber);
        // The unique key has no `type`, so an MCQ and a written question for the
        // same paper/number would clash. Tag these practice MCQs so they don't.
        variant = "MCQ";
        questionId = `${subjectAbbrev(subj)}_${year}_${sessionAbbrev(session)}_${paperAbbrev(paper)}_MCQ_Q${questionNumber}`;
      }

      const topicId = await resolveTopicId(subj, raw.topic, raw.theme, raw.syllabus_ref);
      batch.push({
        row: {
          question_id: questionId,
          subject: subj,
          type: "mcq",
          exam_year: year,
          session,
          paper,
          variant,
          question_number: questionNumber,
          topic: cleanText(raw.topic) || null,
          theme: cleanText(raw.theme) || null,
          topic_id: topicId,
          question_text: cleanText(raw.mcq_stem) || cleanText(raw.question_text),
          marks: intOrNull(raw.marks),
          options: asJsonObjectOrNull(raw.options),
          correct_option: cleanText(raw.correct_option) || null,
          marking_scheme: cleanText(raw.marking_scheme) || cleanText(raw.answer) || null,
          requires_diagram: Boolean(raw.requires_diagram),
          images: buildImages(raw),
          reference: asJsonObjectOrNull(raw.reference),
          sources: Array.isArray(raw.sources) ? raw.sources : [],
          source_note: cleanText(raw.source_note) || null,
          dedup_group: cleanText(raw.dedup_group) || dedupGroup(raw, "mcq"),
        },
        parts: [],
      });
      mcqCount += 1;
      if (batch.length >= batchSize) await flush();
    }
  }

  // ---- Structured "question per year" files (flat questions[] arrays) -----
  for (const file of structFiles) {
    const data = await readJson(path.join(structDir, file));
    const fallbackYear = data.year || file.replace(".json", "");
    const questions = Array.isArray(data.questions) ? data.questions : [];

    for (const raw of questions) {
      const subj = cleanText(raw.subject) || subject;
      const topicId = await resolveTopicId(subj, raw.topic, raw.theme, raw.syllabus_ref);
      const parts = Array.isArray(raw.parts) ? raw.parts : [];
      // A non-numeric question_number suffix (e.g. "1(b)") collapses to the same
      // int and clashes on the composite key — fold the suffix into the variant.
      const rawQn = String(raw.question_number == null ? "" : raw.question_number).trim();
      const qnSuffix = rawQn.replace(/^\s*-?\d+/, "").trim();
      batch.push({
        row: {
          question_id: cleanText(raw.question_id),
          subject: subj,
          type: "structured",
          exam_year: intOrNull(raw.year, fallbackYear),
          session: cleanText(raw.session),
          paper: cleanText(raw.paper),
          variant: cleanText(raw.variant) + qnSuffix,
          question_number: intOrNull(raw.question_number) ?? 0,
          topic: cleanText(raw.topic) || null,
          theme: cleanText(raw.theme) || null,
          topic_id: topicId,
          question_text: cleanText(raw.intro_text) || cleanText(raw.preview_text) || cleanText(raw.question_text),
          marks: intOrNull(raw.total_marks, raw.marks),
          options: null,
          correct_option: null,
          marking_scheme: cleanText(raw.marking_scheme) || null,
          requires_diagram: Boolean(raw.requires_diagram || raw.image || raw.images),
          images: buildImages(raw),
          reference: asJsonObjectOrNull(raw.reference),
          sources: Array.isArray(raw.sources) ? raw.sources : [],
          source_note: cleanText(raw.source_note) || cleanText(raw.passage_note) || null,
          dedup_group: cleanText(raw.dedup_group) || dedupGroup(raw, "structured"),
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

    // A subject folder has MCQ year files, "question per year" files, or year
    // files directly inside it.
    const dir = path.join(dataRoot, entry.name);
    const hasMcq = (await readDirSafe(path.join(dir, "mcqs_by_year"))).some(isYearFile);
    const hasStructFolder = (await readDirSafe(path.join(dir, "question per year"))).some(isYearFile);
    const hasDirectYears = (await readDirSafe(dir)).some(isYearFile);
    if (hasMcq || hasStructFolder || hasDirectYears) folders.push(entry.name);
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
