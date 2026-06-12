const fs = require("fs/promises");
const path = require("path");
const dotenv = require("dotenv");
const { createClient } = require("@supabase/supabase-js");

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
dotenv.config({ path: path.resolve(__dirname, "..", ".env.local") });

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;
const BUCKET = process.env.O_LEVEL_QUESTION_ASSET_BUCKET || "question-assets";
const DEFAULT_DATA_ROOT = path.resolve(__dirname, "..", "..", "O_Level_jsons");

const args = new Map(
  process.argv.slice(2).map((arg) => {
    const [key, value = "true"] = arg.replace(/^--/, "").split("=");
    return [key, value];
  }),
);

const dataRoot = path.resolve(args.get("data-root") || DEFAULT_DATA_ROOT);
const imageMode = args.get("images") || "storage";
const onlySubject = args.get("subject");
const dryRun = args.has("dry-run");
const replaceExisting = args.has("replace");

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in OA-backend/.env");
  process.exit(1);
}

if (!["storage", "data-url"].includes(imageMode)) {
  console.error("Use --images=storage or --images=data-url");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

function slugify(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function cleanText(value) {
  if (value === null || value === undefined) return null;
  return String(value)
    .replace(/\u00e2\u20ac\u201c/g, "-")
    .replace(/\u00e2\u20ac\u201d/g, "-")
    .replace(/\u00e2\u20ac\u02dc/g, "'")
    .replace(/\u00e2\u20ac\u2122/g, "'")
    .replace(/\u00e2\u20ac\u0153/g, '"')
    .replace(/\u00e2\u20ac\u009d/g, '"')
    .replace(/\u00e2\u2030\u00a4/g, "<=")
    .replace(/\u00e2\u2030\u00a5/g, ">=")
    .replace(/\u00e2\u0081\u00bb/g, "-")
    .replace(/\u00c2/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function asJsonObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function parseOptionsFromExplicit(options) {
  if (!options) return [];

  if (Array.isArray(options)) {
    return options
      .map((item, index) => {
        if (typeof item === "string") return { label: String.fromCharCode(65 + index), text: cleanText(item) };
        if (item && typeof item === "object") {
          return {
            label: cleanText(item.label || item.option || String.fromCharCode(65 + index)).toUpperCase(),
            text: cleanText(item.text || item.value || item.answer || ""),
          };
        }
        return null;
      })
      .filter((item) => item && item.label && item.text);
  }

  if (typeof options === "object") {
    return Object.entries(options)
      .map(([label, text]) => ({ label: label.toUpperCase(), text: cleanText(text) }))
      .filter((item) => /^[A-D]$/.test(item.label) && item.text);
  }

  return [];
}

function parseQuestionText(questionText, explicitOptions) {
  const explicit = parseOptionsFromExplicit(explicitOptions);
  if (explicit.length >= 2) return { stem: questionText, options: explicit };

  const matches = Array.from(questionText.matchAll(/(?:^|\s)([A-D])\s+(?=\S)/g));

  for (let i = matches.length - 1; i >= 0; i -= 1) {
    if (matches[i][1] !== "A") continue;
    const positions = [];

    for (const label of ["A", "B", "C", "D"]) {
      const match = matches.slice(i).find((candidate) => candidate[1] === label);
      if (!match) break;
      positions.push({ label, index: match.index || 0, tokenLength: match[0].length });
    }

    if (positions.length !== 4) continue;
    if (!positions.every((item, index) => index === 0 || item.index > positions[index - 1].index)) continue;

    const options = positions.map((position, index) => {
      const start = position.index + position.tokenLength;
      const end = positions[index + 1] ? positions[index + 1].index : questionText.length;
      return { label: position.label, text: questionText.slice(start, end).trim() };
    });

    if (options.every((option) => option.text)) {
      return { stem: questionText.slice(0, positions[0].index).trim(), options };
    }
  }

  return { stem: questionText, options: [] };
}

function dataUrlParts(dataUrl) {
  const match = String(dataUrl || "").match(/^data:([^;]+);base64,(.+)$/);
  if (!match) return null;

  const mimeType = match[1];
  const ext = mimeType.includes("jpeg") ? "jpg" : mimeType.includes("webp") ? "webp" : "png";
  return {
    mimeType,
    ext,
    buffer: Buffer.from(match[2], "base64"),
  };
}

function paperRank(paper) {
  const match = String(paper || "").match(/paper[_\s-]*(\d+)/i);
  return match ? Number.parseInt(match[1], 10) : 99;
}

function questionSortKey(question) {
  return [
    Number.parseInt(String(question.question_number || "0").replace(/\D+/g, ""), 10) || 0,
    String(question.sub_question || ""),
  ].join(":");
}

function questionLookupKey(question) {
  return [
    question.year,
    question.session,
    question.paper,
    question.variant,
    question.question_number,
    question.sub_question || "",
  ]
    .map((part) => String(part || "").toLowerCase())
    .join("|");
}

function classifyQuestion(question, parsed) {
  if (question.correct_option || parsed.options.length >= 2) return "mcq";
  if (Number(question.marks) >= 20) return "written";
  return "structured";
}

function imageStoragePath(question, image, imageIndex) {
  if (image.path) return image.path.replace(/\\/g, "/").replace(/\.(png|jpg|jpeg|webp)$/i, "");

  const q = String(question.question_number || imageIndex + 1).padStart(2, "0");
  const role = image.role || "stem";
  const option = image.option ? `_opt_${String(image.option).toUpperCase()}` : "";
  return [
    question.subject,
    "images",
    question.year,
    question.session || "unknown-session",
    question.paper || "unknown-paper",
    question.variant || "unknown-variant",
    `q${q}_${role}${option}_${imageIndex + 1}`,
  ]
    .filter(Boolean)
    .join("/")
    .replace(/\\/g, "/");
}

async function normalizeImages(question, images) {
  if (!Array.isArray(images)) return [];

  const normalized = [];

  for (let imageIndex = 0; imageIndex < images.length; imageIndex += 1) {
    const image = images[imageIndex] || {};
    const base = {
      role: image.role || "stem",
      position: image.position || "after_question_text",
      option: image.option || null,
      path: image.path || image.storage_path || null,
      alt: cleanText(image.alt) || `Question ${question.question_number || imageIndex + 1} image`,
      exists: image.exists !== false,
      embedded: Boolean(image.embedded || image.data_url),
    };

    if (imageMode === "data-url") {
      normalized.push({
        ...base,
        data_url: image.data_url || null,
        public_url: image.public_url || image.url || null,
        storage_path: image.storage_path || image.path || null,
      });
      continue;
    }

    if (!image.data_url) {
      normalized.push({
        ...base,
        public_url: image.public_url || image.url || null,
        storage_path: image.storage_path || image.path || null,
        embedded: false,
      });
      continue;
    }

    const parts = dataUrlParts(image.data_url);
    if (!parts) {
      normalized.push({ ...base, data_url: image.data_url });
      continue;
    }

    const storagePath = `${imageStoragePath(question, image, imageIndex)}.${parts.ext}`;

    if (!dryRun) {
      const { error } = await supabase.storage.from(BUCKET).upload(storagePath, parts.buffer, {
        contentType: parts.mimeType,
        upsert: true,
      });

      if (error) throw new Error(`Image upload failed for ${storagePath}: ${error.message}`);
    }

    const { data } = supabase.storage.from(BUCKET).getPublicUrl(storagePath);

    normalized.push({
      ...base,
      path: image.path || storagePath,
      storage_path: storagePath,
      public_url: data.publicUrl,
      mime_type: parts.mimeType,
      bytes: parts.buffer.length,
      embedded: false,
    });
  }

  return normalized;
}

function questionId(subjectSlug, question) {
  return [
    subjectSlug,
    question.year,
    question.session,
    question.paper,
    question.variant,
    question.question_number,
    question.sub_question,
  ]
    .filter(Boolean)
    .map((part) => slugify(String(part)) || String(part).replace(/\W+/g, "-"))
    .join("-");
}

function uniqueQuestionId(baseId, idCounts) {
  const count = (idCounts.get(baseId) || 0) + 1;
  idCounts.set(baseId, count);
  return count === 1 ? baseId : `${baseId}-part-${count}`;
}

async function upsertSubject(subjectName) {
  const slug = slugify(subjectName);
  const { data, error } = await supabase
    .from("o_level_subjects")
    .upsert({ name: subjectName, slug }, { onConflict: "name" })
    .select("id,name,slug")
    .single();

  if (error) throw new Error(`Subject upsert failed for ${subjectName}: ${error.message}`);
  return data;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function flushRows(rows) {
  if (rows.length === 0 || dryRun) return;

  const { error } = await supabase.from("o_level_questions").upsert(rows, { onConflict: "id" });
  if (error) throw new Error(`Question upsert failed: ${error.message}`);
}

async function importSubject(subjectName) {
  const subject = await upsertSubject(subjectName);

  if (replaceExisting && !dryRun) {
    const { error } = await supabase.from("o_level_questions").delete().eq("subject_id", subject.id);
    if (error) throw new Error(`Could not clear existing questions for ${subject.name}: ${error.message}`);
  }

  const subjectDir = path.join(dataRoot, subjectName);
  const mcqDir = path.join(subjectDir, "mcqs_by_year");
  const batchFiles = (await fs.readdir(subjectDir)).filter((file) => /^\d{4}-\d{4}\.json$|^\d{4}-onwards\.json$/.test(file)).sort();
  const mcqFiles = (await fs.readdir(mcqDir)).filter((file) => /^\d{4}\.json$/.test(file)).sort();
  const mcqAnswerLookup = await buildMcqAnswerLookup(mcqDir, mcqFiles);

  let questionCount = 0;
  let imageCount = 0;
  const seenBaseIds = new Set();
  const idCounts = new Map();
  let batch = [];

  for (const file of batchFiles) {
    const batchData = await readJson(path.join(subjectDir, file));
    const questions = flattenBatchQuestions(batchData, subject.name, file);

    for (let index = 0; index < questions.length; index += 1) {
      const raw = questions[index];
      const lookupKey = questionLookupKey(raw);
      const enrichedAnswer = cleanText(raw.correct_option) || mcqAnswerLookup.get(lookupKey)?.correct_option || null;
      const enrichedSource = {
        batch_file: file,
        path: [raw.year, raw.session, raw.paper, raw.variant, raw.question_number].filter(Boolean),
        mcq_source: mcqAnswerLookup.get(lookupKey)?.source || null,
      };
      const question = {
        ...raw,
        subject: cleanText(raw.subject) || subject.name,
        year: Number.parseInt(cleanText(raw.year), 10),
        session: cleanText(raw.session),
        paper: cleanText(raw.paper),
        variant: cleanText(raw.variant),
        question_number: cleanText(raw.question_number) || String(index + 1),
        sub_question: cleanText(raw.sub_question),
        correct_option: enrichedAnswer,
      };

      const questionText = cleanText(raw.question_text) || "";
      const parsed = parseQuestionText(questionText, raw.options);
      const images = await normalizeImages(question, raw.images);
      imageCount += images.length;
      const baseId = questionId(subject.slug, question);
      const id = uniqueQuestionId(baseId, idCounts);
      seenBaseIds.add(baseId);

      batch.push({
        id,
        subject_id: subject.id,
        subject: subject.name,
        subject_slug: subject.slug,
        year: question.year,
        session: question.session,
        paper: question.paper,
        variant: question.variant,
        question_number: question.question_number,
        sub_question: question.sub_question,
        marks: Number.isFinite(raw.marks) ? raw.marks : null,
        topic_syllabus: cleanText(raw.topic_syllabus),
        topic_general: cleanText(raw.topic_general),
        question_text: questionText,
        stem: parsed.stem || questionText,
        question_kind: classifyQuestion(question, parsed),
        source_type: "batch",
        batch_file: file,
        options: parsed.options,
        marking_scheme: cleanText(raw.marking_scheme),
        correct_option: enrichedAnswer ? enrichedAnswer.toUpperCase() : null,
        requires_diagram: Boolean(raw.requires_diagram || images.length),
        images,
        syllabus_ref: asJsonObject(raw.syllabus_ref),
        reference: asJsonObject(raw.reference),
        source: enrichedSource,
        raw_question: raw,
      });

      questionCount += 1;

      if (batch.length >= 100) {
        await flushRows(batch);
        batch = [];
      }
    }
  }

  for (const file of mcqFiles) {
    const yearData = await readJson(path.join(mcqDir, file));
    const questions = Array.isArray(yearData.mcqs) ? yearData.mcqs : [];

    for (let index = 0; index < questions.length; index += 1) {
      const raw = questions[index];
      const question = {
        ...raw,
        subject: cleanText(raw.subject) || subject.name,
        year: Number.parseInt(cleanText(raw.year) || yearData.year || file.replace(".json", ""), 10),
        session: cleanText(raw.session),
        paper: cleanText(raw.paper),
        variant: cleanText(raw.variant),
        question_number: cleanText(raw.question_number) || String(index + 1),
        sub_question: cleanText(raw.sub_question),
      };
      const baseId = questionId(subject.slug, question);
      if (seenBaseIds.has(baseId)) continue;
      const id = uniqueQuestionId(baseId, idCounts);

      const questionText = cleanText(raw.question_text) || "";
      const parsed = parseQuestionText(questionText, raw.options);
      const images = await normalizeImages(question, raw.images);
      imageCount += images.length;

      batch.push({
        id,
        subject_id: subject.id,
        subject: subject.name,
        subject_slug: subject.slug,
        year: question.year,
        session: question.session,
        paper: question.paper,
        variant: question.variant,
        question_number: question.question_number,
        sub_question: question.sub_question,
        marks: Number.isFinite(raw.marks) ? raw.marks : null,
        topic_syllabus: cleanText(raw.topic_syllabus),
        topic_general: cleanText(raw.topic_general),
        question_text: questionText,
        stem: parsed.stem || questionText,
        question_kind: "mcq",
        source_type: "mcqs_by_year",
        batch_file: raw.batch || null,
        options: parsed.options,
        marking_scheme: cleanText(raw.marking_scheme),
        correct_option: cleanText(raw.correct_option) ? cleanText(raw.correct_option).toUpperCase() : null,
        requires_diagram: Boolean(raw.requires_diagram || images.length),
        images,
        syllabus_ref: asJsonObject(raw.syllabus_ref),
        reference: asJsonObject(raw.reference),
        source: asJsonObject(raw.source),
        raw_question: raw,
      });

      questionCount += 1;

      if (batch.length >= 100) {
        await flushRows(batch);
        batch = [];
      }
    }
  }

  await flushRows(batch);
  console.log(`${dryRun ? "[dry-run] " : ""}${subject.name}: ${questionCount} questions, ${imageCount} image records`);
}

async function buildMcqAnswerLookup(mcqDir, files) {
  const lookup = new Map();

  for (const file of files) {
    const yearData = await readJson(path.join(mcqDir, file));
    const questions = Array.isArray(yearData.mcqs) ? yearData.mcqs : [];

    for (const raw of questions) {
      const question = {
        year: cleanText(raw.year) || yearData.year || file.replace(".json", ""),
        session: cleanText(raw.session),
        paper: cleanText(raw.paper),
        variant: cleanText(raw.variant),
        question_number: cleanText(raw.question_number),
        sub_question: cleanText(raw.sub_question),
      };
      const correctOption = cleanText(raw.correct_option);
      if (!correctOption) continue;
      lookup.set(questionLookupKey(question), {
        correct_option: correctOption.toUpperCase(),
        source: asJsonObject(raw.source),
      });
    }
  }

  return lookup;
}

function flattenBatchQuestions(batchData, subjectName, batchFile) {
  const rows = [];

  for (const year of Object.keys(batchData).sort()) {
    const sessions = batchData[year] || {};
    for (const session of Object.keys(sessions).sort()) {
      const papers = sessions[session] || {};
      for (const paper of Object.keys(papers).sort((a, b) => paperRank(a) - paperRank(b) || a.localeCompare(b))) {
        const variants = papers[paper] || {};
        for (const variant of Object.keys(variants).sort()) {
          const questions = Array.isArray(variants[variant]) ? variants[variant] : [];
          const sortedQuestions = [...questions].sort((a, b) => questionSortKey(a).localeCompare(questionSortKey(b), undefined, { numeric: true }));

          for (const question of sortedQuestions) {
            rows.push({
              ...question,
              subject: subjectName,
              batch: batchFile.replace(".json", ""),
              year,
              session,
              paper,
              variant,
            });
          }
        }
      }
    }
  }

  return rows;
}

async function main() {
  const entries = await fs.readdir(dataRoot, { withFileTypes: true });
  const subjects = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    if (onlySubject && entry.name.toLowerCase() !== onlySubject.toLowerCase()) continue;

    try {
      await fs.access(path.join(dataRoot, entry.name, "mcqs_by_year"));
      subjects.push(entry.name);
    } catch {
      // Skip folders that do not contain year-wise MCQ JSONs.
    }
  }

  if (subjects.length === 0) throw new Error(`No subjects found in ${dataRoot}`);

  console.log(`Importing ${subjects.length} subject(s) from ${dataRoot}`);
  console.log(`Image mode: ${imageMode}${imageMode === "storage" ? `, bucket: ${BUCKET}` : ""}`);

  for (const subject of subjects.sort()) {
    await importSubject(subject);
  }

  console.log("O Level import complete.");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
