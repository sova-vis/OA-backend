import dotenv from "dotenv";
import path from "path";
dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

import crypto from "crypto";
// pdf-parse v2+ exposes a PDFParse class, not a function.
const { PDFParse } = require("pdf-parse") as { PDFParse: new (options: { data: Buffer }) => { getText: () => Promise<{ text?: string }>; destroy: () => Promise<void> } };
import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const OLLAMA_URL = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/$/, "");
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "bge-m3";

const PAPER_FILE_ID = (process.env.PAPER_FILE_ID || "").trim(); // optional

// Batch controls
const BATCH_FILES = Number(process.env.BATCH_FILES || 5);
const FILES_OFFSET = Number(process.env.FILES_OFFSET || 0);
const MAX_FILES = Number(process.env.MAX_FILES || 999999);

const CHUNK_SIZE = Number(process.env.CHUNK_SIZE || 2500);
const CHUNK_OVERLAP = Number(process.env.CHUNK_OVERLAP || 250);
const MAX_CHUNKS_PER_FILE = Number(process.env.MAX_CHUNKS_PER_FILE || 120);

const EMBED_CONCURRENCY = Math.max(1, Number(process.env.EMBED_CONCURRENCY || 2));

// Optional filters
const FILTER_LEVEL = (process.env.FILTER_LEVEL || "").trim();          // "O" or "A"
const FILTER_FILE_TYPE = (process.env.FILTER_FILE_TYPE || "").trim();  // "QP"/"MS"/"ER"/"GT"
const FILTER_SUBJECT_CODE = (process.env.FILTER_SUBJECT_CODE || "").trim();
const FILTER_YEAR = (process.env.FILTER_YEAR || "").trim();

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error("Missing env: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY");
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

// ------------------------
// Helpers
// ------------------------
function sha256(s: string): string {
  return crypto.createHash("sha256").update(s).digest("hex");
}

function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  const clean = text
    .replace(/\r/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const chunks: string[] = [];
  let i = 0;

  while (i < clean.length) {
    const end = Math.min(clean.length, i + chunkSize);
    const chunk = clean.slice(i, end).trim();
    if (chunk.length > 80) chunks.push(chunk);
    if (end >= clean.length) break;
    i = Math.max(0, end - overlap);
  }

  return chunks;
}

async function embedWithOllama(text: string): Promise<number[]> {
  const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: OLLAMA_MODEL, prompt: text }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Ollama embeddings failed: ${res.status} ${body}`);
  }

  const data = (await res.json()) as any;
  if (!data?.embedding || !Array.isArray(data.embedding)) {
    throw new Error(`Unexpected Ollama response: ${JSON.stringify(data).slice(0, 300)}`);
  }
  return data.embedding as number[];
}

// small concurrency pool for embeddings
async function runPool<T, R>(
  items: T[],
  concurrency: number,
  worker: (item: T) => Promise<R>
): Promise<R[]> {
  const results: R[] = [];
  let idx = 0;

  async function runner() {
    while (idx < items.length) {
      const my = idx++;
      results[my] = await worker(items[my]);
    }
  }

  const workers = Array.from({ length: concurrency }, () => runner());
  await Promise.all(workers);
  return results;
}

// ------------------------
// DB fetch: paper_files
// ------------------------
async function fetchPaperFiles(offset: number, limit: number) {
  // We want to support filters that may require joining papers->subjects.
  // We'll do it in two steps:
  // 1) fetch paper_files with paper_id, file_type, storage_path
  // 2) if filters need joins, we enrich via papers+subjects.

  // If PAPER_FILE_ID provided, just fetch that one.
  if (PAPER_FILE_ID) {
    const { data, error } = await supabase
      .from("paper_files")
      .select("id, paper_id, file_type, storage_path")
      .eq("id", PAPER_FILE_ID)
      .limit(1);

    if (error) throw new Error(error.message);
    return data || [];
  }

  // Basic page fetch
  const { data: pf, error: pfErr } = await supabase
    .from("paper_files")
    .select("id, paper_id, file_type, storage_path")
    .order("id", { ascending: true })
    .range(offset, offset + limit - 1);

  if (pfErr) throw new Error(pfErr.message);
  const paperFiles = pf || [];

  // If no extra filters, return now
  const needsJoin = Boolean(FILTER_LEVEL || FILTER_SUBJECT_CODE || FILTER_YEAR);
  if (!needsJoin) {
    // file_type filter can be applied here if needed
    return paperFiles.filter((r: any) => (FILTER_FILE_TYPE ? r.file_type === FILTER_FILE_TYPE : true));
  }

  // Enrich paper_ids
  const paperIds = [...new Set(paperFiles.map((r: any) => r.paper_id).filter(Boolean))];
  if (paperIds.length === 0) return [];

  const { data: papers, error: pErr } = await supabase
    .from("papers")
    .select("id, subject_id, year, session, paper, variant")
    .in("id", paperIds);

  if (pErr) throw new Error(pErr.message);

  const subjectIds = [...new Set((papers || []).map((p: any) => p.subject_id).filter(Boolean))];
  const { data: subjects, error: sErr } = await supabase
    .from("subjects")
    .select("id, level, code")
    .in("id", subjectIds);

  if (sErr) throw new Error(sErr.message);

  const paperMap = new Map((papers || []).map((p: any) => [p.id, p]));
  const subjMap = new Map((subjects || []).map((s: any) => [s.id, s]));

  const filtered = paperFiles.filter((pf: any) => {
    if (FILTER_FILE_TYPE && pf.file_type !== FILTER_FILE_TYPE) return false;

    const p = paperMap.get(pf.paper_id);
    if (!p) return false;

    if (FILTER_YEAR && String(p.year) !== String(FILTER_YEAR)) return false;

    const s = subjMap.get(p.subject_id);
    if (!s) return false;

    if (FILTER_LEVEL && String(s.level) !== String(FILTER_LEVEL)) return false;
    if (FILTER_SUBJECT_CODE && String(s.code) !== String(FILTER_SUBJECT_CODE)) return false;

    return true;
  });

  return filtered;
}

// ------------------------
// Ingest one PDF
// ------------------------
async function ingestOne(pf: any) {
  const id = pf.id as string;
  const storagePath = pf.storage_path as string;

  console.log(`\n--- Ingest ${id} (${pf.file_type || "?"})`);
  console.log(`Path: ${storagePath}`);

  // download
  const { data: blob, error: dlErr } = await supabase.storage.from("content").download(storagePath);
  if (dlErr || !blob) {
    console.warn(`Download failed: ${dlErr?.message}`);
    return { chunksMade: 0, embedsMade: 0 };
  }

  // extract text
  const buffer = Buffer.from(await blob.arrayBuffer());
  const parser = new PDFParse({ data: buffer });
  const parsed = await parser.getText();
  const text = (parsed?.text || "").trim();
  await parser.destroy();

  console.log(`Extracted chars: ${text.length}`);
  if (text.length < 200) {
    console.warn("Too little text (scanned?) -> skipping for now");
    return { chunksMade: 0, embedsMade: 0 };
  }

  // chunk
  let chunks = chunkText(text, CHUNK_SIZE, CHUNK_OVERLAP);
  if (chunks.length > MAX_CHUNKS_PER_FILE) chunks = chunks.slice(0, MAX_CHUNKS_PER_FILE);
  console.log(`Chunks: ${chunks.length}`);

  // upsert chunks
  const chunkRows = chunks.map((content, idx) => ({
    paper_file_id: id,                 // adjust if your column is named paper_file_id
    chunk_index: idx,
    content,
    content_hash: sha256(`${id}:${idx}:${content}`),
    embedding_status: "pending",
  }));

  const { error: upErr } = await supabase
    .from("rag_chunks")
    .upsert(chunkRows, { onConflict: "content_hash" });

  if (upErr) {
    console.warn(`Upsert rag_chunks failed: ${upErr.message}`);
    return { chunksMade: 0, embedsMade: 0 };
  }

  // load chunks (get IDs)
  const { data: dbChunks, error: chErr } = await supabase
    .from("rag_chunks")
    .select("id, chunk_index, content, embedding_status")
    .eq("paper_file_id", id)
    .order("chunk_index", { ascending: true });

  if (chErr || !dbChunks) {
    console.warn(`Read rag_chunks failed: ${chErr?.message}`);
    return { chunksMade: chunks.length, embedsMade: 0 };
  }

  const chunkIds = dbChunks.map((c: any) => c.id);

  // find existing embeddings
  const { data: existingEmbeds, error: exErr } = await supabase
    .from("rag_embeddings")
    .select("chunk_id")
    .in("chunk_id", chunkIds);

  if (exErr) {
    console.warn(`Read rag_embeddings failed: ${exErr.message}`);
    return { chunksMade: chunks.length, embedsMade: 0 };
  }

  const existingSet = new Set((existingEmbeds || []).map((r: any) => r.chunk_id));
  const toEmbed = (dbChunks as any[]).filter((c) => !existingSet.has(c.id));

  console.log(`Need embeddings: ${toEmbed.length}`);

  // embed with small concurrency
  let embedsMade = 0;

  await runPool(toEmbed, EMBED_CONCURRENCY, async (c: any) => {
    const vec = await embedWithOllama(c.content);

    // insert embedding (include NOT NULL columns)
    const { error: insErr } = await supabase.from("rag_embeddings").insert({
      chunk_id: c.id,
      embedding: vec,
      model: OLLAMA_MODEL,
    });

    if (insErr) {
      console.warn(`Insert embedding failed chunk=${c.chunk_index}: ${insErr.message}`);
      return;
    }

    // mark status done (optional)
    await supabase
      .from("rag_chunks")
      .update({ embedding_status: "done" })
      .eq("id", c.id);

    embedsMade++;
    if (embedsMade % 5 === 0) console.log(`Embedded ${embedsMade}/${toEmbed.length}...`);
  });

  return { chunksMade: chunks.length, embedsMade };
}

// ------------------------
// Main loop
// ------------------------
async function main() {
  let offset = FILES_OFFSET;
  let processed = 0;
  let totalEmbeds = 0;

  const t0 = Date.now();

  while (processed < MAX_FILES) {
    const files = await fetchPaperFiles(offset, BATCH_FILES);
    if (!files || files.length === 0) break;

    for (const pf of files) {
      const res = await ingestOne(pf);
      processed++;
      totalEmbeds += res.embedsMade;
    }

    // If single PAPER_FILE_ID run, stop after one
    if (PAPER_FILE_ID) break;

    offset += BATCH_FILES;

    const secs = (Date.now() - t0) / 1000;
    console.log(`\nProgress: processed=${processed}, next FILES_OFFSET=${offset}, embedsMade=${totalEmbeds}, elapsed=${secs.toFixed(1)}s`);

    // safety: stop after one batch per run (recommended)
    break;
  }

  console.log(`\n✅ Finished this run.`);
  console.log(`Processed files: ${processed}`);
  console.log(`Embeddings made: ${totalEmbeds}`);
  console.log(`Next: set FILES_OFFSET=${offset} to continue (if not using PAPER_FILE_ID).`);
}

main().catch((e) => {
  console.error("ERROR:", e?.message || e);
  process.exit(1);
});
