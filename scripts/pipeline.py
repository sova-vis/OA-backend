import json
import hashlib
import time
import os
import cohere
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

COHERE_API_KEY  = os.getenv("COHERE_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
JSON_DIR        = os.getenv("JSON_DIR")

EMBEDDING_MODEL = "embed-multilingual-v3.0"
SKIP_SUBJECTS = set()
MAX_TOKENS      = 400   # Cohere limit is 512, stay safely under
MAX_CHARS       = MAX_TOKENS * 4

# ── VALIDATE ENV ──────────────────────────────────────────
missing = [k for k, v in {
    "COHERE_API_KEY": COHERE_API_KEY,
    "SUPABASE_URL":   SUPABASE_URL,
    "SUPABASE_KEY":   SUPABASE_KEY,
    "JSON_DIR":       JSON_DIR,
}.items() if not v]

if missing:
    raise ValueError(f"Missing env variables: {', '.join(missing)}")

co       = cohere.Client(COHERE_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── FORMAT ENTRY — returns one or two chunks ──────────────
def entry_to_chunks(entry, subject, year, session, paper, variant, idx):
    """
    Returns a list of chunk dicts.
    If the full text fits in 400 tokens → 1 chunk.
    If too long → splits into question chunk + mark scheme chunk.
    Both chunks contain full context (subject/year/topic) so
    each is independently retrievable.
    """
    header = "\n".join(filter(None, [
        f"Subject: {subject}",
        f"Year: {year} | Session: {session} | Paper: {paper} | Variant: {variant}",
        f"Topic: {entry.get('topic_general','')} > {entry.get('topic_syllabus','')}",
        f"Question {entry.get('question_number','')} {entry.get('sub_question') or ''} [{entry.get('marks','')} marks]",
    ]))

    question_text = str(entry.get("question_text", "")).strip()
    marking_text  = str(entry.get("marking_scheme", "")).strip()

    full_content = f"{header}\nQuestion: {question_text}\nMark Scheme: {marking_text}"

    # If fits in one chunk — return single chunk
    if len(full_content) <= MAX_CHARS:
        return [{
            "content":       full_content,
            "content_hash":  hashlib.sha256(full_content.encode()).hexdigest(),
            "chunk_index":   idx * 2,
        }]

    # Too long — split into two chunks, both have full header + question context
    chunk_q = f"{header}\nQuestion: {question_text}"
    chunk_ms = f"{header}\nMark Scheme: {marking_text}"

    # If mark scheme itself is too long, truncate it
    if len(chunk_ms) > MAX_CHARS:
        overflow  = len(chunk_ms) - MAX_CHARS
        chunk_ms  = chunk_ms[:MAX_CHARS - 20] + "... [truncated]"

    return [
        {
            "content":      chunk_q,
            "content_hash": hashlib.sha256(chunk_q.encode()).hexdigest(),
            "chunk_index":  idx * 2,
        },
        {
            "content":      chunk_ms,
            "content_hash": hashlib.sha256(chunk_ms.encode()).hexdigest(),
            "chunk_index":  idx * 2 + 1,
        }
    ]

def make_hash(content):
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# ── METADATA ──────────────────────────────────────────────
def get_or_create_meta_id(subject, year, session, paper, variant):
    label = f"{subject}|{year}|{session}|{paper}|{variant}"
    result = supabase.table("past_paper_meta").select("id").eq("label", label).execute()
    if result.data:
        return result.data[0]["id"]
    insert = supabase.table("past_paper_meta").insert({
        "subject": subject,
        "year":    int(year),
        "session": session,
        "paper":   paper,
        "variant": variant,
        "label":   label,
    }).execute()
    return insert.data[0]["id"]

# ── EMBED VIA COHERE ──────────────────────────────────────
def embed_batch(texts, input_type="search_document"):
    for attempt in range(5):
        try:
            response = co.embed(
                texts=texts,
                model=EMBEDDING_MODEL,
                input_type=input_type,
            )
            return response.embeddings
        except cohere.TooManyRequestsError:
            wait = 30 * (attempt + 1)
            print(f"      ⏳ Rate limited — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"      ⚠️  Embed error attempt {attempt+1}/5: {e} — waiting {wait}s")
            time.sleep(wait)
    return None

# ── PROCESS ONE JSON FILE ─────────────────────────────────
def process_json_file(json_path: Path, grand_start: float):
    subject   = json_path.parent.name
    range_key = json_path.stem

    print(f"\n{'='*60}")
    print(f"📂 {subject} / {range_key}")
    print(f"{'='*60}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    contents   = []
    chunk_rows = []
    scan_count = 0
    split_count = 0

    print("   🔍 Scanning entries...")

    for year, sessions in data.items():
        for session, papers in sessions.items():
            for paper, variants in papers.items():
                for variant, entries in variants.items():
                    if not isinstance(entries, list):
                        continue

                    meta_id = get_or_create_meta_id(
                        subject, year, session, paper, variant
                    )

                    for idx, entry in enumerate(entries):
                        scan_count += 1
                        chunks = entry_to_chunks(
                            entry, subject, year, session, paper, variant, idx
                        )

                        if len(chunks) > 1:
                            split_count += 1

                        for chunk in chunks:
                            existing = supabase.table("rag_chunks") \
                                .select("id").eq("content_hash", chunk["content_hash"]).execute()
                            if existing.data:
                                continue

                            contents.append(chunk["content"])
                            chunk_rows.append({
                                "past_paper_id":    meta_id,
                                "syllabus_file_id": None,
                                "chunk_index":      int(chunk["chunk_index"]),
                                "content":          chunk["content"],
                                "content_hash":     chunk["content_hash"],
                                "embedding_status": "pending",
                            })

                        if scan_count % 50 == 0:
                            print(f"   🔍 Scanned {scan_count} entries — {len(chunk_rows)} chunks ({split_count} split)...")

    print(f"   🔍 Done — {scan_count} entries → {len(chunk_rows)} chunks ({split_count} split into 2)")

    if not chunk_rows:
        print("   ✅ Already up to date — nothing new")
        return 0

    print(f"   📊 {len(chunk_rows)} new chunks to process")

    # ── INSERT CHUNKS IN BATCHES ──────────────────────────
    inserted_ids = []
    BATCH = 50
    for i in range(0, len(chunk_rows), BATCH):
        batch = chunk_rows[i:i+BATCH]
        try:
            result = supabase.table("rag_chunks").insert(batch).execute()
            inserted_ids.extend([r["id"] for r in result.data])
        except Exception:
            print(f"   ⚠️  Batch had duplicate — inserting one by one...")
            for row in batch:
                try:
                    result = supabase.table("rag_chunks").insert(row).execute()
                    inserted_ids.extend([r["id"] for r in result.data])
                except Exception:
                    existing = supabase.table("rag_chunks") \
                        .select("id").eq("content_hash", row["content_hash"]).execute()
                    if existing.data:
                        inserted_ids.append(existing.data[0]["id"])
        print(f"   💾 Chunks {i+1}–{min(i+BATCH, len(chunk_rows))} / {len(chunk_rows)} inserted")

    if len(inserted_ids) != len(contents):
        synced = min(len(inserted_ids), len(contents))
        print(
            f"   ⚠️  Insert/embed mismatch: {len(inserted_ids)} IDs for {len(contents)} contents — continuing with {synced}"
        )
        inserted_ids = inserted_ids[:synced]
        contents = contents[:synced]

    # ── EMBED IN BATCHES OF 96 (Cohere max) ──────────────
    print(f"\n   🤖 Generating embeddings for {len(contents)} chunks...")
    success    = 0
    fail       = 0
    file_start = time.time()
    EMB_BATCH  = 96

    for i in range(0, len(contents), EMB_BATCH):
        text_batch = contents[i:i+EMB_BATCH]
        id_batch   = inserted_ids[i:i+EMB_BATCH]

        vectors = embed_batch(text_batch)

        if vectors is None:
            print(f"   ❌ Batch {i}–{i+EMB_BATCH} failed — skipping")
            fail += len(text_batch)
            continue

        for chunk_id, vector in zip(id_batch, vectors):
            # Skip if embedding already exists
            existing_emb = supabase.table("rag_embeddings") \
                .select("chunk_id").eq("chunk_id", chunk_id).execute()
            if existing_emb.data:
                success += 1
                continue

            supabase.table("rag_embeddings").insert({
                "chunk_id":  chunk_id,
                "embedding": vector,
                "model":     EMBEDDING_MODEL,
            }).execute()

            supabase.table("rag_chunks") \
                .update({"embedding_status": "embedded"}) \
                .eq("id", chunk_id).execute()

            success += 1

        elapsed       = round(time.time() - file_start, 1)
        total_elapsed = round(time.time() - grand_start, 1)
        rate          = round(success / elapsed * 60, 1) if elapsed > 0 else 0
        remaining     = len(contents) - success
        eta_mins      = round(remaining / rate, 1) if rate > 0 else "?"
        print(f"   ✅ {success}/{len(contents)} embedded | "
              f"{rate}/min | "
              f"ETA: ~{eta_mins} mins | "
              f"Total elapsed: {total_elapsed}s")

        time.sleep(1)  # Cohere rate limit buffer

    file_elapsed = round(time.time() - file_start, 1)
    print(f"\n   🎉 {subject}/{range_key} done — "
          f"{success} embedded, {fail} failed, {file_elapsed}s")
    return success

# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    grand_start = time.time()
    base        = Path(JSON_DIR)

    if not base.exists():
        raise FileNotFoundError(f"JSON_DIR does not exist: {JSON_DIR}")

    json_files = sorted([
        jf for jf in base.rglob("*.json")
        if jf.parent.name not in SKIP_SUBJECTS
    ])

    print("🚀 Starting RAG pipeline")
    print(f"📁 JSON dir: {JSON_DIR}")
    print(f"📄 Found {len(json_files)} JSON files")
    print(f"⏭️  Skipping subjects: {SKIP_SUBJECTS}")
    print(f"{'='*60}\n")

    total_embedded = 0

    for jf in json_files:
        try:
            n = process_json_file(jf, grand_start)
            total_embedded += n
        except Exception as e:
            print(f"❌ Failed {jf}: {repr(e)}")
            continue

    total_elapsed = round(time.time() - grand_start, 1)
    print(f"\n{'='*60}")
    print("🎉 ALL DONE")
    print(f"📊 Total embedded: {total_embedded}")
    print(f"⏱️  Total time: {total_elapsed}s ({round(total_elapsed/60, 1)} mins)")
    print(f"{'='*60}")


