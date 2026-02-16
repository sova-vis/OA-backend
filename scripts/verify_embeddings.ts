import dotenv from "dotenv";
import path from "path";
dotenv.config({ path: path.resolve(__dirname, "..", ".env") });

import { createClient } from "@supabase/supabase-js";

const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

async function verify() {
  try {
    // Count chunks
    const { count: chunkCount, error: chunkErr } = await supabase
      .from("rag_chunks")
      .select("*", { count: "exact", head: true });

    if (chunkErr) throw chunkErr;

    // Count embeddings
    const { count: embedCount, error: embedErr } = await supabase
      .from("rag_embeddings")
      .select("*", { count: "exact", head: true });

    if (embedErr) throw embedErr;

    // Get sample chunk
    const { data: sampleChunk } = await supabase
      .from("rag_chunks")
      .select("id, content, chunk_index")
      .limit(1);

    // Get model stats
    const { data: modelStats } = await supabase
      .from("rag_embeddings")
      .select("model")
      .limit(1);

    console.log("\n✅ EMBEDDING VERIFICATION STATUS\n");
    console.log(`📊 Total chunks: ${chunkCount || 0}`);
    console.log(`📊 Total embeddings: ${embedCount || 0}`);
    console.log(
      `\n✨ Embedding Model: ${modelStats?.[0]?.model || "Unknown"}`
    );

    if (sampleChunk && sampleChunk.length > 0) {
      console.log(
        `\n📄 Sample chunk (first 100 chars): "${sampleChunk[0].content.substring(0, 100)}..."`
      );
    }

    if (chunkCount && chunkCount > 0) {
      console.log(`\n✅ SUCCESS! Data is saved and ready for search.\n`);
    } else {
      console.log(`\n❌ No chunks found. Try running ingest again.\n`);
    }
  } catch (error) {
    console.error("Verification error:", error);
    process.exit(1);
  }
}

verify();
