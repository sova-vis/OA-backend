/**
 * Query embedding for Ask-AI, in-process (no Python service).
 *
 * Uses transformers.js with the SAME model the index was built with
 * (BAAI/bge-base-en-v1.5, 768-dim). bge's retrieval convention: QUERIES get the
 * instruction prefix; the corpus passages (already embedded in ask_ai_chunks) do
 * not. Output is a pgvector text literal '[...]' the match RPC casts to vector.
 */
import { pipeline, env } from '@xenova/transformers';

const MODEL = 'Xenova/bge-base-en-v1.5';
const QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: ';

env.allowRemoteModels = true;
if (process.env.TRANSFORMERS_CACHE) {
  // Persist the model download across warm requests / restarts where possible.
  env.cacheDir = process.env.TRANSFORMERS_CACHE;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pipePromise: Promise<any> | null = null;

function getPipe(): Promise<any> {
  if (!pipePromise) {
    // quantized:false -> fp32 weights, matching the sentence-transformers build
    // used to embed the corpus (a quantized query embedding drifts from it).
    pipePromise = pipeline('feature-extraction', MODEL, { quantized: false });
  }
  return pipePromise;
}

/** Embed one query -> pgvector text literal '[v1,v2,...]' (768-dim, normalized). */
export async function embedQuery(query: string): Promise<string> {
  const pipe = await getPipe();
  const out = await pipe(QUERY_INSTRUCTION + query, { pooling: 'mean', normalize: true });
  const arr: number[] = Array.from(out.data as Float32Array);
  return '[' + arr.map((x) => Number(x).toFixed(6)).join(',') + ']';
}

/** Warm the model at boot so the first student request isn't slow. */
export async function warmupEmbedder(): Promise<void> {
  try {
    await embedQuery('warmup');
    console.log('Ask-AI embedder ready (bge-base-en-v1.5).');
  } catch (e) {
    console.warn('Ask-AI embedder warmup failed:', e instanceof Error ? e.message : e);
  }
}
