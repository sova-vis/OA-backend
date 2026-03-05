import { Router, Request, Response } from "express";
import { supabase } from "./lib/supabase";
import Groq from "groq-sdk";

const router = Router();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

function parseBooleanEnv(value: string | undefined, defaultValue: boolean): boolean {
  if (!value || !value.trim()) return defaultValue;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function parseNumberEnv(value: string | undefined, defaultValue: number): number {
  if (!value || !value.trim()) return defaultValue;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

const OLLAMA_URL = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/$/, "");
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "bge-m3";
const HUGGINGFACE_API_KEY = (process.env.HUGGINGFACE_API_KEY || "").trim();
const HUGGINGFACE_EMBEDDING_MODEL = (process.env.HUGGINGFACE_EMBEDDING_MODEL || "BAAI/bge-m3").trim();
const HUGGINGFACE_ROUTER_BASE_URL = (process.env.HUGGINGFACE_ROUTER_BASE_URL || "https://router.huggingface.co").replace(/\/$/, "");
const JINA_API_KEY = (process.env.JINA_API_KEY || "").trim();
const SUPABASE_STORAGE_BUCKET = (process.env.SUPABASE_STORAGE_BUCKET || "content").trim() || "content";
const INCLUDE_STUDENT_RETRIEVAL_DIAGNOSTICS = parseBooleanEnv(
  process.env.INCLUDE_STUDENT_RETRIEVAL_DIAGNOSTICS,
  false
);
const ALLOW_OLLAMA_FALLBACK = parseBooleanEnv(
  process.env.ALLOW_OLLAMA_FALLBACK,
  process.env.NODE_ENV !== "production"
);

const SIMILARITY_THRESHOLD = 0.40;
const TOP_K = 16;
const TOP_K_WITH_FILTERS = Math.max(
  TOP_K,
  Math.floor(parseNumberEnv(process.env.TOP_K_WITH_FILTERS, 96))
);
const NEARBY_REFERENCE_LIMIT = Math.max(
  3,
  Math.floor(parseNumberEnv(process.env.NEARBY_REFERENCE_LIMIT, 4))
);
const MAX_CHUNKS_PER_FILE = 4;
const MIN_BEST_SIMILARITY_FOR_CONTEXT = parseNumberEnv(process.env.MIN_BEST_SIMILARITY_FOR_CONTEXT, 0.55);
const MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT = parseNumberEnv(process.env.MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT, 0.48);
const MIN_BEST_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT = parseNumberEnv(
  process.env.MIN_BEST_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT,
  0.40
);
const MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT = parseNumberEnv(
  process.env.MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT,
  0.34
);

// ============================================================================
// SUBJECT CACHE — fetched from DB dynamically, resolved against known names
// ============================================================================

interface CachedSubject {
  id: string;
  code: string;
  level: string;
  name: string; // resolved display name
}

let subjectCache: CachedSubject[] | null = null;

// Known CAIE subject code → name fallbacks.
// Custom/internal codes should come from DB subject name or SUBJECT_CODE_NAMES.
const KNOWN_CODE_NAMES: Record<string, string> = {
  // Standard CAIE O-Level
  "0580": "Mathematics D", "0606": "Additional Mathematics",
  "0620": "Chemistry", "0625": "Physics", "0610": "Biology",
  "0417": "Computer Studies", "0450": "Business Studies",
  "0470": "History", "0460": "Geography",
  "0500": "English Language", "1123": "English Language",
  "2058": "Islamiyat", "2059": "Pakistan Studies",
  "2210": "Computer Science", "4024": "Mathematics Syllabus D",
  "4037": "Additional Mathematics", "5054": "Physics",
  "5070": "Chemistry", "5090": "Biology", "3247": "Urdu",
  "3248": "Art & Design", "2134": "History", "2217": "Geography",
  "2281": "Economics", "7100": "Commerce", "7707": "Accounting",
  "2086": "Religious Studies", "3260": "Sociology",
};

const SUBJECT_KEYWORDS = [
  "computer science", "english language", "english literature",
  "pakistan studies", "pakistan study", "pak studies",
  "chemistry", "physics", "biology", "mathematics", "maths", "math",
  "islamiyat", "islamiat", "islamic studies", "urdu", "english", "computer", "economics",
  "history", "geography", "accounting", "commerce", "sociology",
  "additional mathematics", "add maths",
].sort((a, b) => b.length - a.length);

const SUBJECT_ALIAS_TO_CANONICAL: Record<string, string> = {
  "math": "mathematics",
  "maths": "mathematics",
  "pakistan study": "pakistan studies",
  "pak studies": "pakistan studies",
  "add maths": "additional mathematics",
  "islamiat": "islamiyat",
  "islamic studies": "islamiyat",
  "computer": "computer science",
};

const TOPIC_SUBJECT_HINTS: Array<{ pattern: RegExp; subjectHint: string }> = [
  { pattern: /\b(vector|vectors|scalar|scalars|momentum|force|displacement|velocity|acceleration|newton|resultant)\b/i, subjectHint: "physics" },
  { pattern: /\b(logarithm|logarithms|integration|differentiat|derivative|gradient|trigonometry|algebra|matrix|matrices|determinant|transpose|inverse matrix|simultaneous equations)\b/i, subjectHint: "mathematics" },
  { pattern: /\b(mole|moles|stoichiometry|acid|alkali|electrolysis|oxidation|reduction|salt)\b/i, subjectHint: "chemistry" },
  { pattern: /\b(cell|osmosis|diffusion|photosynthesis|enzyme|mitosis|meiosis|organism)\b/i, subjectHint: "biology" },
  { pattern: /\b(hazrat|sahaba|khulafa|prophet muhammad|quran|qur'an|hadith|sunnah|hijrah|madinah|makkah|abu bakr|umar|umer|uthman|ali)\b/i, subjectHint: "islamiyat" },
  { pattern: /\b(pakistan movement|lahore resolution|two nation theory|muslim league|jinnah|allama iqbal|partition|1947|cabinet mission|simla conference|indus water treaty)\b/i, subjectHint: "pakistan studies" },
];

function normalizeSubjectTerm(value: string): string {
  const normalized = value.toLowerCase().trim().replace(/\s+/g, " ");
  return SUBJECT_ALIAS_TO_CANONICAL[normalized] || normalized;
}

// Custom overrides from env: SUBJECT_CODE_NAMES="1013:Islamiyat,1015:Pakistan Studies,1016:Physics"
function parseCustomSubjectNames(): Record<string, string> {
  const raw = process.env.SUBJECT_CODE_NAMES || "";
  if (!raw.trim()) return {};
  return raw.split(",").reduce((acc: Record<string, string>, entry) => {
    const [code, name] = entry.split(":").map((s) => s.trim());
    if (code && name) acc[code] = name;
    return acc;
  }, {});
}

async function getSubjectCache(): Promise<CachedSubject[]> {
  if (subjectCache) return subjectCache;

  let subjectsData: any[] | null = null;

  const withName = await supabase
    .from("subjects")
    .select("id, code, level, name")
    .order("code");

  if (withName.error) {
    const fallback = await supabase
      .from("subjects")
      .select("id, code, level")
      .order("code");

    if (fallback.error || !fallback.data) {
      console.error("Cannot fetch subjects from DB:", fallback.error?.message || withName.error.message);
      return [];
    }

    subjectsData = fallback.data as any[];
  } else {
    subjectsData = (withName.data || []) as any[];
  }

  const customNames = parseCustomSubjectNames();

  subjectCache = subjectsData.map((s: any) => {
    const dbName = typeof s.name === "string" ? s.name.trim() : "";
    return {
      id: s.id as string,
      code: s.code as string,
      level: s.level as string,
      name: customNames[s.code] || dbName || KNOWN_CODE_NAMES[s.code] || `Subject ${s.code}`,
    };
  });

  console.log("[RAG] Loaded subjects:", subjectCache.map((s) => `${s.code}=${s.name}`).join(", "));
  return subjectCache;
}

// Invalidate cache so next request refetches (useful after admin updates)
function clearSubjectCache() {
  subjectCache = null;
}

function resolveSubjectName(code: string): string {
  if (subjectCache) {
    const found = subjectCache.find((s) => s.code === code);
    if (found) return found.name;
  }
  const customNames = parseCustomSubjectNames();
  return customNames[code] || KNOWN_CODE_NAMES[code] || `Subject ${code}`;
}

// Find the best matching DB subject for a user-typed name
async function findSubjectByName(name: string): Promise<CachedSubject | null> {
  const subjects = await getSubjectCache();
  if (!subjects.length) return null;

  const q = normalizeSubjectTerm(name);

  // Exact name match
  const exact = subjects.find((s) => normalizeSubjectTerm(s.name) === q);
  if (exact) return exact;

  // Partial name match
  const partial = subjects.find(
    (s) => {
      const subjectName = normalizeSubjectTerm(s.name);
      return subjectName.includes(q) || q.includes(subjectName);
    }
  );
  if (partial) return partial;

  // Code match (user typed the code directly)
  const byCode = subjects.find((s) => s.code.toLowerCase() === q);
  if (byCode) return byCode;

  return null;
}

interface SubjectInference {
  keyword: string;
  matchedSubject?: CachedSubject;
  source: "explicit-keyword" | "topic-hint" | "topic-hint-strong";
}

function isEnglishCompositionPrompt(question: string): boolean {
  const numberedPromptLines = (question.match(/(?:^|\n)\s*\d+\s+/g) || []).length;

  const compositionSignals = [
    /\bwrite a story\b/i,
    /\bdescribe (the|a) scene\b/i,
    /\bincludes the words\b/i,
    /\bgreat disappointment or happiness\b/i,
    /\bopen air\b/i,
    /\bsuccess\.?\s*$/im,
    /\bletter\b/i,
    /\bcomposition\b/i,
    /\bnarrative\b/i,
    /\bcreative writing\b/i,
  ];

  const signalHits = compositionSignals.reduce((count, pattern) => count + (pattern.test(question) ? 1 : 0), 0);
  return signalHits >= 2 || (numberedPromptLines >= 2 && signalHits >= 1);
}

function detectSubjectKeyword(question: string): string | undefined {
  const lower = question.toLowerCase();
  return SUBJECT_KEYWORDS.find((kw) => lower.includes(kw));
}

function isStrongMathematicsPrompt(question: string): boolean {
  return /\b(matrix|matrices|determinant|inverse matrix|matrix multiplication|adjoint|cofactor|simultaneous equations)\b/i.test(question);
}

async function inferSubjectFromQuestion(question: string): Promise<SubjectInference | null> {
  const keyword = detectSubjectKeyword(question);
  if (keyword) {
    const matchedSubject = await findSubjectByName(keyword);
    if (matchedSubject) return { keyword, matchedSubject, source: "explicit-keyword" };

    return { keyword, source: "explicit-keyword" };
  }

  if (isEnglishCompositionPrompt(question)) {
    const matchedSubject = await findSubjectByName("english");
    if (matchedSubject) {
      return {
        keyword: "english",
        matchedSubject,
        source: "topic-hint-strong",
      };
    }

    return {
      keyword: "english",
      source: "topic-hint-strong",
    };
  }

  if (isStrongMathematicsPrompt(question)) {
    const matchedSubject = await findSubjectByName("mathematics");
    if (matchedSubject) {
      return {
        keyword: "mathematics",
        matchedSubject,
        source: "topic-hint-strong",
      };
    }

    return {
      keyword: "mathematics",
      source: "topic-hint-strong",
    };
  }

  for (const hint of TOPIC_SUBJECT_HINTS) {
    if (!hint.pattern.test(question)) continue;

    const matchedSubject = await findSubjectByName(hint.subjectHint);
    if (matchedSubject) {
      return {
        keyword: hint.subjectHint,
        matchedSubject,
        source: "topic-hint",
      };
    }
  }

  return null;
}

// ============================================================================
// TYPES
// ============================================================================

type Intent = "paper_lookup" | "exam_question" | "smalltalk";

interface ClassificationResult {
  intent: Intent;
  metadata?: {
    subjectKeyword?: string;
    year?: number;
    paper?: string;
    fileType?: string;
  };
}

interface GroupedResult {
  paperFileId?: string;
  filetype: string;
  storagePath: string;
  chunks: Array<{ id: string; content: string; chunkIndex: number; similarity: number }>;
  subject: string;
  year: number;
  session: string;
  paper: string;
}

interface RagRetrievalResult {
  success: boolean;
  groupedResults: GroupedResult[];
  rawSimilarityScores: number[];
  nearbyGroupedResults: GroupedResult[];
  stats?: {
    matchCountRequested: number;
    rpcMatchCount: number;
    thresholdMatchCount: number;
    metadataChunkCount: number;
    groupedFileCount: number;
    bestSimilarity: number;
    avgTop3Similarity: number;
    usedThresholdFallback: boolean;
  };
  error?: string;
}

interface RelatedQuestion {
  type: "exact" | "similar";
  text: string;
  source: { subject: string; year: number; session: string; paper: string; file_type: string; paper_file_id?: string; storage_path?: string; paper_view_url?: string };
  similarity: number;
}

interface ExamAnswer {
  answer: string;
  markingPoints: Array<{ point: string; marks: number }>;
  commonMistakes: string[];
  citations: Citation[];
  confidenceScore: number;
  coveragePercentage: number;
  relatedQuestion?: RelatedQuestion;
}

interface HistoryMessage { role: "user" | "assistant"; content: string; }

interface RagQueryRequest {
  question: string;
  limit?: number;
  filters?: { subject?: string; year?: number; file_type?: string; level?: string };
  history?: HistoryMessage[];
}

interface Citation {
  subject: string; subjectName: string; year: number; session: string;
  paper: string; file_type: string; storage_path: string;
  chunk_index: number; similarity: number;
  paper_file_id?: string;
  paper_view_url?: string;
  relation?: "direct" | "nearby";
}

interface RetrievalDiagnostics {
  mode: "grounded" | "nearby" | "general";
  used_embeddings: boolean;
  subject_filter?: string;
  reason?: string;
  match_count_requested?: number;
  rpc_match_count?: number;
  threshold_match_count?: number;
  metadata_chunk_count?: number;
  grouped_file_count?: number;
  best_similarity?: number;
  avg_top3_similarity?: number;
}

interface RagQueryResponse {
  type: "smalltalk" | "paper_lookup" | "exam_question";
  answer: string;
  marking_points?: Array<{ point: string; marks: number }>;
  common_mistakes?: string[];
  citations: Citation[];
  confidence_score?: number;
  coverage_percentage?: number;
  low_confidence?: boolean;
  results?: any[];
  available_subjects?: Array<{ code: string; name: string; level: string }>;
  related_question?: RelatedQuestion;
  retrieval?: RetrievalDiagnostics;
  nearby_references?: Citation[];
}

// ============================================================================
// EMBEDDING — HuggingFace bge-m3 primary, Jina secondary, Ollama optional
// ============================================================================

// Mean-pool a 2D token matrix [seq_len, hidden] → [hidden]
function meanPoolTokens(tokenMatrix: number[][]): number[] {
  if (!tokenMatrix.length) return [];
  const dim = tokenMatrix[0].length;
  const result = new Array(dim).fill(0);
  for (const vec of tokenMatrix) {
    for (let i = 0; i < dim; i++) result[i] += vec[i];
  }
  return result.map((v) => v / tokenMatrix.length);
}

function isNumberVector(value: unknown): value is number[] {
  return Array.isArray(value) && value.length > 0 && value.every((n) => typeof n === "number");
}

function extractEmbeddingFromPayload(payload: any): number[] | null {
  if (isNumberVector(payload)) return payload;

  if (Array.isArray(payload) && payload.length > 0 && Array.isArray(payload[0])) {
    const tokenMatrix = payload.filter((row: unknown) => isNumberVector(row)) as number[][];
    const pooled = meanPoolTokens(tokenMatrix);
    if (pooled.length > 0) return pooled;
  }

  if (isNumberVector(payload?.embedding)) return payload.embedding;
  if (isNumberVector(payload?.data?.[0]?.embedding)) return payload.data[0].embedding;

  return null;
}

function buildHuggingFaceModelPath(model: string): string {
  return model
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

async function embedWithHuggingFace(text: string): Promise<number[] | null> {
  try {
    const modelPath = buildHuggingFaceModelPath(HUGGINGFACE_EMBEDDING_MODEL);
    const endpointCandidates = [
      {
        label: "router/models/pipeline",
        url: `${HUGGINGFACE_ROUTER_BASE_URL}/hf-inference/models/${modelPath}/pipeline/feature-extraction`,
        body: {
          inputs: text,
          options: { wait_for_model: true, use_cache: true },
        },
      },
      {
        label: "router/pipeline",
        url: `${HUGGINGFACE_ROUTER_BASE_URL}/hf-inference/pipeline/feature-extraction/${modelPath}`,
        body: {
          inputs: text,
          options: { wait_for_model: true, use_cache: true },
        },
      },
      {
        label: "router/models",
        url: `${HUGGINGFACE_ROUTER_BASE_URL}/hf-inference/models/${modelPath}`,
        body: {
          inputs: text,
          options: { wait_for_model: true, use_cache: true },
        },
      },
    ];

    for (const endpoint of endpointCandidates) {
      console.log(`[Embeddings] Calling HuggingFace ${HUGGINGFACE_EMBEDDING_MODEL} via ${endpoint.label}...`);

      const res = await fetch(endpoint.url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${HUGGINGFACE_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(endpoint.body),
        signal: AbortSignal.timeout(45000),
      });

      const bodyText = await res.text();

      if (!res.ok) {
        if (endpoint.label === "router/models" && /SentenceSimilarityPipeline/i.test(bodyText)) {
          console.warn("[Embeddings] HuggingFace router/models mapped to sentence-similarity for this model. Skipping endpoint.");
          continue;
        }
        console.warn(`[Embeddings] HuggingFace ${endpoint.label} HTTP ${res.status}: ${bodyText.slice(0, 300)}`);
        continue;
      }

      let payload: any;
      try {
        payload = JSON.parse(bodyText);
      } catch {
        console.warn(`[Embeddings] HuggingFace ${endpoint.label} returned non-JSON response.`);
        continue;
      }

      const embedding = extractEmbeddingFromPayload(payload);
      if (embedding) {
        console.log("[Embeddings] HuggingFace success:", embedding.length, "dims");
        return embedding;
      }

      if (typeof payload?.error === "string") {
        console.warn(`[Embeddings] HuggingFace ${endpoint.label} response error: ${payload.error}`);
      } else {
        console.warn(`[Embeddings] HuggingFace ${endpoint.label} response shape not recognized.`);
      }
    }
  } catch (err) {
    console.warn("[Embeddings] HuggingFace error:", err);
  }

  return null;
}

async function embedWithJina(text: string): Promise<number[] | null> {
  try {
    console.log("[Embeddings] Calling Jina AI...");
    const res = await fetch("https://api.jina.ai/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${JINA_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "jina-embeddings-v3",
        input: [text],
        task: "text-matching",
        dimensions: 1024,
      }),
      signal: AbortSignal.timeout(30000),
    });

    if (!res.ok) {
      const body = await res.text();
      console.warn(`[Embeddings] Jina AI HTTP ${res.status}: ${body}`);
      return null;
    }

    const data = await res.json() as any;
    const embedding = data?.data?.[0]?.embedding as number[];
    if (isNumberVector(embedding)) {
      console.log("[Embeddings] Jina AI success:", embedding.length, "dims");
      return embedding;
    }

    console.warn("[Embeddings] Jina AI response missing embedding vector.");
  } catch (err) {
    console.warn("[Embeddings] Jina AI error:", err);
  }

  return null;
}

async function embedWithOllama(text: string): Promise<number[] | null> {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: OLLAMA_MODEL, prompt: text }),
      signal: AbortSignal.timeout(20000),
    });

    if (!res.ok) {
      const body = await res.text();
      console.warn(`[Embeddings] Ollama HTTP ${res.status}: ${body}`);
      return null;
    }

    const data = await res.json() as any;
    if (isNumberVector(data?.embedding)) {
      console.log("[Embeddings] Ollama success:", data.embedding.length, "dims");
      return data.embedding;
    }

    console.warn("[Embeddings] Ollama response missing embedding vector.");
  } catch (err) {
    console.warn("Ollama embedding error:", err);
  }

  return null;
}

async function getEmbedding(text: string): Promise<number[]> {
  const attempts: string[] = [];

  if (HUGGINGFACE_API_KEY) {
    attempts.push("HuggingFace");
    const hfEmbedding = await embedWithHuggingFace(text);
    if (hfEmbedding) return hfEmbedding;
  }

  if (JINA_API_KEY) {
    attempts.push("Jina");
    const jinaEmbedding = await embedWithJina(text);
    if (jinaEmbedding) return jinaEmbedding;
  }

  if (ALLOW_OLLAMA_FALLBACK) {
    attempts.push("Ollama");
    const ollamaEmbedding = await embedWithOllama(text);
    if (ollamaEmbedding) return ollamaEmbedding;
  } else {
    console.warn("[Embeddings] Ollama fallback disabled (ALLOW_OLLAMA_FALLBACK=false).");
  }

  const configured = [
    HUGGINGFACE_API_KEY ? "HuggingFace" : null,
    JINA_API_KEY ? "Jina" : null,
    ALLOW_OLLAMA_FALLBACK ? "Ollama" : null,
  ].filter(Boolean).join(", ") || "none";

  throw new Error(
    `Embedding service unavailable. Configured providers: ${configured}. Attempted: ${attempts.join(", ") || "none"}.`
  );
}

// ============================================================================
// STAGE A: INTENT CLASSIFICATION
// ============================================================================

function classifyIntent(question: string): ClassificationResult {
  const q = question.toLowerCase().trim();
  const normalized = q
    .replace(/[\u2019']/g, "'")
    .replace(/[^a-z0-9'\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const smalltalkExact = new Set([
    "hello", "hi", "hey", "yo", "thanks", "thank you", "ok", "okay", "yes", "no",
    "sure", "good", "great", "nice", "cool", "bye", "goodbye", "good night", "gn",
    "lol", "haha", "hmm", "umm", "hru", "sup",
    "how are you", "how r u", "how are u", "how you doing", "what's up", "whats up",
    "who are you", "are you there", "good morning", "good afternoon", "good evening",
  ]);
  if (smalltalkExact.has(normalized)) return { intent: "smalltalk" };

  const smalltalkPatterns = [
    /^(hello|hi|hey|thanks|thank you|ok|okay|yes|no|sure|good|great|nice|cool|bye|goodbye)$/,
    /^(how are you|how r u|how are u|how you doing|what'?s up|who are you|are you there|good morning|good afternoon|good evening)$/,
    /^[\?\!\.]{1,3}$/, /^(lol|haha|hmm|umm|err)$/,
  ];
  if (smalltalkPatterns.some((p) => p.test(q))) return { intent: "smalltalk" };

  const shortCasualPattern = /^(hi+|he+y+|ok(ay)?|thanks?|thank you|cool|nice|great|good|bye|goodbye)$/;
  if (normalized.split(" ").length <= 4 && shortCasualPattern.test(normalized)) {
    return { intent: "smalltalk" };
  }

  const fileTypeIndicators = ["qp", "ms", "er", "gt"];
  const yearMatch = q.match(/\b(20\d{2})\b/);
  const paperMatch = q.match(/\b(p1|p2|p3)\b/i);

  let detectedSubjectKeyword: string | undefined;
  for (const kw of SUBJECT_KEYWORDS) {
    if (q.includes(kw)) { detectedSubjectKeyword = kw; break; }
  }

  let detectedFileType: string | undefined;
  for (const ft of fileTypeIndicators) {
    if (q.includes(ft)) { detectedFileType = ft.toUpperCase(); break; }
  }

  const hasLookupKeyword = /\b(paper|past paper|find|show|get|list|download|papers|available)\b/.test(q);

  if (detectedSubjectKeyword && (yearMatch || detectedFileType || paperMatch || hasLookupKeyword)) {
    return {
      intent: "paper_lookup",
      metadata: {
        subjectKeyword: detectedSubjectKeyword,
        year: yearMatch ? parseInt(yearMatch[1]) : undefined,
        paper: paperMatch ? paperMatch[1].toUpperCase() : undefined,
        fileType: detectedFileType,
      },
    };
  }

  // Also treat "show papers for [year]" without subject as lookup
  if (hasLookupKeyword && yearMatch) {
    return { intent: "paper_lookup", metadata: { year: parseInt(yearMatch[1]) } };
  }

  return { intent: "exam_question" };
}

// ============================================================================
// STAGE B: RAG RETRIEVAL
// ============================================================================

function isUsableId(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const trimmed = value.trim();
  return Boolean(trimmed && trimmed !== "undefined" && trimmed !== "null");
}

function normalizeStoragePath(value?: string): string | undefined {
  if (!value || !value.trim()) return undefined;
  let normalized = value.trim().replace(/^\/+/, "");
  const bucketPrefix = `${SUPABASE_STORAGE_BUCKET}/`;
  if (normalized.toLowerCase().startsWith(bucketPrefix.toLowerCase())) {
    normalized = normalized.slice(bucketPrefix.length);
  }
  return normalized || undefined;
}

function getPaperViewUrl(paperFileId?: string, storagePath?: string): string | undefined {
  const normalizedStoragePath = normalizeStoragePath(storagePath);
  const storagePathQuery = normalizedStoragePath
    ? `?storagePath=${encodeURIComponent(normalizedStoragePath)}`
    : "";

  if (isUsableId(paperFileId)) {
    return `/rag/paper-file/${encodeURIComponent(paperFileId.trim())}/view${storagePathQuery}`;
  }

  if (normalizedStoragePath) {
    return `/rag/paper-file/by-path/view?storagePath=${encodeURIComponent(normalizedStoragePath)}`;
  }

  return undefined;
}

function summarizeSimilarityScores(scores: number[]): { best: number; avgTop3: number } {
  const sortedScores = [...scores].sort((a, b) => b - a);
  const best = sortedScores[0] || 0;
  const top3 = sortedScores.slice(0, 3);
  const avgTop3 = top3.length
    ? top3.reduce((sum, score) => sum + score, 0) / top3.length
    : 0;
  return { best, avgTop3 };
}

function groupChunksByPaperFile(chunks: any[], similarityMap: Map<string, number>): GroupedResult[] {
  const grouped = new Map<string, GroupedResult>();
  const sortedChunks = [...chunks].sort(
    (a: any, b: any) => (similarityMap.get(b.id) || 0) - (similarityMap.get(a.id) || 0)
  );

  for (const chunk of sortedChunks as any[]) {
    const fileId = chunk.paper_file_id ? String(chunk.paper_file_id) : undefined;
    const pf = chunk.paper_files;
    const paper = pf?.papers;
    const subject = paper?.subjects;
    const groupingKey = fileId || `path:${pf?.storage_path || chunk.id}`;

    if (!grouped.has(groupingKey)) {
      grouped.set(groupingKey, {
        paperFileId: fileId,
        filetype: pf?.file_type || "Unknown",
        storagePath: pf?.storage_path || "",
        chunks: [],
        subject: subject?.code || "Unknown",
        year: paper?.year || 0,
        session: paper?.session || "Unknown",
        paper: paper?.paper || "P",
      });
    }

    const group = grouped.get(groupingKey)!;
    if (group.chunks.length < MAX_CHUNKS_PER_FILE) {
      group.chunks.push({
        id: chunk.id,
        content: chunk.content,
        chunkIndex: chunk.chunk_index,
        similarity: similarityMap.get(chunk.id) || 0,
      });
    }
  }

  return Array.from(grouped.values());
}

function buildCitationsFromGroups(groups: GroupedResult[], relation: "direct" | "nearby" = "direct"): Citation[] {
  return groups
    .flatMap((group) =>
      group.chunks.map((chunk) => {
        const paperViewUrl = getPaperViewUrl(group.paperFileId, group.storagePath);
        return {
          subject: group.subject,
          subjectName: resolveSubjectName(group.subject),
          year: group.year,
          session: group.session,
          paper: group.paper,
          file_type: group.filetype,
          storage_path: group.storagePath,
          chunk_index: chunk.chunkIndex,
          similarity: chunk.similarity,
          paper_file_id: group.paperFileId,
          paper_view_url: paperViewUrl,
          relation,
        };
      })
    )
    .sort((a, b) => b.similarity - a.similarity);
}

function dedupeCitations(citations: Citation[]): Citation[] {
  const deduped = new Map<string, Citation>();

  for (const citation of citations) {
    const normalizedStoragePath = normalizeStoragePath(citation.storage_path);
    const primaryKey = isUsableId(citation.paper_file_id)
      ? citation.paper_file_id.trim()
      : (normalizedStoragePath || `${citation.subject}-${citation.year}-${citation.session}-${citation.paper}-${citation.file_type}`);
    const dedupeKey = `${primaryKey}:${citation.file_type}`;

    const existing = deduped.get(dedupeKey);
    if (!existing || citation.similarity >= existing.similarity) {
      deduped.set(dedupeKey, citation);
    }
  }

  return Array.from(deduped.values()).sort((a, b) => b.similarity - a.similarity);
}

function hasReliableRagContext(ragResult: RagRetrievalResult, expectedSubjectCode?: string): boolean {
  if (!ragResult.success || ragResult.groupedResults.length === 0) return false;

  if (expectedSubjectCode) {
    const hasUnexpectedSubject = ragResult.groupedResults.some(
      (g) => String(g.subject) !== String(expectedSubjectCode)
    );
    if (hasUnexpectedSubject) return false;
  }

  const { best: bestScore, avgTop3 } = summarizeSimilarityScores(ragResult.rawSimilarityScores);

  const minBest = expectedSubjectCode
    ? MIN_BEST_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT
    : MIN_BEST_SIMILARITY_FOR_CONTEXT;
  const minAvgTop3 = expectedSubjectCode
    ? MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT
    : MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT;

  return bestScore >= minBest && avgTop3 >= minAvgTop3;
}

async function ragRetrieval(question: string, filters?: any): Promise<RagRetrievalResult> {
  try {
    let questionEmbedding: number[];
    try {
      questionEmbedding = await getEmbedding(question);
    } catch (embedErr: any) {
      return {
        success: false,
        groupedResults: [],
        nearbyGroupedResults: [],
        rawSimilarityScores: [],
        error: embedErr.message,
      };
    }

    const hasMetadataFilters = Boolean(
      filters?.subject || filters?.year || filters?.file_type || filters?.level
    );

    const searchParams: Record<string, any> = {
      query_embedding: questionEmbedding,
      match_count: hasMetadataFilters ? TOP_K_WITH_FILTERS : TOP_K,
    };

    console.log(
      `[RAG] Querying rag_search with match_count=${searchParams.match_count}${hasMetadataFilters ? " (expanded for backend metadata filtering)" : ""}`
    );

    const { data: searchResults, error: searchErr } = await supabase.rpc("rag_search", searchParams);

    if (searchErr) {
      console.error("rag_search RPC error:", searchErr);
      return {
        success: false,
        groupedResults: [],
        nearbyGroupedResults: [],
        rawSimilarityScores: [],
        error: searchErr.message,
      };
    }

    if (!searchResults || searchResults.length === 0) {
      return {
        success: false,
        groupedResults: [],
        nearbyGroupedResults: [],
        rawSimilarityScores: [],
        stats: {
          matchCountRequested: searchParams.match_count,
          rpcMatchCount: 0,
          thresholdMatchCount: 0,
          metadataChunkCount: 0,
          groupedFileCount: 0,
          bestSimilarity: 0,
          avgTop3Similarity: 0,
          usedThresholdFallback: false,
        },
        error: "No results found in database",
      };
    }

    const sortedSearchResults = [...(searchResults as any[])].sort(
      (a, b) => (b.similarity as number) - (a.similarity as number)
    );
    console.log(`[RAG] rag_search returned ${sortedSearchResults.length} embedding matches.`);

    const thresholdResults = sortedSearchResults.filter(
      (r) => (r.similarity as number) >= SIMILARITY_THRESHOLD
    );

    const usedThresholdFallback = thresholdResults.length === 0;
    if (usedThresholdFallback) {
      console.warn(
        `[RAG] No matches met similarity threshold ${SIMILARITY_THRESHOLD}. Keeping top nearby candidates for relation hints.`
      );
    }

    const candidateResults = usedThresholdFallback
      ? sortedSearchResults.slice(0, Math.min(sortedSearchResults.length, Math.max(TOP_K, 24)))
      : thresholdResults;

    if (candidateResults.length === 0) {
      return {
        success: false,
        groupedResults: [],
        nearbyGroupedResults: [],
        rawSimilarityScores: [],
        stats: {
          matchCountRequested: searchParams.match_count,
          rpcMatchCount: sortedSearchResults.length,
          thresholdMatchCount: thresholdResults.length,
          metadataChunkCount: 0,
          groupedFileCount: 0,
          bestSimilarity: 0,
          avgTop3Similarity: 0,
          usedThresholdFallback,
        },
        error: "No embedding candidates available after search",
      };
    }

    const candidateScores = candidateResults.map((r: any) => Number(r.similarity) || 0);
    const { best: bestCandidateSimilarity, avgTop3: avgTop3CandidateSimilarity } = summarizeSimilarityScores(candidateScores);
    const chunkIds = Array.from(new Set(candidateResults.map((r: any) => r.chunk_id).filter(Boolean)));

    const { data: enrichedChunks, error: enrichErr } = await supabase
      .from("rag_chunks")
      .select(`
        id, chunk_index, content, paper_file_id,
        paper_files!inner(
          id, file_type, storage_path, paper_id,
          papers!inner(
            id, year, session, paper, subject_id,
            subjects!inner(id, code, level)
          )
        )
      `)
      .in("id", chunkIds);

    if (enrichErr || !enrichedChunks) {
      return {
        success: false,
        groupedResults: [],
        nearbyGroupedResults: [],
        rawSimilarityScores: candidateScores,
        stats: {
          matchCountRequested: searchParams.match_count,
          rpcMatchCount: sortedSearchResults.length,
          thresholdMatchCount: thresholdResults.length,
          metadataChunkCount: 0,
          groupedFileCount: 0,
          bestSimilarity: bestCandidateSimilarity,
          avgTop3Similarity: avgTop3CandidateSimilarity,
          usedThresholdFallback,
        },
        error: "Failed to fetch chunk metadata",
      };
    }

    const similarityMap = new Map<string, number>(
      candidateResults.map((r: any) => [r.chunk_id as string, r.similarity as number])
    );

    const nearbyGroupedResults = groupChunksByPaperFile(enrichedChunks as any[], similarityMap);

    const metadataFilteredChunks = (enrichedChunks as any[]).filter((chunk: any) => {
      const pf = chunk.paper_files;
      const paper = pf?.papers;
      const subject = paper?.subjects;

      if (filters?.subject && String(subject?.code) !== String(filters.subject)) return false;
      if (filters?.year && Number(paper?.year) !== Number(filters.year)) return false;
      if (filters?.file_type && String(pf?.file_type) !== String(filters.file_type)) return false;
      if (filters?.level && String(subject?.level) !== String(filters.level)) return false;

      return true;
    });

    if (metadataFilteredChunks.length === 0) {
      return {
        success: false,
        groupedResults: [],
        nearbyGroupedResults,
        rawSimilarityScores: [],
        stats: {
          matchCountRequested: searchParams.match_count,
          rpcMatchCount: sortedSearchResults.length,
          thresholdMatchCount: thresholdResults.length,
          metadataChunkCount: 0,
          groupedFileCount: 0,
          bestSimilarity: bestCandidateSimilarity,
          avgTop3Similarity: avgTop3CandidateSimilarity,
          usedThresholdFallback,
        },
        error: "No results matched the requested filters",
      };
    }

    console.log(`[RAG] Context chunks after metadata filters: ${metadataFilteredChunks.length}`);

    const rawScores = metadataFilteredChunks
      .map((chunk: any) => similarityMap.get(chunk.id) || 0)
      .filter((score) => score > 0);
    const groupedResults = groupChunksByPaperFile(metadataFilteredChunks as any[], similarityMap);
    const uniqueSubjects = Array.from(new Set(groupedResults.map((g) => g.subject)));
    console.log(`[RAG] Grouped into ${groupedResults.length} files across subjects: ${uniqueSubjects.join(", ")}`);

    const { best: bestFilteredSimilarity, avgTop3: avgTop3FilteredSimilarity } = summarizeSimilarityScores(rawScores);

    return {
      success: true,
      groupedResults,
      nearbyGroupedResults,
      rawSimilarityScores: rawScores,
      stats: {
        matchCountRequested: searchParams.match_count,
        rpcMatchCount: sortedSearchResults.length,
        thresholdMatchCount: thresholdResults.length,
        metadataChunkCount: metadataFilteredChunks.length,
        groupedFileCount: groupedResults.length,
        bestSimilarity: bestFilteredSimilarity,
        avgTop3Similarity: avgTop3FilteredSimilarity,
        usedThresholdFallback,
      },
    };
  } catch (err: any) {
    return {
      success: false,
      groupedResults: [],
      nearbyGroupedResults: [],
      rawSimilarityScores: [],
      error: err?.message || "Retrieval error",
    };
  }
}

// ============================================================================
// STAGE C: LLM ANSWER GENERATION
// ============================================================================

interface AnswerStyle {
  detailLevel: "short" | "standard" | "long";
  format: "plain" | "structured";
}

function inferAnswerStyle(question: string, history: HistoryMessage[] = []): AnswerStyle {
  const combined = `${question}`.toLowerCase();

  const wantsLong = /(long answer|in detail|detailed|elaborate|thorough|comprehensive|step by step|in depth|properly formatted|full answer)/i.test(combined);
  const wantsShort = /(short answer|briefly|in short|concise)/i.test(combined);
  const wantsStructured = /(format|formatted|bullet|points|headings|steps)/i.test(combined) || wantsLong;

  if (wantsShort) return { detailLevel: "short", format: wantsStructured ? "structured" : "plain" };
  if (wantsLong) return { detailLevel: "long", format: "structured" };
  return { detailLevel: "long", format: "structured" };
}

function buildExamAnswerFormatInstruction(answerStyle: AnswerStyle): string {
  if (answerStyle.detailLevel === "long") {
    return `- Write a complete, detailed answer with clear markdown formatting.
- Structure: short intro paragraph, then 3-5 sections with headings and bullet points.
- Include an exam-focused "How this appears in past papers" section.
- End with a clear summary and quick exam tip.
- Minimum length: around 180-260 words unless the user explicitly asked for a short reply.`;
  }

  if (answerStyle.detailLevel === "short") {
    return `- Keep the answer concise (2-4 lines) but still exam-focused.`;
  }

  if (answerStyle.format === "structured") {
    return `- Use a structured format with short heading + bullet points where useful.`;
  }

  return `- Keep the answer clear and exam-focused in 1-2 short paragraphs.`;
}

function buildMaxTokens(answerStyle: AnswerStyle, withContext: boolean): number {
  if (answerStyle.detailLevel === "long") return withContext ? 2600 : 2200;
  if (answerStyle.detailLevel === "short") return withContext ? 900 : 800;
  return withContext ? 1600 : 1300;
}

function buildAnswerMarkdownHeadingInstruction(answerStyle: AnswerStyle): string {
  const bulletCount = answerStyle.detailLevel === "short" ? "2-3" : "3-5";
  return `- In "answer", write markdown using EXACTLY these headings in this order:
## Introduction
## Key Points
## Exam Tip
## Quick Summary
- Under "Key Points", include ${bulletCount} bullet points.`;
}

function splitSentences(text: string): string[] {
  return (text.replace(/\n+/g, " ").match(/[^.!?]+[.!?]?/g) || [text])
    .map((s) => s.trim())
    .filter(Boolean);
}

function enforceConsistentAnswerMarkdown(answer: string, answerStyle: AnswerStyle): string {
  const cleaned = decodeJsonLikeString(answer || "").trim();
  if (!cleaned) {
    return `## Introduction
Here is a clear response to your question.

## Key Points
- Key idea 1.
- Key idea 2.

## Exam Tip
Use key terms and write in short, focused points.

## Quick Summary
Answer in a clear and structured way.`;
  }

  const hasRequiredHeadings =
    /(^|\n)##\s*Introduction\b/i.test(cleaned) &&
    /(^|\n)##\s*Key Points\b/i.test(cleaned) &&
    /(^|\n)##\s*Exam Tip\b/i.test(cleaned) &&
    /(^|\n)##\s*Quick Summary\b/i.test(cleaned);

  if (hasRequiredHeadings) return cleaned;

  const plain = cleaned.replace(/^\s*#{1,6}\s+/gm, "").trim();
  const sentences = splitSentences(plain);

  const intro = sentences.slice(0, Math.min(2, sentences.length)).join(" ") || plain;
  const summary = sentences[sentences.length - 1] || intro;

  const bodyCandidates = sentences.length > 3
    ? sentences.slice(1, -1)
    : sentences.slice(1);
  const maxPoints = answerStyle.detailLevel === "short" ? 3 : 5;
  const minPoints = answerStyle.detailLevel === "short" ? 2 : 3;

  const points = bodyCandidates.slice(0, maxPoints);
  while (points.length < minPoints && points.length < sentences.length) {
    const next = sentences[points.length] || plain;
    if (!points.includes(next)) points.push(next);
    else break;
  }

  const normalizedPoints = points.length
    ? points
    : [plain];

  const examTip = answerStyle.detailLevel === "short"
    ? "Keep your answer short, use subject keywords, and write direct points."
    : "Use headings, include key terms, and support each point with a relevant explanation.";

  return `## Introduction
${intro}

## Key Points
${normalizedPoints.map((point) => `- ${point}`).join("\n")}

## Exam Tip
${examTip}

## Quick Summary
${summary}`;
}

function decodeJsonLikeString(text: string): string {
  return text
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"')
    .trim();
}

function isTemplateAnswer(text: string): boolean {
  const normalized = text.toLowerCase().trim();
  if (!normalized) return true;

  const templateSnippets = [
    "detailed, properly formatted answer with headings and bullet points",
    "brief exam-focused answer in 2-4 lines",
    "clear exam-focused answer in 1-2 short paragraphs",
    "write the actual answer content here",
    "write the real answer here",
    "actual answer text here",
  ];

  return templateSnippets.some((snippet) => normalized.includes(snippet));
}

function cleanAnswerText(text: string): string {
  const cleaned = text
    .replace(/^```(?:json|markdown)?\s*/i, "")
    .replace(/```\s*$/i, "")
    .trim();

  if (!cleaned) return "";

  try {
    const parsed = JSON.parse(cleaned);
    if (parsed && typeof parsed.answer === "string") {
      return decodeJsonLikeString(parsed.answer);
    }
  } catch {
    // fall through
  }

  const fullJsonMatch = cleaned.match(/\{[\s\S]*\}/);
  if (fullJsonMatch) {
    try {
      const parsed = JSON.parse(fullJsonMatch[0]);
      if (parsed && typeof parsed.answer === "string") {
        return decodeJsonLikeString(parsed.answer);
      }
    } catch {
      // fall through
    }
  }

  const strictAnswerMatch = cleaned.match(/"answer"\s*:\s*"([\s\S]*?)"\s*(,\s*"(?:marking_points|common_mistakes)"|\})/i);
  if (strictAnswerMatch?.[1]) {
    return decodeJsonLikeString(strictAnswerMatch[1]);
  }

  const looseAnswerMatch = cleaned.match(/"answer"\s*:\s*"([\s\S]*)$/i);
  if (looseAnswerMatch?.[1]) {
    return decodeJsonLikeString(
      looseAnswerMatch[1]
        .replace(/",?\s*$/, "")
        .replace(/\}\s*$/, "")
    );
  }

  if (cleaned.startsWith("{") || cleaned.includes('"answer"')) {
    return "I can help with that. Please ask your question again in one sentence and I will give a clear exam-style answer.";
  }

  return cleaned;
}

function maybeStudentRetrievalDiagnostics(
  ragResult: RagRetrievalResult | null,
  mode: "grounded" | "nearby" | "general",
  subjectFilter?: string,
  reason?: string
): RetrievalDiagnostics | undefined {
  if (!INCLUDE_STUDENT_RETRIEVAL_DIAGNOSTICS) return undefined;
  return buildRetrievalDiagnostics(ragResult, mode, subjectFilter, reason);
}

async function generateExamAnswer(
  question: string,
  ragResult: RagRetrievalResult,
  history: HistoryMessage[] = [],
  answerStyle: AnswerStyle = inferAnswerStyle(question, history)
): Promise<ExamAnswer | null> {
  if (!ragResult.success || ragResult.groupedResults.length === 0) return null;

  try {
    const filePriority: Record<string, number> = { MS: 3, QP: 2, ER: 1, GT: 0 };
    const sortedGroups = [...ragResult.groupedResults].sort(
      (a, b) => (filePriority[b.filetype] || 0) - (filePriority[a.filetype] || 0)
    );

    const contextChunks = sortedGroups
      .flatMap((group) =>
        group.chunks
          .sort((a, b) => b.similarity - a.similarity)
          .map(
            (c) =>
              `[${resolveSubjectName(group.subject)} ${group.year} ${group.session} ${group.paper} ${group.filetype}]\n${c.content}`
          )
      )
      .join("\n\n---\n\n")
      .substring(0, 6000);

    if (!contextChunks.trim()) return null;

    const avgSimilarity =
      ragResult.rawSimilarityScores.length > 0
        ? ragResult.rawSimilarityScores.reduce((a, b) => a + b, 0) / ragResult.rawSimilarityScores.length
        : 0;

    const recentHistory = history.slice(-8).map((h) => ({
      role: h.role as "user" | "assistant",
      content: h.content,
    }));

    const messages: any[] = [
      {
        role: "system",
        content: `You are an expert O-Level Cambridge exam tutor. Answer the student's question based ONLY on the provided exam paper extracts.

STRICT RULES:
- Answer the question accurately in exam-focused language.
- Do NOT mention or guess any paper codes, series numbers (e.g. "1123", "2058"), or exam codes in your answer. The citation system identifies sources automatically.
- If the student asks "which paper is this from" or "what paper number", tell them to look at the source citations shown below your answer — do NOT guess codes yourself.
- Use only context that clearly matches the student's question/topic; ignore unrelated extracts.
- Be exam-focused, clear, and thorough.

OUTPUT FORMAT RULES:
${buildExamAnswerFormatInstruction(answerStyle)}
${buildAnswerMarkdownHeadingInstruction(answerStyle)}

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "answer": "<write the real answer here>",
  "marking_points": [
    {"point": "Key marking criterion as in a mark scheme", "marks": 1}
  ],
  "common_mistakes": ["A typical error O-Level students make"]
}

marking_points: 3-5 items, marks value 1-3 each.
common_mistakes: 2-3 items.`,
      },
      ...recentHistory,
      {
        role: "user",
        content: `Question: ${question}\n\nContext from past papers (avg relevance ${(avgSimilarity * 100).toFixed(0)}%):\n\n${contextChunks}`,
      },
    ];

    const completion = await groq.chat.completions.create({
      messages,
      model: GROQ_MODEL,
      max_tokens: buildMaxTokens(answerStyle, true),
      temperature: 0.1,
    });

    const responseText = completion.choices[0]?.message?.content || "";

    let parsed: any = null;
    try {
      const cleaned = responseText.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
      const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
      parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
    } catch { parsed = null; }

    if (!parsed?.answer) {
      parsed = {
        answer: cleanAnswerText(responseText) || "Unable to generate a structured answer.",
        marking_points: [{ point: "Refer to source documents", marks: 1 }],
        common_mistakes: ["Review the exam context above"],
      };
    }

    let finalAnswer = typeof parsed?.answer === "string"
      ? decodeJsonLikeString(parsed.answer)
      : "";
    if (!finalAnswer || isTemplateAnswer(finalAnswer)) {
      const recovered = cleanAnswerText(responseText);
      finalAnswer = !isTemplateAnswer(recovered) ? recovered : "I can help with that. Please ask again and I will provide a full, structured answer.";
    }
    finalAnswer = enforceConsistentAnswerMarkdown(finalAnswer, answerStyle);

    let markingPoints: Array<{ point: string; marks: number }> = [];
    if (Array.isArray(parsed.marking_points) && parsed.marking_points.length > 0) {
      markingPoints = parsed.marking_points.map((item: any) => {
        if (typeof item === "string") return { point: item, marks: 1 };
        return { point: String(item.point || item), marks: Math.max(1, Math.min(3, Number(item.marks) || 1)) };
      });
    } else {
      markingPoints = [{ point: "Key concept from exam sources", marks: 1 }];
    }

    const commonMistakes: string[] = Array.isArray(parsed.common_mistakes)
      ? parsed.common_mistakes.filter((m: any) => typeof m === "string" && m.trim())
      : [];

    let confidenceScore = Math.min(1, Math.max(0.1, avgSimilarity));
    if (ragResult.groupedResults.length >= 3) confidenceScore = Math.min(1, confidenceScore * 1.15);

    const citations: Citation[] = dedupeCitations(buildCitationsFromGroups(sortedGroups, "direct"));

    // Find the best chunk that actually looks like an exam question
    // Reject ER/mark-scheme content, require a "?" or question-style opening
    function looksLikeExamQuestion(text: string): boolean {
      const lower = text.toLowerCase();
      // Reject examiner report / mark scheme prose
      const isReportContent = /\b(candidates|principal examiner|examiner report|examiners|many candidates|few candidates|common errors|mark scheme|© 20\d{2}|cambridge ordinary level\s+\d{4})\b/.test(lower);
      if (isReportContent) return false;
      // Must contain a question mark OR start with a question action verb
      const hasQuestion = text.includes("?");
      const hasQuestionVerb = /^\s*(explain|describe|state|define|calculate|suggest|give|what|how|why|outline|compare|evaluate|discuss|draw|predict|deduce|justify|identify|name|list|show|determine|find|write|complete|use)/i.test(text);
      return hasQuestion || hasQuestionVerb;
    }

    // Collect all chunks from QP groups first, then all groups, and find the best exam-like chunk
    let relatedQuestion: RelatedQuestion | undefined;
    const allCandidates = [
      ...sortedGroups.filter((g) => g.filetype === "QP"),
      ...sortedGroups.filter((g) => g.filetype !== "QP"),
    ].flatMap((group) =>
      group.chunks
        .filter((c) => looksLikeExamQuestion(c.content))
        .map((c) => ({ group, chunk: c }))
    ).sort((a, b) => b.chunk.similarity - a.chunk.similarity);

    const best = allCandidates[0];
    if (best && best.chunk.similarity >= 0.55) {
      relatedQuestion = {
        type: best.chunk.similarity >= 0.75 ? "exact" : "similar",
        text: best.chunk.content.trim(),
        source: {
          subject: resolveSubjectName(best.group.subject),
          year: best.group.year,
          session: best.group.session,
          paper: best.group.paper,
          file_type: best.group.filetype,
          paper_file_id: best.group.paperFileId,
          storage_path: best.group.storagePath,
          paper_view_url: getPaperViewUrl(best.group.paperFileId, best.group.storagePath),
        },
        similarity: best.chunk.similarity,
      };
    }

    return {
      answer: finalAnswer,
      markingPoints,
      commonMistakes,
      citations: citations.slice(0, 6),
      confidenceScore,
      coveragePercentage: Math.min(100, (ragResult.groupedResults.length / 8) * 100),
      relatedQuestion,
    };
  } catch (err) {
    console.error("Answer generation error:", err);
    return null;
  }
}

// ============================================================================
// DIRECT LLM ANSWER (no RAG context — fallback when retrieval fails)
// ============================================================================

async function generateDirectAnswer(
  question: string,
  history: HistoryMessage[] = [],
  answerStyle: AnswerStyle = inferAnswerStyle(question, history)
): Promise<{ answer: string; markingPoints: Array<{ point: string; marks: number }>; commonMistakes: string[] } | null> {
  try {
    const recentHistory = history.slice(-8).map((h) => ({
      role: h.role as "user" | "assistant",
      content: h.content,
    }));

    const messages: any[] = [
      {
        role: "system",
        content: `You are an expert O-Level Cambridge exam tutor with deep knowledge of the Cambridge syllabus for all O-Level subjects including Physics, Chemistry, Biology, Mathematics, English, Islamiyat, Pakistan Studies, Computer Science, Economics, Geography, History, and more.

Answer the student's question using your Cambridge O-Level curriculum knowledge.

RULES:
- Give a clear, accurate, exam-focused answer suitable for O-Level students.
- Use Cambridge mark-scheme language where possible.
- Do NOT mention paper codes or series numbers.
- Be clear, structured, and educational.

OUTPUT FORMAT RULES:
${buildExamAnswerFormatInstruction(answerStyle)}
${buildAnswerMarkdownHeadingInstruction(answerStyle)}

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "answer": "<write the real answer here>",
  "marking_points": [
    {"point": "Key marking criterion", "marks": 1}
  ],
  "common_mistakes": ["A typical error students make"]
}

marking_points: 3-5 items, marks value 1-3 each.
common_mistakes: 2-3 items.`,
      },
      ...recentHistory,
      { role: "user", content: question },
    ];

    const completion = await groq.chat.completions.create({
      messages,
      model: GROQ_MODEL,
      max_tokens: buildMaxTokens(answerStyle, false),
      temperature: 0.1,
    });

    const responseText = completion.choices[0]?.message?.content || "";

    let parsed: any = null;
    try {
      const cleaned = responseText.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
      const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
      parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
    } catch { parsed = null; }

    if (!parsed?.answer) {
      parsed = {
        answer: cleanAnswerText(responseText) || "Unable to generate an answer.",
        marking_points: [],
        common_mistakes: [],
      };
    }

    let finalAnswer = typeof parsed?.answer === "string"
      ? decodeJsonLikeString(parsed.answer)
      : "";
    if (!finalAnswer || isTemplateAnswer(finalAnswer)) {
      const recovered = cleanAnswerText(responseText);
      finalAnswer = !isTemplateAnswer(recovered) ? recovered : "I can help with that. Please ask again and I will provide a clear exam-style answer.";
    }
    finalAnswer = enforceConsistentAnswerMarkdown(finalAnswer, answerStyle);

    let markingPoints: Array<{ point: string; marks: number }> = [];
    if (Array.isArray(parsed.marking_points) && parsed.marking_points.length > 0) {
      markingPoints = parsed.marking_points.map((item: any) => {
        if (typeof item === "string") return { point: item, marks: 1 };
        return { point: String(item.point || item), marks: Math.max(1, Math.min(3, Number(item.marks) || 1)) };
      });
    }

    const commonMistakes: string[] = Array.isArray(parsed.common_mistakes)
      ? parsed.common_mistakes.filter((m: any) => typeof m === "string" && m.trim())
      : [];

    return { answer: finalAnswer, markingPoints, commonMistakes };
  } catch (err) {
    console.error("Direct answer generation error:", err);
    return null;
  }
}

// ============================================================================
// SMALLTALK
// ============================================================================

const smalltalkResponses = [
  "Hi! Ask me any O-Level exam question — Chemistry, Physics, Biology, Maths, Islamiyat, and more. I'll answer using real past exam papers with marking points.",
  "Hello! Try asking a topic question like \"explain osmosis\" or find papers like \"Chemistry 2023 QP\".",
  "Hey! I can answer O-Level exam questions and help find past papers. What do you need?",
  "Hi! Ask me anything about your O-Level subjects. I'll give you a mark-scheme style answer from past papers.",
];

function getSmallTalkResponse(): string {
  return smalltalkResponses[Math.floor(Math.random() * smalltalkResponses.length)];
}

function buildRetrievalDiagnostics(
  ragResult: RagRetrievalResult | null,
  mode: "grounded" | "nearby" | "general",
  subjectFilter?: string,
  reason?: string
): RetrievalDiagnostics {
  const stats = ragResult?.stats;
  return {
    mode,
    used_embeddings: Boolean(stats && stats.rpcMatchCount > 0),
    subject_filter: subjectFilter,
    reason,
    match_count_requested: stats?.matchCountRequested,
    rpc_match_count: stats?.rpcMatchCount,
    threshold_match_count: stats?.thresholdMatchCount,
    metadata_chunk_count: stats?.metadataChunkCount,
    grouped_file_count: stats?.groupedFileCount,
    best_similarity: stats?.bestSimilarity,
    avg_top3_similarity: stats?.avgTop3Similarity,
  };
}

function buildNearbyReferences(
  ragResult: RagRetrievalResult,
  subjectFilter?: string,
  limit: number = NEARBY_REFERENCE_LIMIT
): Citation[] {
  if (!subjectFilter) return [];

  const sourceGroups = ragResult.nearbyGroupedResults.length
    ? ragResult.nearbyGroupedResults
    : ragResult.groupedResults;

  let candidates = buildCitationsFromGroups(sourceGroups, "nearby");

  const subjectCandidates = candidates.filter((c) => String(c.subject) === String(subjectFilter));
  if (subjectCandidates.length === 0) return [];
  candidates = subjectCandidates;

  return dedupeCitations(candidates).slice(0, limit);
}

async function streamPaperPdfFromStorage(
  storagePath: string,
  res: Response,
  fallbackFileName?: string
): Promise<boolean> {
  const normalizedStoragePath = normalizeStoragePath(storagePath);
  if (!normalizedStoragePath) return false;

  const { data: pdfBlob, error: downloadError } = await supabase
    .storage
    .from(SUPABASE_STORAGE_BUCKET)
    .download(normalizedStoragePath);

  if (downloadError || !pdfBlob) {
    return false;
  }

  const fileName = normalizedStoragePath.split("/").pop() || fallbackFileName || "paper.pdf";
  const safeFileName = fileName.replace(/"/g, "");
  const fileBuffer = Buffer.from(await pdfBlob.arrayBuffer());

  res.setHeader("Content-Type", "application/pdf");
  res.setHeader("Content-Disposition", `inline; filename="${safeFileName}"`);
  res.send(fileBuffer);
  return true;
}

// ============================================================================
// GET /rag/paper-file/by-path/view — stream cited paper PDF via storage path
// ============================================================================

router.get("/paper-file/by-path/view", async (req: Request, res: Response) => {
  try {
    const storagePath = typeof req.query.storagePath === "string"
      ? req.query.storagePath
      : "";

    const normalizedStoragePath = normalizeStoragePath(storagePath);
    if (!normalizedStoragePath) {
      return res.status(400).json({ error: "storagePath is required" });
    }

    const streamed = await streamPaperPdfFromStorage(normalizedStoragePath, res);
    if (!streamed) {
      return res.status(404).json({ error: "Unable to load paper file from storage" });
    }

    return;
  } catch (error) {
    console.error("Paper file by-path view error:", error);
    return res.status(500).json({ error: "Failed to open paper file" });
  }
});

// ============================================================================
// GET /rag/paper-file/:paperFileId/view — stream cited paper PDF from Supabase
// ============================================================================

router.get("/paper-file/:paperFileId/view", async (req: Request, res: Response) => {
  try {
    const { paperFileId } = req.params;
    const storagePathFromQuery = typeof req.query.storagePath === "string"
      ? req.query.storagePath
      : undefined;
    const normalizedStoragePathFromQuery = normalizeStoragePath(storagePathFromQuery);

    if (!isUsableId(paperFileId) && !normalizedStoragePathFromQuery) {
      return res.status(400).json({ error: "paperFileId or storagePath is required" });
    }

    let resolvedStoragePath = normalizedStoragePathFromQuery;
    if (isUsableId(paperFileId)) {
      const { data: paperFile, error: paperFileError } = await supabase
        .from("paper_files")
        .select("id, storage_path")
        .eq("id", paperFileId)
        .single();

      if (!paperFileError && paperFile?.storage_path) {
        resolvedStoragePath = normalizeStoragePath(paperFile.storage_path) || resolvedStoragePath;
      }
    }

    if (!resolvedStoragePath) {
      return res.status(404).json({ error: "Paper file not found" });
    }

    const streamed = await streamPaperPdfFromStorage(
      resolvedStoragePath,
      res,
      isUsableId(paperFileId) ? `${paperFileId}.pdf` : undefined
    );

    if (!streamed) {
      return res.status(404).json({ error: "Unable to load paper file from storage" });
    }

    return;
  } catch (error) {
    console.error("Paper file view error:", error);
    return res.status(500).json({ error: "Failed to open paper file" });
  }
});

// ============================================================================
// GET /rag/subjects — returns all subjects from DB with resolved names
// ============================================================================

router.get("/subjects", async (_req: Request, res: Response) => {
  try {
    clearSubjectCache(); // Force fresh fetch
    const subjects = await getSubjectCache();
    res.json(subjects.map((s) => ({ code: s.code, name: s.name, level: s.level })));
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch subjects" });
  }
});

// ============================================================================
// POST /rag/query — main 3-stage RAG pipeline
// ============================================================================

router.post("/query", async (req: Request, res: Response) => {
  try {
    const { question, limit = 5, filters, history = [] } = req.body as RagQueryRequest;

    if (!question?.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }

    // Warm up subject cache in background (don't await)
    getSubjectCache().catch(() => {});

    // STAGE A: INTENT
    const classification = classifyIntent(question);

    // CASE 1: SMALLTALK
    if (classification.intent === "smalltalk") {
      return res.json({ type: "smalltalk", answer: getSmallTalkResponse(), citations: [] } as RagQueryResponse);
    }

    // CASE 2: PAPER LOOKUP
    if (classification.intent === "paper_lookup") {
      const meta = classification.metadata;

      // Try to find the subject in DB dynamically
      let subjectId: string | null = null;
      let resolvedSubjectName: string | null = null;

      if (meta?.subjectKeyword) {
        const matched = await findSubjectByName(meta.subjectKeyword);
        if (matched) {
          subjectId = matched.id;
          resolvedSubjectName = matched.name;
        }
      }

      let query = supabase.from("papers").select(`
        id, year, session, paper, subject_id,
        subjects(code, level),
        paper_files(file_type, storage_path, id)
      `);

      if (subjectId) query = query.eq("subject_id", subjectId);
      if (meta?.year) query = query.eq("year", meta.year);

      const { data, error } = await query;
      if (error) return res.status(500).json({ error: error.message });

      let results = (data || []) as any[];

      if (meta?.fileType) {
        results = results.filter((p) =>
          p.paper_files?.some((pf: any) => pf.file_type === meta.fileType)
        );
      }
      if (meta?.paper) {
        results = results.filter((p) => p.paper === meta.paper);
      }

      const grouped = results.reduce((acc: any, paper: any) => {
        const key = `${paper.subjects?.code}-${paper.year}`;
        if (!acc[key]) {
          acc[key] = {
            subject: paper.subjects?.code,
            subjectName: resolveSubjectName(paper.subjects?.code),
            year: paper.year,
            level: paper.subjects?.level,
            sessions: {},
          };
        }
        if (!acc[key].sessions[paper.session]) {
          acc[key].sessions[paper.session] = { session: paper.session, papers: {} };
        }
        if (!acc[key].sessions[paper.session].papers[paper.paper]) {
          acc[key].sessions[paper.session].papers[paper.paper] = { paper: paper.paper, files: {} };
        }
        (paper.paper_files || []).forEach((pf: any) => {
          acc[key].sessions[paper.session].papers[paper.paper].files[pf.file_type] = {
            file_type: pf.file_type, storage_path: pf.storage_path, id: pf.id,
          };
        });
        return acc;
      }, {});

      const groupedArr = Object.values(grouped) as any[];

      // If we couldn't match the subject, include available subjects as hint
      const subjects = await getSubjectCache();
      const availableSubjects = subjects.map((s) => ({ code: s.code, name: s.name, level: s.level }));

      let answerText = "";
      if (groupedArr.length > 0) {
        const subjectLabel = resolvedSubjectName || meta?.subjectKeyword || "your query";
        answerText = `Found ${groupedArr.length} paper set(s) for ${subjectLabel}:`;
      } else if (!subjectId && meta?.subjectKeyword) {
        answerText = `Could not identify subject "${meta.subjectKeyword}" in the database. Available subjects are shown below — try using the subject selector or type the exact name.`;
      } else {
        answerText = "No papers found for your query.";
      }

      return res.json({
        type: "paper_lookup",
        answer: answerText,
        results: groupedArr,
        citations: [],
        available_subjects: !subjectId ? availableSubjects : undefined,
      } as RagQueryResponse);
    }

    // CASE 3: EXAM QUESTION
    const answerStyle = inferAnswerStyle(question, history);
    const effectiveFilters: RagQueryRequest["filters"] = { ...(filters || {}) };
    let inferredSubjectKeywordMissing: string | null = null;
    let skipRagForSubjectSafety = false;
    let subjectSafetyReason = "";
    const subjectInference = await inferSubjectFromQuestion(question);

    if (
      effectiveFilters.subject &&
      subjectInference?.matchedSubject &&
      String(effectiveFilters.subject) !== String(subjectInference.matchedSubject.code)
    ) {
      const shouldOverrideProvidedFilter =
        subjectInference.source === "explicit-keyword" ||
        subjectInference.source === "topic-hint-strong";

      if (shouldOverrideProvidedFilter) {
        console.log(
          `[RAG] Overriding provided subject filter ${effectiveFilters.subject} -> ${subjectInference.matchedSubject.code} based on ${subjectInference.source} (${subjectInference.keyword}).`
        );
        effectiveFilters.subject = subjectInference.matchedSubject.code;
      }
    }

    if (!effectiveFilters.subject) {
      if (subjectInference?.matchedSubject) {
        effectiveFilters.subject = subjectInference.matchedSubject.code;
        console.log(
          `[RAG] Auto subject filter (${subjectInference.source}): "${subjectInference.keyword}" -> ${subjectInference.matchedSubject.code} (${subjectInference.matchedSubject.name})`
        );
      } else if (subjectInference?.keyword) {
        inferredSubjectKeywordMissing = subjectInference.keyword;
        skipRagForSubjectSafety = true;
        subjectSafetyReason = `Subject inferred from question but not indexed: ${subjectInference.keyword}`;
        console.log(
          `[RAG] Subject hint (${subjectInference.source}) "${subjectInference.keyword}" detected but not found in indexed subjects. Skipping RAG retrieval.`
        );
      } else {
        skipRagForSubjectSafety = true;
        subjectSafetyReason = "No confident subject inferred from question; skipping cross-subject retrieval.";
        console.log("[RAG] No confident subject inferred. Skipping unfiltered retrieval to avoid wrong-subject context.");
      }
    }

    if (inferredSubjectKeywordMissing || skipRagForSubjectSafety) {
      const directAnswer = await generateDirectAnswer(question, history, answerStyle);
      if (directAnswer) {
        return res.json({
          type: "exam_question",
          answer: directAnswer.answer,
          marking_points: directAnswer.markingPoints,
          common_mistakes: directAnswer.commonMistakes,
          citations: [],
          low_confidence: true,
          retrieval: maybeStudentRetrievalDiagnostics(
            null,
            "general",
            effectiveFilters.subject,
            inferredSubjectKeywordMissing
              ? `No indexed subject matched inferred topic: ${inferredSubjectKeywordMissing}`
              : subjectSafetyReason
          ),
        } as RagQueryResponse);
      }
    }

    const ragResult = await ragRetrieval(question, effectiveFilters);
    const nearbyReferences = buildNearbyReferences(ragResult, effectiveFilters.subject);

    if (!ragResult.success) {
      if (ragResult.error?.includes("unavailable")) {
        return res.json({
          type: "exam_question",
          answer: "The AI study assistant is temporarily unavailable. Please try again later.",
          citations: [],
          retrieval: maybeStudentRetrievalDiagnostics(
            ragResult,
            nearbyReferences.length > 0 ? "nearby" : "general",
            effectiveFilters.subject,
            ragResult.error
          ),
          nearby_references: nearbyReferences.length > 0 ? nearbyReferences : undefined,
        } as RagQueryResponse);
      }
      // RAG found nothing — fall back to direct LLM answer
      const directAnswer = await generateDirectAnswer(question, history, answerStyle);
      if (directAnswer) {
        const hasNearby = nearbyReferences.length > 0;
        return res.json({
          type: "exam_question",
          answer: directAnswer.answer,
          marking_points: directAnswer.markingPoints,
          common_mistakes: directAnswer.commonMistakes,
          citations: [],
          low_confidence: true,
          nearby_references: hasNearby ? nearbyReferences : undefined,
          retrieval: maybeStudentRetrievalDiagnostics(
            ragResult,
            hasNearby ? "nearby" : "general",
            effectiveFilters.subject,
            ragResult.error || "No reliable grounded context"
          ),
        } as RagQueryResponse);
      }
      return res.json({
        type: "exam_question",
        answer: "I couldn't generate an answer right now. Please try again.",
        citations: [],
        nearby_references: nearbyReferences.length > 0 ? nearbyReferences : undefined,
        retrieval: maybeStudentRetrievalDiagnostics(
          ragResult,
          nearbyReferences.length > 0 ? "nearby" : "general",
          effectiveFilters.subject,
          ragResult.error || "Direct answer generation failed"
        ),
      } as RagQueryResponse);
    }

    const reliableContext = hasReliableRagContext(ragResult, effectiveFilters.subject);
    if (!reliableContext) {
      const directAnswer = await generateDirectAnswer(question, history, answerStyle);
      if (directAnswer) {
        return res.json({
          type: "exam_question",
          answer: directAnswer.answer,
          marking_points: directAnswer.markingPoints,
          common_mistakes: directAnswer.commonMistakes,
          citations: [],
          low_confidence: true,
          nearby_references: nearbyReferences.length > 0 ? nearbyReferences : undefined,
          retrieval: maybeStudentRetrievalDiagnostics(
            ragResult,
            nearbyReferences.length > 0 ? "nearby" : "general",
            effectiveFilters.subject,
            "Embeddings retrieved but reliability thresholds were not met"
          ),
        } as RagQueryResponse);
      }
      return res.json({
        type: "exam_question",
        answer: "I couldn't find reliable past-paper context for this question right now. Please try again.",
        citations: [],
        low_confidence: true,
        nearby_references: nearbyReferences.length > 0 ? nearbyReferences : undefined,
        retrieval: maybeStudentRetrievalDiagnostics(
          ragResult,
          nearbyReferences.length > 0 ? "nearby" : "general",
          effectiveFilters.subject,
          "Embeddings retrieved but no reliable context"
        ),
      } as RagQueryResponse);
    }

    const examAnswer = await generateExamAnswer(question, ragResult, history, answerStyle);

    if (!examAnswer) {
      const directAnswer = await generateDirectAnswer(question, history, answerStyle);
      if (directAnswer) {
        return res.json({
          type: "exam_question",
          answer: directAnswer.answer,
          marking_points: directAnswer.markingPoints,
          common_mistakes: directAnswer.commonMistakes,
          citations: [],
          low_confidence: true,
          nearby_references: nearbyReferences.length > 0 ? nearbyReferences : undefined,
          retrieval: maybeStudentRetrievalDiagnostics(
            ragResult,
            nearbyReferences.length > 0 ? "nearby" : "general",
            effectiveFilters.subject,
            "Context found but answer synthesis failed"
          ),
        } as RagQueryResponse);
      }

      return res.json({
        type: "exam_question",
        answer: "I couldn't generate an answer right now. Please try again.",
        citations: [],
        nearby_references: nearbyReferences.length > 0 ? nearbyReferences : undefined,
        retrieval: maybeStudentRetrievalDiagnostics(
          ragResult,
          nearbyReferences.length > 0 ? "nearby" : "general",
          effectiveFilters.subject,
          "Context answer generation failed"
        ),
      } as RagQueryResponse);
    }

    return res.json({
      type: "exam_question",
      answer: examAnswer.answer,
      marking_points: examAnswer.markingPoints,
      common_mistakes: examAnswer.commonMistakes,
      citations: examAnswer.citations.slice(0, limit),
      confidence_score: examAnswer.confidenceScore,
      coverage_percentage: examAnswer.coveragePercentage,
      low_confidence: examAnswer.confidenceScore < 0.4,
      related_question: examAnswer.relatedQuestion,
      retrieval: maybeStudentRetrievalDiagnostics(
        ragResult,
        "grounded",
        effectiveFilters.subject,
        "Embeddings grounded answer"
      ),
    } as RagQueryResponse);

  } catch (error) {
    console.error("RAG query error:", error);
    return res.status(500).json({ error: error instanceof Error ? error.message : "Unknown error" });
  }
});

export default router;
