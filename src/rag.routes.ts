import { Router, Request, Response } from "express";
import { supabase } from "./lib/supabase";
import Groq from "groq-sdk";
import { promises as fs } from "fs";
import path from "path";

const router = Router();
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";
const COHERE_API_KEY = (process.env.COHERE_API_KEY || "").trim();
const EMBEDDING_MODEL = "embed-multilingual-v3.0";

function parseBooleanEnv(value: string | undefined, defaultValue: boolean): boolean {
  if (!value || !value.trim()) return defaultValue;
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function parseNumberEnv(value: string | undefined, defaultValue: number): number {
  if (!value || !value.trim()) return defaultValue;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : defaultValue;
}

const INCLUDE_STUDENT_RETRIEVAL_DIAGNOSTICS = parseBooleanEnv(
  process.env.INCLUDE_STUDENT_RETRIEVAL_DIAGNOSTICS,
  false
);
const SIMILARITY_THRESHOLD = 0.34;
// Unfiltered (no subject) retrieval: fetch more candidates to compensate for
// searching across all subjects at once.
const TOP_K = 32;
const TOP_K_WITH_FILTERS = Math.max(
  TOP_K,
  Math.floor(parseNumberEnv(process.env.TOP_K_WITH_FILTERS, 96))
);
const NEARBY_REFERENCE_LIMIT = Math.max(
  3,
  Math.floor(parseNumberEnv(process.env.NEARBY_REFERENCE_LIMIT, 4))
);
// Reliability thresholds for unfiltered retrieval (no subject filter).
// Slightly looser than the old 0.55/0.48 — embedding cosine similarity of
// ~0.48 across ALL subjects is still a strong topical match.
const MIN_BEST_SIMILARITY = parseNumberEnv(
  process.env.MIN_BEST_SIMILARITY_FOR_CONTEXT,
  0.43
);
const MIN_AVG_TOP3_SIMILARITY = parseNumberEnv(
  process.env.MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT,
  0.36
);
const MIN_BEST_SIMILARITY_WITH_SUBJECT = parseNumberEnv(
  process.env.MIN_BEST_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT,
  0.36
);
const MIN_AVG_TOP3_WITH_SUBJECT = parseNumberEnv(
  process.env.MIN_AVG_TOP3_SIMILARITY_FOR_CONTEXT_WITH_SUBJECT,
  0.30
);

// -- SUBJECT MAP -------------------------------------------------------------
// Maps normalized keyword -> exact subject name in past_paper_meta
const SUBJECT_NAME_MAP: Record<string, string> = {
  chemistry: "Chemistry 1011",
  english: "English 1012",
  islamiyat: "Islamiyat 1013",
  islamiat: "Islamiyat 1013",
  "islamic studies": "Islamiyat 1013",
  mathematics: "Mathematics 1014",
  maths: "Mathematics 1014",
  math: "Mathematics 1014",
  "pakistan studies": "Pakistan Studies 1015",
  "pak studies": "Pakistan Studies 1015",
  "pakistan study": "Pakistan Studies 1015",
  physics: "Physics 1016",
};

const SUBJECT_KEYWORDS = [
  "pakistan studies",
  "pak studies",
  "pakistan study",
  "islamic studies",
  "additional mathematics",
  "chemistry",
  "physics",
  "mathematics",
  "islamiyat",
  "islamiat",
  "english",
  "maths",
  "math",
].sort((a, b) => b.length - a.length);

const TOPIC_SUBJECT_HINTS: Array<{ pattern: RegExp; subjectHint: string }> = [
  {
    pattern: /\b(vector|momentum|force|displacement|velocity|acceleration|newton|resultant|pressure|density|wave|current|voltage|resistance|circuit|lens|refraction|thermal energy|kinetic energy|potential energy|internal energy|specific heat|latent heat|heat transfer|conduction|convection|radiation|temperature|thermometer|energy transfer|work done|power output|gravitational|centripetal|magnetic field|electric field|capacitor|transistor|nuclear|radioactive|half.?life|photon|electron|proton|neutron)\b/i,
    subjectHint: "physics",
  },
  {
    pattern: /\b(logarithm|integration|differentiat|derivative|gradient|trigonometry|algebra|matrix|matrices|simultaneous equations|probability|statistics|quadratic)\b/i,
    subjectHint: "mathematics",
  },
  {
    pattern: /\b(mole|moles|stoichiometry|acid|alkali|electrolysis|oxidation|reduction|salt|bond|ionic|covalent|periodic table|redox)\b/i,
    subjectHint: "chemistry",
  },
  {
    pattern: /\b(hazrat|sahaba|prophet muhammad|quran|hadith|sunnah|hijrah|madinah|makkah|abu bakr|umar|uthman|ali|khalifa)\b/i,
    subjectHint: "islamiyat",
  },
  {
    pattern: /\b(pakistan movement|lahore resolution|two nation theory|muslim league|jinnah|allama iqbal|partition|1947|mughal|british raj)\b/i,
    subjectHint: "pakistan studies",
  },
];

function resolveSubjectName(keyword: string): string {
  return SUBJECT_NAME_MAP[keyword.toLowerCase().trim()] || keyword;
}

function detectSubjectKeyword(question: string): string | undefined {
  const lower = question.toLowerCase();
  return SUBJECT_KEYWORDS.find((kw) => lower.includes(kw));
}

function inferSubjectFromQuestion(question: string): string | null {
  const keyword = detectSubjectKeyword(question);
  if (keyword) return keyword;

  for (const hint of TOPIC_SUBJECT_HINTS) {
    if (hint.pattern.test(question)) return hint.subjectHint;
  }
  return null;
}

// -- TYPES -------------------------------------------------------------------
type Intent = "exam_question" | "smalltalk";

interface PastPaperChunk {
  chunk_id: string;
  content: string;
  similarity: number;
  subject: string;
  year: number;
  session: string;
  paper: string;
  variant: string;
}

interface Citation {
  subject: string;
  year: number;
  session: string;
  paper: string;
  variant: string;
  similarity: number;
  relation: "direct" | "nearby";
  questionNumber?: string;
  subQuestion?: string;
  marks?: number;
  topicGeneral?: string;
  topicSyllabus?: string;
}

interface RetrievalDiagnostics {
  mode: "grounded" | "nearby" | "general";
  subject_filter?: string;
  match_count?: number;
  best_similarity?: number;
  avg_top3?: number;
  reason?: string;
  requested_match_count?: number;
  similarity_threshold?: number;
}

interface DeveloperTrace {
  intent: Intent;
  answer_mode: AnswerMode;
  should_search_rag: boolean;
  followup_detected: boolean;
  decision_reason: string;
  history_turns: number;
  retrieval_query?: string;
  retrieval_query_rewritten?: boolean;
  subject_keyword?: string;
  subject_filter?: string;
  inferred_subject_from_results?: string;
  embedding_dimensions?: number;
}

interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

type AnswerMode = "full_answer" | "mark_scheme_only";

type SourceType = "past_paper" | "nearby_only" | "none";

interface QuestionResolution {
  retrievalQuestion: string;
  answerMode: AnswerMode;
  usedHistoryQuestion: boolean;
}

interface RagQueryRequest {
  question: string;
  limit?: number;
  filters?: { subject?: string; year?: number };
  history?: HistoryMessage[];
}

interface RagQueryResponse {
  type: "smalltalk" | "exam_question";
  answer: string;
  marking_points?: Array<{ point: string; marks: number }>;
  common_mistakes?: string[];
  citations: Citation[];
  confidence_score?: number;
  low_confidence?: boolean;
  retrieval?: RetrievalDiagnostics;
  nearby_references?: Citation[];
  source_note?: string;
  source_type?: SourceType;
  resolved_question?: string;
  developer_trace?: DeveloperTrace;
}

// -- EMBEDDING ---------------------------------------------------------------
async function getEmbedding(text: string): Promise<number[]> {
  if (!COHERE_API_KEY) throw new Error("COHERE_API_KEY not set");

  const res = await fetch("https://api.cohere.com/v2/embed", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${COHERE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      texts: [text],
      input_type: "search_query",
      embedding_types: ["float"],
    }),
    signal: AbortSignal.timeout(30000),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Cohere HTTP ${res.status}: ${body.slice(0, 200)}`);
  }

  const data = (await res.json()) as any;
  const embedding = data?.embeddings?.float?.[0] as number[];
  if (!Array.isArray(embedding) || embedding.length === 0) {
    throw new Error("Cohere returned empty embedding");
  }

  console.log(`[Embeddings] Cohere success: ${embedding.length} dims`);
  return embedding;
}

// -- RAG RETRIEVAL -----------------------------------------------------------
interface RagRetrievalResult {
  success: boolean;
  chunks: PastPaperChunk[];
  nearbyChunks: PastPaperChunk[];
  scores: number[];
  bestSimilarity: number;
  avgTop3: number;
  embeddingDimensions?: number;
  error?: string;
}

interface LocalFallbackEntry {
  subject: string;
  year: number;
  session: string;
  paper: string;
  variant: string;
  questionNumber?: string;
  subQuestion?: string;
  marks?: number;
  topicGeneral?: string;
  topicSyllabus?: string;
  questionText: string;
  markingScheme: string;
}

let localFallbackCache: LocalFallbackEntry[] | null = null;

function normalizeTokenList(text: string): string[] {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 3);
}

function lexicalSimilarity(a: string, b: string): number {
  const aTokens = Array.from(new Set(normalizeTokenList(a)));
  const bTokens = new Set(normalizeTokenList(b));
  if (!aTokens.length || !bTokens.size) return 0;

  let common = 0;
  for (const token of aTokens) {
    if (bTokens.has(token)) common += 1;
  }
  return common / aTokens.length;
}

function buildSyntheticChunkContent(entry: LocalFallbackEntry): string {
  const qRef = `${entry.questionNumber || ""}${entry.subQuestion ? ` ${entry.subQuestion}` : ""}`.trim();
  return [
    qRef ? `Question ${qRef}` : "Question",
    entry.topicGeneral || entry.topicSyllabus
      ? `Topic: ${entry.topicGeneral || "General"}${entry.topicSyllabus ? ` > ${entry.topicSyllabus}` : ""}`
      : "",
    Number.isFinite(entry.marks) ? `Marks: ${entry.marks}` : "",
    `Question: ${entry.questionText}`,
    `Mark Scheme: ${entry.markingScheme}`,
  ]
    .filter(Boolean)
    .join("\n");
}

async function listJsonFilesRecursively(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listJsonFilesRecursively(fullPath)));
      continue;
    }
    if (entry.isFile() && entry.name.toLowerCase().endsWith(".json")) {
      files.push(fullPath);
    }
  }

  return files;
}

function parseLocalPaperJson(raw: any, subjectName: string): LocalFallbackEntry[] {
  const entries: LocalFallbackEntry[] = [];
  if (!raw || typeof raw !== "object") return entries;

  for (const [yearKey, sessions] of Object.entries(raw as Record<string, any>)) {
    if (!sessions || typeof sessions !== "object") continue;
    const yearNum = Number(yearKey);

    for (const [sessionKey, papers] of Object.entries(sessions as Record<string, any>)) {
      if (!papers || typeof papers !== "object") continue;

      for (const [paperKey, variants] of Object.entries(papers as Record<string, any>)) {
        if (!variants || typeof variants !== "object") continue;

        for (const [variantKey, questions] of Object.entries(variants as Record<string, any>)) {
          if (!Array.isArray(questions)) continue;

          for (const q of questions) {
            if (!q || typeof q !== "object") continue;
            const questionText = String(q.question_text || "").trim();
            const markingScheme = String(q.marking_scheme || "").trim();
            if (!questionText) continue;

            entries.push({
              subject: subjectName,
              year: Number.isFinite(yearNum) ? yearNum : 0,
              session: String(sessionKey || "").trim(),
              paper: String(paperKey || "").trim(),
              variant: String(variantKey || "").trim(),
              questionNumber:
                q.question_number == null ? undefined : String(q.question_number).trim(),
              subQuestion:
                q.sub_question == null ? undefined : String(q.sub_question).trim(),
              marks:
                Number.isFinite(Number(q.marks)) && Number(q.marks) > 0
                  ? Number(q.marks)
                  : undefined,
              topicGeneral:
                q.topic_general == null ? undefined : String(q.topic_general).trim(),
              topicSyllabus:
                q.topic_syllabus == null ? undefined : String(q.topic_syllabus).trim(),
              questionText,
              markingScheme,
            });
          }
        }
      }
    }
  }

  return entries;
}

async function loadLocalFallbackEntries(): Promise<LocalFallbackEntry[]> {
  if (localFallbackCache) return localFallbackCache;

  const configuredDir = (process.env.JSON_DIR || "").trim();
  const candidates = [
    configuredDir,
    path.resolve(process.cwd(), "../_ext/Subject-Grading/O_LEVEL_MAIN_JSON"),
    path.resolve(process.cwd(), "_ext/Subject-Grading/O_LEVEL_MAIN_JSON"),
  ].filter(Boolean);

  for (const root of candidates) {
    try {
      const stat = await fs.stat(root);
      if (!stat.isDirectory()) continue;

      const subjectDirs = await fs.readdir(root, { withFileTypes: true });
      const allEntries: LocalFallbackEntry[] = [];

      for (const dirent of subjectDirs) {
        if (!dirent.isDirectory()) continue;
        const subjectPath = path.join(root, dirent.name);
        const jsonFiles = await listJsonFilesRecursively(subjectPath);

        for (const filePath of jsonFiles) {
          try {
            const text = await fs.readFile(filePath, "utf8");
            const parsed = JSON.parse(text);
            allEntries.push(...parseLocalPaperJson(parsed, dirent.name));
          } catch {
            // Skip malformed file and continue.
          }
        }
      }

      if (allEntries.length > 0) {
        console.log(`[RAG] Local fallback corpus loaded: ${allEntries.length} questions from ${root}`);
        localFallbackCache = allEntries;
        return allEntries;
      }
    } catch {
      // Try next candidate path.
    }
  }

  localFallbackCache = [];
  return localFallbackCache;
}

async function localFallbackRetrieval(
  question: string,
  subjectFilter?: string,
  yearFilter?: number,
  limit: number = TOP_K
): Promise<RagRetrievalResult | null> {
  const corpus = await loadLocalFallbackEntries();
  if (!corpus.length) return null;

  const filtered = corpus.filter((entry) => {
    const subjectOk = subjectFilter
      ? entry.subject.toLowerCase() === subjectFilter.toLowerCase()
      : true;
    const yearOk = yearFilter ? entry.year === yearFilter : true;
    return subjectOk && yearOk;
  });

  const pool = filtered.length > 0 ? filtered : corpus;

  const ranked = pool
    .map((entry) => {
      const sourceText = `${entry.questionText}\n${entry.markingScheme}`;
      const similarity = lexicalSimilarity(question, sourceText);
      return { entry, similarity };
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, Math.max(limit, 24));

  if (!ranked.length) return null;

  const all: PastPaperChunk[] = ranked.map(({ entry, similarity }, idx) => ({
    chunk_id: `local-${entry.subject}-${entry.year}-${entry.session}-${entry.paper}-${entry.variant}-${entry.questionNumber || idx}`,
    content: buildSyntheticChunkContent(entry),
    similarity,
    subject: entry.subject,
    year: entry.year,
    session: entry.session,
    paper: entry.paper,
    variant: entry.variant,
  }));

  const scores = all.map((c) => c.similarity);
  const sorted = [...scores].sort((a, b) => b - a);
  const bestSimilarity = sorted[0] || 0;
  const top3 = sorted.slice(0, 3);
  const avgTop3 = top3.length
    ? top3.reduce((s, v) => s + v, 0) / top3.length
    : 0;

  const aboveThreshold = all.filter((c) => c.similarity >= SIMILARITY_THRESHOLD);
  const nearbyChunks = all.slice(0, Math.min(all.length, TOP_K));
  const chunks = aboveThreshold.length > 0 ? aboveThreshold : nearbyChunks.slice(0, 6);

  console.log(
    `[RAG] Local fallback retrieval used; ${chunks.length} chunks selected (best=${bestSimilarity.toFixed(3)})`
  );

  return {
    success: chunks.length > 0,
    chunks,
    nearbyChunks,
    scores: chunks.map((c) => c.similarity),
    bestSimilarity,
    avgTop3,
    embeddingDimensions: undefined,
    error: "local_fallback_retrieval",
  };
}

async function ragRetrieval(
  question: string,
  subjectFilter?: string,
  yearFilter?: number
): Promise<RagRetrievalResult> {
  const empty: RagRetrievalResult = {
    success: false,
    chunks: [],
    nearbyChunks: [],
    scores: [],
    bestSimilarity: 0,
    avgTop3: 0,
    embeddingDimensions: undefined,
  };

  try {
    let embedding: number[];
    try {
      embedding = await getEmbedding(question);
    } catch (embeddingErr: any) {
      const local = await localFallbackRetrieval(question, subjectFilter, yearFilter);
      if (local) {
        return {
          ...local,
          error: `embedding_unavailable:${String(embeddingErr?.message || embeddingErr || "unknown")}`,
        };
      }
      throw embeddingErr;
    }

    const embeddingDimensions = embedding.length;
    const matchCount = subjectFilter ? TOP_K_WITH_FILTERS : TOP_K;

    const { data, error } = await supabase.rpc("search_past_papers", {
      query_embedding: embedding,
      match_count: matchCount,
      filter_subject: subjectFilter || null,
      filter_year: yearFilter || null,
    });

    if (error || !data) {
      const local = await localFallbackRetrieval(question, subjectFilter, yearFilter, matchCount);
      if (local) {
        return {
          ...local,
          embeddingDimensions,
          error: `supabase_rpc_failed:${error?.message || "RPC failed"}`,
        };
      }
      return { ...empty, error: error?.message || "RPC failed" };
    }

    const all = (data as PastPaperChunk[]).sort(
      (a, b) => b.similarity - a.similarity
    );

    if (!all.length || (all[0]?.similarity || 0) <= 0) {
      const local = await localFallbackRetrieval(question, subjectFilter, yearFilter, matchCount);
      if (local) {
        return {
          ...local,
          embeddingDimensions,
          error: "supabase_empty_or_zero_similarity_local_fallback",
        };
      }
    }

    const scores = all.map((c) => c.similarity);
    const sorted = [...scores].sort((a, b) => b - a);
    const bestSimilarity = sorted[0] || 0;
    const top3 = sorted.slice(0, 3);
    const avgTop3 = top3.length
      ? top3.reduce((s, v) => s + v, 0) / top3.length
      : 0;

    const aboveThreshold = all.filter(
      (c) => c.similarity >= SIMILARITY_THRESHOLD
    );
    const nearbyChunks = all.slice(0, Math.min(all.length, TOP_K));
    const chunks = aboveThreshold.length > 0 ? aboveThreshold : [];

    console.log(
      `[RAG] Retrieved ${all.length}/${matchCount} candidates; ${chunks.length} are >= similarity ${SIMILARITY_THRESHOLD}`
    );

    return {
      success: chunks.length > 0,
      chunks,
      nearbyChunks,
      scores: chunks.map((c) => c.similarity),
      bestSimilarity,
      avgTop3,
      embeddingDimensions,
    };
  } catch (err: any) {
    return { ...empty, error: err.message };
  }
}

function hasReliableContext(
  result: RagRetrievalResult,
  hasSubjectFilter: boolean
): boolean {
  if (!result.success || result.chunks.length === 0) return false;
  const minBest = hasSubjectFilter
    ? MIN_BEST_SIMILARITY_WITH_SUBJECT
    : MIN_BEST_SIMILARITY;
  const minAvg = hasSubjectFilter
    ? MIN_AVG_TOP3_WITH_SUBJECT
    : MIN_AVG_TOP3_SIMILARITY;
  return result.bestSimilarity >= minBest && result.avgTop3 >= minAvg;
}

function tokenizeForOverlap(text: string): string[] {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length >= 3);
}

function lexicalOverlapScore(question: string, content: string): number {
  const qTokens = Array.from(new Set(tokenizeForOverlap(question)));
  if (!qTokens.length) return 0;

  const cSet = new Set(tokenizeForOverlap(content));
  let hit = 0;
  for (const token of qTokens) {
    if (cSet.has(token)) hit += 1;
  }

  return hit / qTokens.length;
}

function hasRecoverableContext(question: string, result: RagRetrievalResult): boolean {
  const candidates = result.nearbyChunks.slice(0, 5);
  if (!candidates.length) return false;

  const bestOverlap = candidates.reduce((best, chunk) => {
    const overlap = lexicalOverlapScore(question, chunk.content || "");
    return Math.max(best, overlap);
  }, 0);

  return result.bestSimilarity >= 0.32 && bestOverlap >= 0.18;
}

function buildCitations(
  chunks: PastPaperChunk[],
  relation: "direct" | "nearby" = "direct"
): Citation[] {
  const seen = new Set<string>();
  return chunks
    .map((c) => {
      const key = `${c.subject}|${c.year}|${c.session}|${c.paper}|${c.variant}`;
      if (seen.has(key)) return null;
      seen.add(key);
      const meta = extractQuestionMeta(c.content);
      return {
        subject: c.subject,
        year: c.year,
        session: c.session,
        paper: c.paper,
        variant: c.variant,
        similarity: c.similarity,
        relation,
        questionNumber: meta.questionNumber,
        subQuestion: meta.subQuestion,
        marks: meta.marks,
        topicGeneral: meta.topicGeneral,
        topicSyllabus: meta.topicSyllabus,
      } as Citation;
    })
    .filter(Boolean) as Citation[];
}

function inferDominantSubject(
  chunks: PastPaperChunk[],
  takeTop: number = 8
): string | undefined {
  if (!chunks.length) return undefined;

  const bySimilarity = [...chunks]
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, takeTop);

  const weighted: Record<string, number> = {};
  for (const chunk of bySimilarity) {
    weighted[chunk.subject] = (weighted[chunk.subject] || 0) + chunk.similarity;
  }

  return Object.entries(weighted).sort((a, b) => b[1] - a[1])[0]?.[0];
}

function isMarkSchemeRequest(question: string): boolean {
  return /(mark(?:ing)?\s*scheme|mark\s*breakdown|mark\s*distribution|how\s+many\s+marks|award\s+marks?|marking\s+points?)/i.test(
    question
  );
}

const FOLLOWUP_PHRASES = [
  "tell me more",
  "explain more",
  "elaborate",
  "expand on",
  "what do you mean",
  "can you explain",
  "more about",
  "continue",
  "go on",
  "and then",
  "what about",
  "why is that",
  "how so",
  "give me an example",
  "more details",
  "in detail",
  "further",
  "what else",
  "anything else",
  "keep going",
  "also tell me",
];

const FOLLOWUP_ELABORATE_PHRASES = [
  "tell me more",
  "explain more",
  "elaborate",
  "expand",
  "more details",
  "in detail",
  "further",
  "continue",
  "go on",
  "what else",
  "anything else",
  "keep going",
];

const FOLLOWUP_EXAMPLE_PHRASES = [
  "give me an example",
  "for instance",
  "such as",
  "examples of",
  "example",
];

function containsAnyPhrase(text: string, phrases: string[]): boolean {
  const normalized = text.toLowerCase();
  return phrases.some((phrase) => normalized.includes(phrase));
}

function hasExplicitTaskVerb(question: string): boolean {
  return /\b(write|explain|define|describe|calculate|find|solve|work out|compare|contrast|list|state|name|give|summari[sz]e|draw|sketch|label|essay|speech|letter|report|article|summary|comprehension|diagram|mark(?:ing)?\s*scheme)\b/i.test(
    question
  );
}

function hasExplicitNewTopicCue(question: string): boolean {
  return /\b(now|new topic|different topic|another topic|instead|switch to|change topic)\b/i.test(
    question
  );
}

function isWhatAboutNewTopic(question: string): boolean {
  const normalized = question.toLowerCase().trim();
  if (!normalized.startsWith("what about ")) return false;

  const remainder = normalized.replace(/^what about\s+/, "").trim();
  if (!remainder) return false;
  return !/\b(it|this|that|same|above|previous|last|earlier|that question|this question)\b/i.test(
    remainder
  );
}

function looksLikeFollowUpReference(question: string): boolean {
  const normalized = question.toLowerCase().trim();
  if (!normalized) return false;

  if (/\b(same|above|previous|last|earlier)\b/i.test(normalized)) {
    return true;
  }

  if (
    /\b(this|that)\s+(question|topic|answer|response|one|paper|context)\b/i.test(
      normalized
    )
  ) {
    return true;
  }

  if (/\b(about|on)\s+(this|that|it)\b/i.test(normalized)) {
    return true;
  }

  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length <= 8 && /\bit\b/i.test(normalized)) {
    return true;
  }

  return false;
}

function isLikelyFollowUpQuestion(question: string): boolean {
  if (isWhatAboutNewTopic(question)) return false;
  return (
    looksLikeFollowUpReference(question) ||
    containsAnyPhrase(question, FOLLOWUP_PHRASES)
  );
}

function isVagueSearchQuery(question: string): boolean {
  const normalized = question.trim();
  if (!normalized) return true;

  const words = normalized.split(/\s+/).filter(Boolean);
  if (words.length < 8 && !hasExplicitTaskVerb(normalized)) return true;

  return isLikelyFollowUpQuestion(normalized);
}

function looksLikeStandaloneExamQuestion(question: string): boolean {
  const normalized = question.trim();
  if (!normalized) return false;

  const hasExamTokens =
    /\b(question\s*\d+|q\.?\s*\d+|\d+\s*marks?|paper\s*\d+|variant\s*\d+|mark\s*scheme|o\s*-?level|igcse|gcse|cambridge)\b/i.test(
      normalized
    ) ||
    /\b(chemistry|physics|mathematics|maths|english|islamiyat|pakistan\s+studies)\b/i.test(
      normalized
    );

  const hasCommandWords =
    hasExplicitTaskVerb(normalized) ||
    SHORT_COMMAND_RE.test(normalized) ||
    /\b(explain|define|calculate|compare|contrast|state|list|describe|evaluate|discuss|justify)\b/i.test(
      normalized
    );

  const hasReasonableLength = normalized.split(/\s+/).filter(Boolean).length >= 7;

  return hasExamTokens || (hasCommandWords && hasReasonableLength);
}

function findLastExamUserQuestion(history: HistoryMessage[]): string | null {
  for (let i = history.length - 1; i >= 0; i -= 1) {
    const h = history[i];
    if (h.role !== "user") continue;
    const content = (h.content || "").trim();
    if (!content) continue;
    if (classifyIntent(content) === "smalltalk") continue;
    if (isMarkSchemeRequest(content) && looksLikeFollowUpReference(content)) {
      continue;
    }
    return content;
  }
  return null;
}

interface RagDecision {
  shouldSearch: boolean;
  followUpDetected: boolean;
  reason: string;
}

function evaluateRagDecision(
  userMessage: string,
  history: HistoryMessage[],
  answerMode: AnswerMode
): RagDecision {
  if (answerMode === "mark_scheme_only") {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "mark_scheme_mode_requires_retrieval",
    };
  }

  const trimmed = userMessage.trim();
  if (!trimmed) {
    return {
      shouldSearch: false,
      followUpDetected: false,
      reason: "empty_user_message",
    };
  }

  if (history.length === 0) {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "no_history_new_turn",
    };
  }

  if (looksLikeStandaloneExamQuestion(trimmed)) {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "standalone_exam_question_forces_retrieval",
    };
  }

  if (isWhatAboutNewTopic(trimmed)) {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "explicit_new_topic_what_about",
    };
  }

  const explicitTask = hasExplicitTaskVerb(trimmed);
  const explicitNewTopic = hasExplicitNewTopicCue(trimmed);
  const followUpLikely = isLikelyFollowUpQuestion(trimmed);
  const hasReferencePronoun = looksLikeFollowUpReference(trimmed);
  const wordCount = trimmed.split(/\s+/).filter(Boolean).length;

  if (explicitTask && explicitNewTopic) {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "explicit_task_and_new_topic",
    };
  }

  if (explicitTask && !followUpLikely) {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "explicit_task_without_followup_reference",
    };
  }

  if (followUpLikely && explicitTask && !hasReferencePronoun && wordCount >= 6) {
    return {
      shouldSearch: true,
      followUpDetected: false,
      reason: "explicit_task_overrides_soft_followup_phrase",
    };
  }

  if (followUpLikely) {
    return {
      shouldSearch: false,
      followUpDetected: true,
      reason: "followup_reference_detected",
    };
  }

  if (wordCount < 6 && !explicitTask) {
    return {
      shouldSearch: false,
      followUpDetected: true,
      reason: "short_turn_assumed_followup",
    };
  }

  return {
    shouldSearch: true,
    followUpDetected: false,
    reason: "default_new_query",
  };
}

function shouldSearchRag(
  userMessage: string,
  history: HistoryMessage[],
  answerMode: AnswerMode
): boolean {
  return evaluateRagDecision(userMessage, history, answerMode).shouldSearch;
}

function rewriteQueryForRag(
  retrievalQuestion: string,
  userMessage: string,
  history: HistoryMessage[]
): { query: string; usedHistory: boolean } {
  const normalizedRetrieval = retrievalQuestion.trim();
  const normalizedUser = userMessage.trim();

  if (!normalizedRetrieval) {
    return { query: normalizedUser, usedHistory: false };
  }

  if (history.length === 0) {
    return { query: normalizedRetrieval, usedHistory: false };
  }

  // If retrieval was already rewritten by explicit resolver logic, keep it stable.
  if (normalizedRetrieval.toLowerCase() !== normalizedUser.toLowerCase()) {
    return { query: normalizedRetrieval, usedHistory: false };
  }

  if (!isVagueSearchQuery(normalizedUser)) {
    return { query: normalizedRetrieval, usedHistory: false };
  }

  const previousQuestion = findLastExamUserQuestion(history);
  if (!previousQuestion) {
    return { query: normalizedRetrieval, usedHistory: false };
  }

  const previousNormalized = previousQuestion.trim();
  if (!previousNormalized) {
    return { query: normalizedRetrieval, usedHistory: false };
  }

  if (previousNormalized.toLowerCase() === normalizedRetrieval.toLowerCase()) {
    return { query: normalizedRetrieval, usedHistory: false };
  }

  return {
    query: `${previousNormalized} ${normalizedRetrieval}`.trim(),
    usedHistory: true,
  };
}

function resolveQuestionForAnswering(
  question: string,
  history: HistoryMessage[]
): QuestionResolution {
  const trimmed = question.trim();
  if (!isMarkSchemeRequest(trimmed)) {
    return {
      retrievalQuestion: trimmed,
      answerMode: "full_answer",
      usedHistoryQuestion: false,
    };
  }

  const followUpLikely =
    looksLikeFollowUpReference(trimmed) || trimmed.split(/\s+/).length <= 8;

  if (followUpLikely) {
    const previousQuestion = findLastExamUserQuestion(history);
    if (previousQuestion) {
      return {
        retrievalQuestion: previousQuestion,
        answerMode: "mark_scheme_only",
        usedHistoryQuestion: true,
      };
    }
  }

  return {
    retrievalQuestion: trimmed,
    answerMode: "mark_scheme_only",
    usedHistoryQuestion: false,
  };
}

// -- INTENT ------------------------------------------------------------------
function normalizeIntentText(question: string): string {
  return question
    .toLowerCase()
    .replace(/[^a-z0-9\s']/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function classifyIntent(question: string): Intent {
  const q = normalizeIntentText(question);

  const smalltalkExact = new Set([
    "hello",
    "hi",
    "hey",
    "yo",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "yes",
    "no",
    "sure",
    "good",
    "great",
    "nice",
    "cool",
    "bye",
    "goodbye",
    "good night",
    "gn",
    "lol",
    "haha",
    "hmm",
    "umm",
    "hru",
    "sup",
    "how are you",
    "how r u",
    "how are u",
    "how you doing",
    "what's up",
    "whats up",
    "who are you",
    "are you there",
    "good morning",
    "good afternoon",
    "good evening",
    "hello how are you",
    "hey how are you",
    "hi how are you",
  ]);
  if (smalltalkExact.has(q)) return "smalltalk";

  if (
    /\bhow\s+(are|r)\s+(you|u)\b/.test(q) &&
    q.split(" ").length <= 8 &&
    !hasExplicitTaskVerb(question)
  ) {
    return "smalltalk";
  }

  if (
    /\b(hello|hi|hey)\b/.test(q) &&
    q.split(" ").length <= 6 &&
    !hasExplicitTaskVerb(question)
  ) {
    return "smalltalk";
  }

  if (
    q.split(" ").length <= 4 &&
    /^(hi+|hello+|hey+|ok(ay)?|thanks?|cool|nice|great|good|bye)$/.test(q)
  ) {
    return "smalltalk";
  }
  return "exam_question";
}

const smalltalkResponses = [
  "Hi! Ask me any O-Level exam question - Chemistry, Physics, Mathematics, English, Islamiyat, Pakistan Studies and more. I'll answer using real past exam papers.",
  "Hello! Try asking something like \"explain Newton's second law\" or \"what is oxidation in chemistry\".",
  "Hey! I can answer O-Level exam questions with mark-scheme style answers from past papers. What do you need?",
];

function getSmallTalkResponse(): string {
  return smalltalkResponses[Math.floor(Math.random() * smalltalkResponses.length)];
}

// -- ANSWER STYLE ------------------------------------------------------------
interface AnswerStyle {
  detailLevel: "short" | "long";
  responseShape: "standard" | "directed_writing";
  requiresHeadline: boolean;
  preferHeadings: boolean;
}

type QuestionType =
  | "essay"
  | "speech"
  | "letter"
  | "report"
  | "article"
  | "summary"
  | "comprehension"
  | "explanation"
  | "example"
  | "list"
  | "comparison"
  | "calculation"
  | "definition"
  | "diagram"
  | "followup_elaborate"
  | "followup_example";

interface QuestionTypeDetection {
  questionType: QuestionType;
  marks?: number;
}

interface VisualNeeds {
  table: boolean;
  ascii: boolean;
  graph: boolean;
  timeline: boolean;
  flowchart: boolean;
}

const QUESTION_TYPE_PATTERNS: Array<{ type: QuestionType; keywords: string[] }> = [
  {
    type: "essay",
    keywords: [
      "write an essay",
      "write a composition",
      "composition on",
      "essay on",
    ],
  },
  {
    type: "speech",
    keywords: [
      "write a speech",
      "speech about",
      "speech on",
      "give a speech",
      "deliver a speech",
    ],
  },
  {
    type: "letter",
    keywords: [
      "write a letter",
      "letter to",
      "formal letter",
      "informal letter",
      "write to",
    ],
  },
  {
    type: "report",
    keywords: ["write a report", "report on", "prepare a report"],
  },
  {
    type: "article",
    keywords: [
      "write an article",
      "article about",
      "article on",
      "newspaper article",
      "magazine article",
    ],
  },
  {
    type: "summary",
    keywords: [
      "summarise",
      "summarize",
      "write a summary",
      "summary of",
      "in summary",
    ],
  },
  {
    type: "comprehension",
    keywords: [
      "according to the passage",
      "from the passage",
      "what does the writer",
      "answer the question",
    ],
  },
  {
    type: "example",
    keywords: ["give an example", "examples of", "for example"],
  },
  {
    type: "list",
    keywords: ["list", "mention", "name", "state", "types of"],
  },
  {
    type: "comparison",
    keywords: [
      "compare",
      "difference between",
      "similarities",
      "contrast",
      "versus",
      " vs ",
    ],
  },
  {
    type: "calculation",
    keywords: [
      "calculate",
      "find the",
      "solve",
      "work out",
      "value of",
      "compute",
    ],
  },
  {
    type: "definition",
    keywords: ["define", "what is meant by", "meaning of", "definition of"],
  },
  {
    type: "diagram",
    keywords: ["draw", "sketch", "label", "diagram"],
  },
  {
    type: "explanation",
    keywords: [
      "explain",
      "what is",
      "what are",
      "describe",
      "tell me about",
      "how does",
      "why is",
    ],
  },
];

function detectQuestionType(
  userMessage: string,
  contextChunks: ContextChunkForPrompt[] = [],
  history: HistoryMessage[] = []
): QuestionTypeDetection {
  const msgLower = userMessage.toLowerCase().trim();
  const detected: QuestionType[] = [];

  const marks = contextChunks.find(
    (chunk) => Number.isFinite(chunk.marks) && Number(chunk.marks) > 0
  )?.marks;

  for (const pattern of QUESTION_TYPE_PATTERNS) {
    if (containsAnyPhrase(msgLower, pattern.keywords)) {
      detected.push(pattern.type);
    }
  }

  const top = contextChunks[0];
  const paperType = (top?.topicGeneral || "").toLowerCase();
  const syllabusType = (top?.topicSyllabus || "").toLowerCase();

  if (paperType.includes("creative writing")) detected.unshift("essay");
  if (paperType.includes("directed writing")) detected.unshift("letter");
  if (paperType.includes("summary")) detected.unshift("summary");
  if (paperType.includes("comprehension")) detected.unshift("comprehension");

  if (syllabusType.includes("letter")) detected.unshift("letter");
  if (syllabusType.includes("report")) detected.unshift("report");
  if (syllabusType.includes("article")) detected.unshift("article");
  if (syllabusType.includes("narrative")) detected.unshift("essay");
  if (syllabusType.includes("speech")) detected.unshift("speech");
  if (syllabusType.includes("summary")) detected.unshift("summary");

  if (history.length > 0) {
    const followUpLikely = isLikelyFollowUpQuestion(userMessage);

    if (
      followUpLikely &&
      containsAnyPhrase(msgLower, FOLLOWUP_EXAMPLE_PHRASES)
    ) {
      detected.unshift("followup_example");
    } else if (
      (followUpLikely && !hasExplicitTaskVerb(userMessage)) ||
      (containsAnyPhrase(msgLower, FOLLOWUP_ELABORATE_PHRASES) &&
        !hasExplicitTaskVerb(userMessage) &&
        !isWhatAboutNewTopic(userMessage))
    ) {
      detected.unshift("followup_elaborate");
    }
  }

  return {
    questionType: detected[0] || "explanation",
    marks,
  };
}

function getWordGuideFromMarks(marks?: number): string {
  if (!Number.isFinite(marks) || Number(marks) <= 0) return "";
  const safeMarks = Math.floor(Number(marks));
  if (safeMarks <= 2) return "Keep answer concise: 1-3 sentences.";
  if (safeMarks <= 5) return "Write 1-2 short paragraphs (50-100 words).";
  if (safeMarks <= 10) return "Write 2-3 paragraphs (100-200 words).";
  if (safeMarks <= 20) return "Write 3-4 paragraphs (200-350 words).";
  return "Write a full response (350-500 words).";
}

function getQuestionTypeFormatInstructions(
  questionType: QuestionType,
  marks?: number
): string {
  const wordGuide = getWordGuideFromMarks(marks);

  const formatByType: Record<QuestionType, string> = {
    essay: `FORMAT: Full Essay / Composition
- Start with an engaging introduction paragraph.
- Develop 2-3 clear body paragraphs.
- End with a conclusion that ties ideas together.
- Use varied vocabulary and sentence structures.
- No bullet points; use flowing prose.`,
    speech: `FORMAT: Formal Speech
- Start with a greeting (for example: "Good morning, ladies and gentlemen,").
- Use a clear opening statement to hook the audience.
- Build 2-3 main points in separate paragraphs.
- Use rhetorical devices naturally where suitable.
- End with a strong close (for example: "Thank you.").
- No bullet points; use spoken-flow prose.`,
    letter: `FORMAT: Formal/Informal Letter
- Use proper letter structure with address/date/salutation/body/closing/sign-off.
- Match tone to audience (formal or informal).
- Keep body in connected paragraphs, not bullets.`,
    report: `FORMAT: Formal Report
- Use sections: Introduction, Findings, Recommendations/Conclusion.
- Keep formal, objective tone.
- Use subheadings and numbered points where useful.`,
    article: `FORMAT: Newspaper/Magazine Article
- Include a strong headline and byline when suitable.
- Start with a lead paragraph (who/what/where/when/why).
- Expand with clear body paragraphs and a concise conclusion.`,
    summary: `FORMAT: Summary Writing
- Start with one sentence stating what is being summarized.
- Use your own words and include only key points.
- Keep a logical order and avoid unnecessary detail.
- No bullet points unless the user explicitly requests list format.`,
    comprehension: `FORMAT: Comprehension Answer
- Answer directly and specifically.
- Use evidence from the provided text where relevant.
- Keep wording precise and aligned to marks.`,
    explanation: `FORMAT: Clear Explanation
- Start with a direct answer.
- Follow with supporting reasoning.
- Add one concrete example where helpful.
- Keep each paragraph focused on one idea.`,
    example: `FORMAT: Example Response
- Give 1-3 concrete examples.
- Briefly explain why each example is relevant.
- Avoid repeating the same point with different wording.`,
    list: `FORMAT: Listed Answer
- Use numbered lists (1. 2. 3.) when sequence matters.
- Use bullets when order does not matter.
- Keep each point clear and self-contained.`,
    comparison: `FORMAT: Comparison Answer
- Compare point-by-point or in two clear blocks.
- Use comparison language: whereas, however, similarly, in contrast.
- Cover both similarities and differences if asked.`,
    calculation: `FORMAT: Mathematical/Scientific Working
- Show working step by step.
- State formula first, then substitute values.
- Keep units visible throughout.
- State final answer clearly with correct units/significant figures.`,
    definition: `FORMAT: Definition Answer
- Start with a precise one-sentence definition.
- Add a short elaboration only if marks require it.
- Include one quick example when useful.`,
    diagram: `FORMAT: Diagram Response
- Provide clear labels and ordered steps for what to draw/annotate.
- Emphasize key components and relationships.
- Keep instructions concise and exam-focused.`,
    followup_elaborate: `FORMAT: Follow-up Elaboration
- Continue naturally from previous response.
- Do not restart from scratch or repeat unchanged points.
- Add deeper detail, clearer reasoning, or another perspective.`,
    followup_example: `FORMAT: Follow-up Example
- Continue from prior response.
- Add concrete examples that clarify earlier points.
- Keep examples specific and concise.`,
  };

  return [formatByType[questionType], wordGuide].filter(Boolean).join("\n");
}

function detectVisualNeeds(
  userMessage: string,
  contextChunks: ContextChunkForPrompt[] = []
): VisualNeeds {
  const lower = userMessage.toLowerCase();
  const needs: VisualNeeds = {
    table: false,
    ascii: false,
    graph: false,
    timeline: false,
    flowchart: false,
  };

  const tableKeywords = [
    "compare",
    "comparison",
    "difference between",
    "similarities",
    "contrast",
    "versus",
    "pros and cons",
    "advantages and disadvantages",
    "types of",
    "properties of",
    "characteristics",
    "summary table",
    "tabulate",
    "in a table",
  ];

  const asciiKeywords = [
    "draw",
    "diagram",
    "sketch",
    "label",
    "structure of",
    "illustrate",
    "circuit",
    "apparatus",
    "food chain",
    "food web",
    "cycle",
    "cross section",
    "cross-section",
    "anatomy",
  ];

  const graphKeywords = [
    "graph",
    "plot",
    "chart",
    "axes",
    "x-axis",
    "y-axis",
    "distance time",
    "speed time",
    "velocity",
  ];

  const timelineKeywords = [
    "timeline",
    "chronological",
    "sequence of events",
    "history of",
    "dates of",
    "order of events",
  ];

  const flowchartKeywords = [
    "flowchart",
    "flow chart",
    "steps to",
    "process",
    "procedure",
    "algorithm",
    "decision",
  ];

  if (tableKeywords.some((kw) => lower.includes(kw))) needs.table = true;
  if (asciiKeywords.some((kw) => lower.includes(kw))) needs.ascii = true;
  if (graphKeywords.some((kw) => lower.includes(kw))) needs.graph = true;
  if (timelineKeywords.some((kw) => lower.includes(kw))) needs.timeline = true;
  if (flowchartKeywords.some((kw) => lower.includes(kw))) needs.flowchart = true;

  for (const chunk of contextChunks.slice(0, 2)) {
    const syllabus = (chunk.topicSyllabus || "").toLowerCase();
    const general = (chunk.topicGeneral || "").toLowerCase();
    const contextText = `${syllabus} ${general}`;

    if (
      /\b(atomic|circuit|wave|force|cell|organ|reaction|structure|apparatus)\b/.test(
        contextText
      )
    ) {
      needs.ascii = true;
    }

    if (/\b(comparison|classification|properties)\b/.test(contextText)) {
      needs.table = true;
    }
  }

  return needs;
}

function listActiveVisualNeeds(needs: VisualNeeds): string[] {
  return Object.entries(needs)
    .filter(([, isActive]) => isActive)
    .map(([type]) => type);
}

function getVisualFormatInstructions(needs: VisualNeeds): string {
  const instructions: string[] = [];

  if (needs.table) {
    instructions.push(`TABLE FORMATTING RULES:
- Use markdown pipe tables only.
- Include a clear caption line above the table, for example: **Table: Key Comparison**.
- Keep cells concise and aligned.
- For comparisons, use first column as Feature.`);
  }

  if (needs.ascii) {
    instructions.push(`ASCII DIAGRAM RULES:
- Wrap every diagram in a fenced code block.
- Keep diagram width within about 60 characters.
- Label all parts clearly.
- Add a caption below, for example: *Figure: Cell structure*.`);
  }

  if (needs.graph) {
    instructions.push(`GRAPH RULES:
- Draw graph as ASCII inside a fenced code block.
- Label X and Y axes with units.
- Mark important points clearly.
- Add a caption below, for example: *Graph: Distance-time relationship*.`);
  }

  if (needs.timeline) {
    instructions.push(`TIMELINE RULES:
- Draw timeline in a fenced code block.
- Keep event labels short and chronological.
- Add a caption below, for example: *Timeline: Major events*.`);
  }

  if (needs.flowchart) {
    instructions.push(`FLOWCHART RULES:
- Draw flowchart in a fenced code block.
- Use clear step boxes and arrows.
- Include decision branches when needed.
- Add a caption below, for example: *Flowchart: Process steps*.`);
  }

  return instructions.join("\n\n");
}

// Classic IGCSE/GCSE short-answer command words that expect 1-3 sentence replies
const SHORT_COMMAND_RE = /^\s*(state|name|give|identify|write\s+down|list|define|state\s+one|state\s+two|state\s+three|give\s+one|give\s+two|name\s+one|name\s+two|what\s+is\s+meant\s+by|what\s+are\s+the\s+units)\b/i;

function isDirectedWritingTask(question: string): boolean {
  return /(write\s+(an?|the)?\s*(article|letter|speech|report|email|blog|story|narrative|journal)|school\s+magazine|suitable\s+headline|tone\s+and\s+register|specified\s+situation\s+and\s+audience|task\s+fulfilment|directed\s+writing)/i.test(
    question
  );
}

function requiresHeadlineForTask(question: string): boolean {
  return /\bheadline\b|school\s+magazine|write\s+an\s+article/i.test(question);
}

function isLikelyExamPrompt(question: string): boolean {
  return (
    hasExplicitTaskVerb(question) ||
    /\b(o\s*-?level|igcse|gcse|cambridge|past\s*paper|mark\s*scheme|marks?|question\s*\d+|paper\s*\d+|variant|syllabus|topic)\b/i.test(
      question
    ) ||
    /\b(chemistry|physics|mathematics|maths|english|islamiyat|pakistan\s+studies)\b/i.test(
      question
    )
  );
}

function isSimplePrompt(question: string): boolean {
  const trimmed = question.trim();
  const words = trimmed.split(/\s+/).filter(Boolean);
  return words.length <= 10 && !isLikelyExamPrompt(trimmed);
}

function inferAnswerStyle(
  question: string,
  answerMode: AnswerMode = "full_answer"
): AnswerStyle {
  const directedWriting = isDirectedWritingTask(question);
  const requiresHeadline = requiresHeadlineForTask(question);
  const likelyExamPrompt = isLikelyExamPrompt(question);

  if (answerMode === "mark_scheme_only") {
    return {
      detailLevel: "short",
      responseShape: directedWriting ? "directed_writing" : "standard",
      requiresHeadline,
      preferHeadings: true,
    };
  }

  const wantsShort =
    /(short|briefly|in short|concise)/i.test(question) ||
    SHORT_COMMAND_RE.test(question);

  const conversationalTone =
    /\b(how\s+are\s+you|who\s+are\s+you|what\s+can\s+you\s+do|tell\s+me\s+about\s+yourself|thank\s+you|thanks)\b/i.test(
      question
    );

  return {
    detailLevel: wantsShort || conversationalTone || isSimplePrompt(question) ? "short" : "long",
    responseShape: directedWriting ? "directed_writing" : "standard",
    requiresHeadline,
    preferHeadings: likelyExamPrompt,
  };
}

function buildFormatInstruction(
  style: AnswerStyle,
  answerMode: AnswerMode = "full_answer"
): string {
  if (answerMode === "mark_scheme_only") {
    return `- Return ONLY the marking scheme for the question.
- Keep "answer" short and focused on assessable criteria, not full teaching notes.
- Use markdown heading ## Marking Scheme and concise bullets.`;
  }

  if (style.responseShape === "directed_writing") {
    return `- Write the final student-facing response in the requested text type (e.g., article/letter/speech), not as tutor notes.
- Do NOT use meta sections like Core Idea, Explanation, Key Points, Exam Tip, or Quick Summary.
- Cover every required bullet/point from the prompt in full detail.
- Keep tone, register, and audience appropriate to the task.
- Use natural paragraphs and clear flow.
- Do NOT use HTML tags such as <h1>, <p>, <br>, <ul>, or <li>.
- ${
      style.requiresHeadline
        ? "Start with a suitable headline before the body."
        : "Use an appropriate opening for the requested format."
    }
- Do not output planning notes; output the final polished writing piece.`;
  }

  if (style.detailLevel === "short") {
    return style.preferHeadings
      ? "- Keep the answer concise (2-4 lines), exam-focused."
      : "- Keep the answer concise (2-4 lines) and natural in tone.";
  }

  if (!style.preferHeadings) {
    return `- Write a natural, direct response in plain markdown.
- Use headings only when they clearly improve readability.
- Do not force fixed templates and do not repeat the same sentence across sections.
- Keep response length proportional to the question.`;
  }

  return `- Write a complete, detailed answer in markdown.
- Choose section headings intelligently based on the question type; do NOT force one fixed template every time.
- For quantitative questions, use headings like: ## Given, ## Working, ## Final Answer.
- For theory / process / explain questions, use headings like: ## Core Idea, ## Explanation, ## Key Points.
- Use bullets only where they improve clarity.
- Write as much as the question genuinely needs — never pad sections with repeated content.`;
}

function buildMaxTokens(
  style: AnswerStyle,
  withContext: boolean,
  answerMode: AnswerMode = "full_answer"
): number {
  if (answerMode === "mark_scheme_only") return withContext ? 1200 : 900;
  if (style.responseShape === "directed_writing") return withContext ? 3200 : 2400;
  if (style.detailLevel === "short") return withContext ? 900 : 800;
  return withContext ? 2600 : 2200;
}

// -- TEXT HELPERS ------------------------------------------------------------
function decodeJsonLikeString(text: string): string {
  return text
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"')
    .trim();
}

function isTemplateAnswer(text: string): boolean {
  const snippets = [
    "write the actual answer content here",
    "write the real answer here",
    "detailed, properly formatted answer",
  ];
  return !text.trim() || snippets.some((s) => text.toLowerCase().includes(s));
}

function extractLooseJsonStringField(
  text: string,
  fieldName: string
): string | undefined {
  const keyPattern = new RegExp(`"${escapeRegExp(fieldName)}"\\s*:\\s*"`, "i");
  const keyMatch = keyPattern.exec(text);
  if (!keyMatch) return undefined;

  let i = keyMatch.index + keyMatch[0].length;
  let raw = "";
  let escaped = false;

  for (; i < text.length; i += 1) {
    const ch = text[i];

    if (escaped) {
      raw += ch;
      escaped = false;
      continue;
    }

    if (ch === "\\") {
      raw += ch;
      escaped = true;
      continue;
    }

    if (ch === '"') break;
    raw += ch;
  }

  const decoded = decodeJsonLikeString(raw);
  return decoded || undefined;
}

function extractLooseAnswerFromJsonish(text: string): string | undefined {
  const direct = extractLooseJsonStringField(text, "answer");
  if (direct) return direct;

  const blockMatch = text.match(
    /"answer"\s*:\s*([\s\S]*?)(?=\n\s*"(?:marking_points|common_mistakes)"\s*:|\}\s*$)/i
  );
  if (!blockMatch) return undefined;

  const raw = blockMatch[1]
    .trim()
    .replace(/^"/, "")
    .replace(/"\s*,?\s*$/, "")
    .trim();
  const decoded = decodeJsonLikeString(raw);
  return decoded || undefined;
}

function cleanAnswerText(text: string): string {
  const cleaned = text
    .replace(/^```(?:json|markdown)?\s*/i, "")
    .replace(/```\s*$/i, "")
    .trim();
  if (!cleaned) return "";

  try {
    const parsed = JSON.parse(cleaned);
    if (parsed?.answer) return decodeJsonLikeString(String(parsed.answer));
  } catch {
    // ignore parse error
  }

  const m = cleaned.match(/\{[\s\S]*\}/);
  if (m) {
    try {
      const p = JSON.parse(m[0]);
      if (p?.answer) return decodeJsonLikeString(String(p.answer));
    } catch {
      // ignore parse error
    }

    const looseFromObject = extractLooseAnswerFromJsonish(m[0]);
    if (looseFromObject) return looseFromObject;
  }

  const loose = extractLooseAnswerFromJsonish(cleaned);
  if (loose) return loose;

  if (cleaned.startsWith("{") || cleaned.includes('"answer"')) return "";
  return cleaned;
}

function escapeRegExp(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function stripRepeatedHeadingEcho(markdown: string): string {
  const lines = markdown.split("\n");
  for (let i = 0; i < lines.length - 1; i += 1) {
    const heading = lines[i].match(/^##\s+(.+)$/);
    if (!heading) continue;
    const title = heading[1].trim();
    if (!title) continue;
    const pattern = new RegExp(`^${escapeRegExp(title)}\\s*[:.-]?\\s*`, "i");
    if (pattern.test(lines[i + 1].trim())) {
      lines[i + 1] = lines[i + 1].replace(pattern, "");
    }
  }
  return lines.join("\n");
}

type HeadingProfile = "quantitative" | "comparison" | "definition" | "conceptual";

function stripDefaultScaffoldHeadings(text: string): string {
  return text
    .replace(/^##\s*(Core Idea|Explanation|Key Points|Exam Tip|Quick Summary)\s*$/gim, "")
    .replace(/^##\s*(Introduction|Final Answer|Comparison Focus|Key Differences|Definition|Given|Working)\s*$/gim, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function normalizePlainResponse(text: string): string {
  return stripDefaultScaffoldHeadings(text)
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function detectHeadingProfile(text: string): HeadingProfile {
  const plain = text.toLowerCase();
  if (/\b(compare|contrast|difference|distinguish|similarities|similarity)\b/.test(plain)) {
    return "comparison";
  }
  if (/\b(define|definition|what is|state the meaning)\b/.test(plain)) {
    return "definition";
  }
  if (
    /\b(calculate|solve|find|equation|formula|discriminant|velocity|acceleration|ratio|probability|m\/s|cm|kg|mol|newton)\b/.test(
      plain
    ) ||
    /[=<>]|\d/.test(plain)
  ) {
    return "quantitative";
  }
  return "conceptual";
}

function enforceMarkdownStructure(answer: string, style: AnswerStyle): string {
  const cleaned = stripRepeatedHeadingEcho(
    decodeJsonLikeString(answer || "").trim()
  );
  if (!cleaned) return buildFallbackAnswer(style);

  if (style.responseShape === "directed_writing") {
    const stripped = stripDefaultScaffoldHeadings(cleaned);
    const normalized = stripped
      .replace(/<\/?h[1-6][^>]*>/gi, "")
      .replace(/<\/?p[^>]*>/gi, "")
      .replace(/<br\s*\/?>/gi, "\n")
      .replace(/<\/?(?:ul|ol|li)[^>]*>/gi, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    if (!normalized) return buildFallbackAnswer(style);

    if (!style.requiresHeadline) return normalized;

    const firstNonEmptyLine =
      normalized.split("\n").find((line) => line.trim().length > 0)?.trim() || "";
    const looksLikeHeadline =
      firstNonEmptyLine.length > 0 &&
      firstNonEmptyLine.length <= 90 &&
      !/[.!?]$/.test(firstNonEmptyLine) &&
      !/^[-*\d]+[.)\]]/.test(firstNonEmptyLine);

    if (looksLikeHeadline) return normalized;

    // If the model wrote headline + body in one line, split at the first sentence.
    if (!normalized.includes("\n")) {
      const sentenceBreak = normalized.indexOf(". ");
      if (sentenceBreak > 8 && sentenceBreak < 90) {
        const headline = normalized.slice(0, sentenceBreak + 1).trim();
        const body = normalized.slice(sentenceBreak + 2).trim();
        if (headline && body) return `${headline}\n\n${body}`;
      }
    }

    return normalized;
  }

  if (!style.preferHeadings) {
    const plain = normalizePlainResponse(cleaned);
    return plain || buildFallbackAnswer(style);
  }

  const headingCount = (cleaned.match(/(^|\n)##\s+[^\n]+/g) || []).length;
  if (headingCount >= 2) return cleaned;
  if (style.detailLevel === "short" && cleaned.length <= 420) return cleaned;

  const plain = cleaned.replace(/^\s*#{1,6}\s+/gm, "").trim();
  if (!plain) return buildFallbackAnswer(style);

  if (style.detailLevel === "short") return plain;

  const sentences = (plain.replace(/\n+/g, " ").match(/[^.!?]+[.!?]?/g) || [plain])
    .map((s) => s.trim())
    .filter(Boolean);

  const intro = sentences.slice(0, 2).join(" ") || plain;
  const points = sentences.slice(2, 8);
  const summary = sentences[sentences.length - 1] || intro;
  const profile = detectHeadingProfile(plain);

  if (profile === "quantitative") {
    return `## Given
${intro}

## Working
${points.length ? points.join("\n\n") : plain}

## Final Answer
${summary}`;
  }

  if (profile === "comparison") {
    return `## Comparison Focus
${intro}

## Key Differences
${(points.length ? points : [plain]).map((p) => `- ${p}`).join("\n")}

## Exam Tip
Use explicit comparison words such as "whereas", "however", and "in contrast".

## Quick Summary
${summary}`;
  }

  if (profile === "definition") {
    return `## Definition
${intro}

## Explanation
${points.length ? points.join("\n\n") : plain}

## Exam Tip
State the exact definition first, then add one clear supporting detail.

## Quick Summary
${summary}`;
  }

  return `## Core Idea
${intro}

## Explanation
${points.length ? points.join("\n\n") : plain}

## Key Points
${(points.length ? points : [plain]).map((p) => `- ${p}`).join("\n")}

## Exam Tip
Use precise subject vocabulary and link each point to what examiners award marks for.

## Quick Summary
${summary}`;
}

function buildFallbackAnswer(style: AnswerStyle): string {
  if (style.responseShape === "directed_writing") {
    return style.requiresHeadline
      ? "A Homecoming That Inspires\n\nMy cousin was one of the most successful students at our school, known for excellent grades, leadership in activities, and a positive attitude towards others. After school, my cousin moved abroad, studied and worked hard, and gained valuable international experience. Living in another country taught independence, discipline, and respect for different cultures.\n\nToday, my cousin is a role model for students because success was achieved through consistency, humility, and service. Instead of showing off, my cousin shares practical advice with younger students: set clear goals, manage time well, and keep learning from every challenge. This journey proves that with effort and character, students from our school can succeed anywhere in the world."
      : "I could not generate the full response in the required writing format. Please ask again.";
  }

  if (!style.preferHeadings) {
    return "I can help with that. Share the exact topic or question and I will answer directly and clearly.";
  }

  return `## Introduction\nHere is a clear response to your question.\n\n## Key Points\n- Key idea 1.\n- Key idea 2.\n\n## Exam Tip\n${
    style.detailLevel === "short"
      ? "Keep answers concise and use key terms."
      : "Use headings and support each point with evidence."
  }\n\n## Quick Summary\nReview the key points above.`;
}

interface ContextChunkForPrompt {
  subject: string;
  year: number;
  session: string;
  paper: string;
  variant: string;
  similarity: number;
  questionNumber?: string;
  subQuestion?: string;
  marks?: number;
  topicGeneral?: string;
  topicSyllabus?: string;
  questionPart?: string;
  markSchemePart?: string;
  rawContent?: string;
}

function normalizeFingerprintPart(value: string | undefined): string {
  return (value || "").toLowerCase().replace(/\s+/g, " ").trim();
}

function extractHeaderQuestionLine(content: string): string | undefined {
  const match = content.match(/(?:^|\n)\s*Question\s+(?!:)([^\n]+)/i);
  return match?.[1]?.trim();
}

function extractTopicLine(content: string): string | undefined {
  const match = content.match(/(?:^|\n)\s*Topic:\s*([^\n]+)/i);
  return match?.[1]?.trim();
}

function extractTopicParts(content: string): {
  topicGeneral?: string;
  topicSyllabus?: string;
} {
  const raw = extractTopicLine(content);
  if (!raw) return {};

  const parts = raw
    .split(">")
    .map((part) => part.trim())
    .filter(Boolean);

  if (parts.length >= 2) {
    return {
      topicGeneral: parts[0],
      topicSyllabus: parts.slice(1).join(" > "),
    };
  }

  return { topicGeneral: parts[0] };
}

function extractQuestionMeta(content: string): {
  questionNumber?: string;
  subQuestion?: string;
  marks?: number;
  topicGeneral?: string;
  topicSyllabus?: string;
} {
  const header = extractHeaderQuestionLine(content);
  const topic = extractTopicParts(content);

  if (!header) {
    return topic;
  }

  const marksMatch = header.match(/\[(\d+)\s*marks?\]/i);
  const marksValue = marksMatch ? Number(marksMatch[1]) : undefined;

  const headerWithoutMarks = header.replace(/\[[^\]]+\]/g, "").trim();
  const idMatch = headerWithoutMarks.match(/^(\S+)(?:\s+(.+))?$/);

  return {
    ...topic,
    questionNumber: idMatch?.[1]?.trim() || undefined,
    subQuestion: idMatch?.[2]?.trim() || undefined,
    marks: Number.isFinite(marksValue) ? marksValue : undefined,
  };
}

function extractQuestionBody(content: string): string | undefined {
  const match = content.match(
    /(?:^|\n)\s*Question\s*:\s*([\s\S]*?)(?=(?:\n\s*Mark\s*Scheme\s*:)|$)/i
  );
  const text = match?.[1]?.trim();
  return text || undefined;
}

function extractMarkSchemeBody(content: string): string | undefined {
  const match = content.match(/(?:^|\n)\s*Mark\s*Scheme\s*:\s*([\s\S]*)$/i);
  const text = match?.[1]?.trim();
  return text || undefined;
}

function buildChunkFingerprint(chunk: PastPaperChunk): string | null {
  const questionHeader = normalizeFingerprintPart(
    extractHeaderQuestionLine(chunk.content)
  );
  if (!questionHeader) return null;

  const topic = normalizeFingerprintPart(extractTopicLine(chunk.content));

  return [
    normalizeFingerprintPart(chunk.subject),
    String(chunk.year),
    normalizeFingerprintPart(chunk.session),
    normalizeFingerprintPart(chunk.paper),
    normalizeFingerprintPart(chunk.variant),
    questionHeader,
    topic,
  ].join("|");
}

function toContextChunk(chunk: PastPaperChunk): ContextChunkForPrompt {
  const meta = extractQuestionMeta(chunk.content);
  return {
    subject: chunk.subject,
    year: chunk.year,
    session: chunk.session,
    paper: chunk.paper,
    variant: chunk.variant,
    similarity: chunk.similarity,
    questionNumber: meta.questionNumber,
    subQuestion: meta.subQuestion,
    marks: meta.marks,
    topicGeneral: meta.topicGeneral,
    topicSyllabus: meta.topicSyllabus,
    questionPart: extractQuestionBody(chunk.content),
    markSchemePart: extractMarkSchemeBody(chunk.content),
    rawContent: chunk.content,
  };
}

function mergeSplitChunksForContext(
  chunks: PastPaperChunk[],
  contextLimit: number
): ContextChunkForPrompt[] {
  type MergeBucket = {
    combined?: PastPaperChunk;
    questionOnly?: PastPaperChunk;
    markSchemeOnly?: PastPaperChunk;
  };

  const sorted = [...chunks].sort((a, b) => b.similarity - a.similarity);
  const candidates = sorted.slice(0, Math.max(contextLimit * 3, contextLimit + 6));
  const buckets = new Map<string, MergeBucket>();
  const passthrough: PastPaperChunk[] = [];

  for (const chunk of candidates) {
    const questionPart = extractQuestionBody(chunk.content);
    const markSchemePart = extractMarkSchemeBody(chunk.content);
    const hasQuestion = Boolean(questionPart);
    const hasMarkScheme = Boolean(markSchemePart);

    if (!hasQuestion && !hasMarkScheme) {
      passthrough.push(chunk);
      continue;
    }

    const fingerprint = buildChunkFingerprint(chunk);
    if (!fingerprint) {
      // No reliable source signature -> do not attempt merge.
      passthrough.push(chunk);
      continue;
    }

    const bucket = buckets.get(fingerprint) || {};

    if (hasQuestion && hasMarkScheme) {
      if (!bucket.combined || chunk.similarity > bucket.combined.similarity) {
        bucket.combined = chunk;
      }
    } else if (hasQuestion) {
      if (!bucket.questionOnly || chunk.similarity > bucket.questionOnly.similarity) {
        bucket.questionOnly = chunk;
      }
    } else if (hasMarkScheme) {
      if (
        !bucket.markSchemeOnly ||
        chunk.similarity > bucket.markSchemeOnly.similarity
      ) {
        bucket.markSchemeOnly = chunk;
      }
    }

    buckets.set(fingerprint, bucket);
  }

  const merged: ContextChunkForPrompt[] = [];
  let mergedPairs = 0;

  for (const bucket of buckets.values()) {
    if (bucket.combined) {
      merged.push(toContextChunk(bucket.combined));
      continue;
    }

    if (bucket.questionOnly && bucket.markSchemeOnly) {
      const sameSource =
        bucket.questionOnly.subject === bucket.markSchemeOnly.subject &&
        bucket.questionOnly.year === bucket.markSchemeOnly.year &&
        bucket.questionOnly.session === bucket.markSchemeOnly.session &&
        bucket.questionOnly.paper === bucket.markSchemeOnly.paper &&
        bucket.questionOnly.variant === bucket.markSchemeOnly.variant;

      const questionPart = extractQuestionBody(bucket.questionOnly.content);
      const markSchemePart = extractMarkSchemeBody(bucket.markSchemeOnly.content);

      if (sameSource && questionPart && markSchemePart) {
        const meta = extractQuestionMeta(bucket.questionOnly.content);
        mergedPairs += 1;
        merged.push({
          subject: bucket.questionOnly.subject,
          year: bucket.questionOnly.year,
          session: bucket.questionOnly.session,
          paper: bucket.questionOnly.paper,
          variant: bucket.questionOnly.variant,
          similarity: Math.max(
            bucket.questionOnly.similarity,
            bucket.markSchemeOnly.similarity
          ),
          questionNumber: meta.questionNumber,
          subQuestion: meta.subQuestion,
          marks: meta.marks,
          topicGeneral: meta.topicGeneral,
          topicSyllabus: meta.topicSyllabus,
          questionPart,
          markSchemePart,
        });
        continue;
      }
    }

    if (bucket.questionOnly) merged.push(toContextChunk(bucket.questionOnly));
    if (bucket.markSchemeOnly) merged.push(toContextChunk(bucket.markSchemeOnly));
  }

  for (const chunk of passthrough) {
    merged.push(toContextChunk(chunk));
  }

  const seen = new Set<string>();
  const deduped: ContextChunkForPrompt[] = [];

  for (const item of merged.sort((a, b) => b.similarity - a.similarity)) {
    const key = [
      normalizeFingerprintPart(item.subject),
      String(item.year),
      normalizeFingerprintPart(item.session),
      normalizeFingerprintPart(item.paper),
      normalizeFingerprintPart(item.variant),
      normalizeFingerprintPart((item.questionPart || "").slice(0, 120)),
      normalizeFingerprintPart((item.markSchemePart || "").slice(0, 120)),
      normalizeFingerprintPart((item.rawContent || "").slice(0, 120)),
    ].join("|");

    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(item);

    if (deduped.length >= contextLimit) break;
  }

  if (mergedPairs > 0) {
    console.log(`[RAG] Merged ${mergedPairs} split question/mark-scheme pairs into unified context chunks`);
  }

  return deduped;
}

function truncateForPrompt(text: string | undefined, maxChars: number): string {
  const normalized = (text || "").trim();
  if (!normalized) return "";
  if (normalized.length <= maxChars) return normalized;
  return `${normalized.slice(0, maxChars).trim()}...`;
}

function formatSourceCitationFromContext(chunk: ContextChunkForPrompt): string {
  return `${chunk.subject} | ${chunk.year} ${chunk.session} | Paper ${humanizePaperToken(
    chunk.paper
  )} | Variant ${humanizePaperToken(chunk.variant)}`;
}

function buildRagContextBlock(
  contextChunks: ContextChunkForPrompt[],
  maxSources: number = 3
): string {
  const selected = contextChunks.slice(0, Math.max(1, maxSources));
  if (selected.length === 0) return "";

  return selected
    .map((chunk, index) => {
      const topicPieces = [chunk.topicSyllabus, chunk.topicGeneral]
        .filter((value): value is string => Boolean(value && value.trim()))
        .join(" | ");
      const marksText = Number.isFinite(chunk.marks)
        ? String(Math.floor(Number(chunk.marks)))
        : "N/A";
      const questionText = truncateForPrompt(
        chunk.questionPart || chunk.rawContent,
        1200
      );
      const markSchemeText = truncateForPrompt(chunk.markSchemePart, 1400);

      return [
        `--- PAST PAPER SOURCE ${index + 1} ---`,
        `Source: ${formatSourceCitationFromContext(chunk)}`,
        topicPieces ? `Topic: ${topicPieces}` : undefined,
        `Marks: ${marksText}`,
        "",
        "QUESTION:",
        questionText || "(Question text unavailable in this chunk)",
        "",
        "MARKING SCHEME (what earns marks):",
        markSchemeText || "(Mark scheme text unavailable in this chunk)",
        "------------------------------",
      ]
        .filter((line): line is string => Boolean(line))
        .join("\n");
    })
    .join("\n\n");
}

function buildFallbackMarkingPointsFromContext(
  contextChunks: ContextChunkForPrompt[],
  expectedMarks?: number,
  style?: AnswerStyle
): Array<{ point: string; marks: number }> {
  const markSchemeCorpus = contextChunks
    .map((chunk) => chunk.markSchemePart || "")
    .filter((text) => text.trim().length > 0)
    .slice(0, 4)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

  if (!markSchemeCorpus) return [];

  const clauseCandidates = markSchemeCorpus
    .split(/(?:;|\.(?=\s+[A-Z]))/g)
    .map((part) => part.replace(/\s+/g, " ").trim())
    .filter((part) => part.length >= 24 && part.length <= 260);

  const scoreRegex =
    /(award|marks?|band|criterion|criteria|purpose|audience|text type|tone|register|required points|developed in detail|organisation|spelling|punctuation|grammar|vocabulary|observations?|equation|method|correct)/i;

  const seen = new Set<string>();
  const extracted: string[] = [];

  for (const clause of clauseCandidates) {
    if (!scoreRegex.test(clause)) continue;
    const normalized = clause.toLowerCase();
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    extracted.push(clause.replace(/^award\s+\d+\s+mark\s+for\s+/i, "").trim());
  }

  if (style?.responseShape === "directed_writing") {
    const rubricPoints = [
      "Address purpose, situation, and audience clearly throughout the article.",
      "Cover all required content points in detail with relevant development.",
      "Use appropriate article format, tone, and register for a school magazine.",
      "Organise ideas logically with clear paragraphing and coherent progression.",
      "Use accurate spelling, punctuation, and grammar.",
      "Use a suitable range of vocabulary to keep the writing informative and engaging.",
    ];

    for (const point of rubricPoints) {
      if (!extracted.some((item) => item.toLowerCase().includes(point.toLowerCase().slice(0, 18)))) {
        extracted.push(point);
      }
    }
  }

  const maxPoints =
    style?.responseShape === "directed_writing" ||
    (Number.isFinite(expectedMarks) && Number(expectedMarks) >= 20)
      ? 8
      : 5;

  const selected = extracted.slice(0, maxPoints);
  if (!selected.length) return [];

  const perPointMarks = Number.isFinite(expectedMarks) && Number(expectedMarks) > 1
    ? Math.max(1, Math.min(5, Math.floor(Number(expectedMarks) / Math.max(1, selected.length))))
    : 1;

  return selected.map((point) => ({ point, marks: perPointMarks }));
}

async function createStructuredCompletion(messages: any[], maxTokens: number): Promise<any> {
  const baseRequest = {
    messages,
    model: GROQ_MODEL,
    max_tokens: maxTokens,
    temperature: 0.1,
  };

  try {
    return await groq.chat.completions.create({
      ...baseRequest,
      response_format: { type: "json_object" },
    } as any);
  } catch (err: any) {
    const message = String(err?.message || err || "");
    if (!/(response_format|json_object|unsupported|schema)/i.test(message)) {
      throw err;
    }

    console.warn("[RAG] JSON response_format unavailable; retrying without strict JSON mode");
    return groq.chat.completions.create(baseRequest as any);
  }
}

// -- LLM ANSWER WITH CONTEXT -------------------------------------------------
async function generateExamAnswer(
  question: string,
  chunks: PastPaperChunk[],
  history: HistoryMessage[],
  style: AnswerStyle,
  answerMode: AnswerMode = "full_answer"
): Promise<{
  answer: string;
  markingPoints: Array<{ point: string; marks: number }>;
  commonMistakes: string[];
} | null> {
  if (chunks.length === 0) return null;

  const contextChunks = mergeSplitChunksForContext(chunks, 12);

  const expectedMarks = contextChunks.find(
    (chunk) => Number.isFinite(chunk.marks) && Number(chunk.marks) > 0
  )?.marks;

  const fallbackMarkingPoints = buildFallbackMarkingPointsFromContext(
    contextChunks,
    expectedMarks,
    style
  );

  const markingPointGuidance =
    expectedMarks === 1
      ? "marking_points: exactly 1 specific criterion from the mark scheme, marks must be 1."
      : style.responseShape === "directed_writing" ||
          (Number.isFinite(expectedMarks) && Number(expectedMarks) >= 20)
        ? "marking_points: 5-8 specific criteria covering both content/task fulfilment and language criteria from the mark scheme."
        : "marking_points: 3-5 specific criteria extracted from the mark scheme. marks value 1-3 each.";

  const questionTypeInfo = detectQuestionType(question, contextChunks, history);
  const dynamicFormatInstruction = getQuestionTypeFormatInstructions(
    questionTypeInfo.questionType,
    questionTypeInfo.marks || expectedMarks
  );
  const visualNeeds = detectVisualNeeds(question, contextChunks);
  const visualInstructions = getVisualFormatInstructions(visualNeeds);
  const activeVisuals = listActiveVisualNeeds(visualNeeds);
  const baseFormatInstruction =
    questionTypeInfo.questionType === "followup_elaborate" ||
    questionTypeInfo.questionType === "followup_example"
      ? "- Continue naturally from conversation context without repeating unchanged points."
      : buildFormatInstruction(style, answerMode);

  const contextBlockLimit = style.responseShape === "directed_writing" ? 4 : 3;
  const contextCharLimit = style.responseShape === "directed_writing" ? 14000 : 10000;
  const ragContext = buildRagContextBlock(contextChunks, contextBlockLimit)
    .slice(0, contextCharLimit)
    .trim();

  if (!ragContext) return null;

  const primarySource = contextChunks[0]
    ? formatSourceCitationFromContext(contextChunks[0])
    : "N/A";
  const sourceLineRule =
    style.responseShape === "directed_writing"
      ? "- Keep the final writing piece clean; do not append citation lines inside the body."
      : `- End with one short source line: "Based on: ${primarySource}".`;

  const avgSim =
    contextChunks.reduce((s, c) => s + c.similarity, 0) /
    Math.max(contextChunks.length, 1);

  const messages: any[] = [
    {
      role: "system",
      content: `You are an expert O-Level Cambridge exam tutor.

You have been given real past paper question extracts and official marking schemes. Ground your answer in that material.

QUESTION TYPE DETECTED: ${questionTypeInfo.questionType}

QUESTION-TYPE FORMAT INSTRUCTIONS:
${dynamicFormatInstruction}

VISUAL NEEDS DETECTED: ${activeVisuals.length ? activeVisuals.join(", ") : "none"}

VISUAL OUTPUT RULES:
${
  visualInstructions ||
  "- Use normal prose unless the student explicitly asks for a table, diagram, graph, timeline, or flowchart."
}

BASE FORMAT:
${baseFormatInstruction}

GROUNDING RULES:
- Treat retrieved context as the authoritative source for this answer.
- Use marking scheme points as the backbone of the response.
- Do not invent criteria that are not supported by retrieved mark schemes.
- If multiple sources are relevant, synthesize them without contradiction.
- If marks are provided, calibrate answer depth to those marks.
- If marks = 1, provide one direct valid point only.
- Response mode: ${answerMode === "mark_scheme_only" ? "MARK_SCHEME_ONLY" : "FULL_ANSWER"}.
${sourceLineRule}

The "marking_points" array must contain SPECIFIC mark scheme criteria extracted from the retrieved context.
Each "point" should be a real criterion (not generic phrases).

The "common_mistakes" array should contain realistic mistakes students make on this specific question type.

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "answer": "<full structured answer here>",
  "marking_points": [
    {"point": "Specific mark scheme criterion", "marks": 1}
  ],
  "common_mistakes": ["Specific mistake students make on this question type"]
}
${markingPointGuidance}
common_mistakes: 2-3 specific items.`,
    },
    ...history.slice(-10).map((h) => ({ role: h.role, content: h.content })),
    {
      role: "user",
      content: `Retrieved past paper context (avg relevance ${(avgSim * 100).toFixed(
        0
      )}%${expectedMarks ? `, expected marks ${expectedMarks}` : ""}):

${ragContext}

Now answer this student question using the above past paper content:
${question}`,
    },
  ];

  try {
    const completion = await createStructuredCompletion(
      messages,
      buildMaxTokens(style, true, answerMode)
    );

    return parseGroqResponse(completion.choices[0]?.message?.content || "", style, {
      expectedMarks,
      answerMode,
      fallbackMarkingPoints,
    });
  } catch (err) {
    console.error("generateExamAnswer error:", err);
    return null;
  }
}

// -- LLM ANSWER WITHOUT CONTEXT ----------------------------------------------
async function generateDirectAnswer(
  question: string,
  history: HistoryMessage[],
  style: AnswerStyle,
  answerMode: AnswerMode = "full_answer"
): Promise<{
  answer: string;
  markingPoints: Array<{ point: string; marks: number }>;
  commonMistakes: string[];
} | null> {
  const questionTypeInfo = detectQuestionType(question, [], history);
  const dynamicFormatInstruction = getQuestionTypeFormatInstructions(
    questionTypeInfo.questionType,
    questionTypeInfo.marks
  );
  const visualNeeds = detectVisualNeeds(question, []);
  const visualInstructions = getVisualFormatInstructions(visualNeeds);
  const activeVisuals = listActiveVisualNeeds(visualNeeds);
  const followUpMode =
    questionTypeInfo.questionType === "followup_elaborate" ||
    questionTypeInfo.questionType === "followup_example";
  const baseFormatInstruction = followUpMode
    ? "- Continue naturally from the ongoing conversation and add new detail without repetition."
    : buildFormatInstruction(style, answerMode);

  const messages: any[] = [
    {
      role: "system",
      content: `You are an expert O-Level Cambridge exam tutor. Answer using Cambridge O-Level curriculum knowledge.

QUESTION TYPE DETECTED: ${questionTypeInfo.questionType}

RULES:
- Clear, accurate, exam-focused answer for O-Level students.
- Use Cambridge mark-scheme language.
- Do NOT mention paper codes.
- Response mode: ${answerMode === "mark_scheme_only" ? "MARK_SCHEME_ONLY" : "FULL_ANSWER"}.
${
  followUpMode
    ? "- This is a follow-up: continue from recent conversation context, do not restart from scratch, and avoid repeating unchanged points."
    : ""
}

QUESTION-TYPE FORMAT INSTRUCTIONS:
${dynamicFormatInstruction}

VISUAL NEEDS DETECTED: ${activeVisuals.length ? activeVisuals.join(", ") : "none"}

VISUAL OUTPUT RULES:
${
  visualInstructions ||
  "- Use normal prose unless the student explicitly asks for a table, diagram, graph, timeline, or flowchart."
}

FORMAT:
${baseFormatInstruction}

Respond ONLY with this JSON (no markdown fences):
{
  "answer": "<answer here>",
  "marking_points": [{"point": "criterion", "marks": 1}],
  "common_mistakes": ["mistake"]
}`,
    },
    ...history.slice(-10).map((h) => ({ role: h.role, content: h.content })),
    { role: "user", content: question },
  ];

  try {
    const completion = await createStructuredCompletion(
      messages,
      buildMaxTokens(style, false, answerMode)
    );

    return parseGroqResponse(completion.choices[0]?.message?.content || "", style, {
      answerMode,
    });
  } catch (err) {
    console.error("generateDirectAnswer error:", err);
    return null;
  }
}

function parseGroqResponse(
  responseText: string,
  style: AnswerStyle,
  options: {
    expectedMarks?: number;
    answerMode?: AnswerMode;
    fallbackMarkingPoints?: Array<{ point: string; marks: number }>;
  } = {}
): {
  answer: string;
  markingPoints: Array<{ point: string; marks: number }>;
  commonMistakes: string[];
} {
  const expectedMarks =
    Number.isFinite(options.expectedMarks) && Number(options.expectedMarks) > 0
      ? Math.floor(Number(options.expectedMarks))
      : undefined;
  const answerMode = options.answerMode || "full_answer";

  let parsed: any = null;

  try {
    const cleaned = responseText
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/```\s*$/, "")
      .trim();
    const m = cleaned.match(/\{[\s\S]*\}/);
    parsed = m ? JSON.parse(m[0]) : null;
  } catch {
    parsed = null;
  }

  let finalAnswer =
    typeof parsed?.answer === "string" ? decodeJsonLikeString(parsed.answer) : "";

  if (!finalAnswer || isTemplateAnswer(finalAnswer)) {
    const recovered = cleanAnswerText(responseText);
    finalAnswer = !isTemplateAnswer(recovered)
      ? recovered
      : buildFallbackAnswer(style);
  }

  finalAnswer = enforceMarkdownStructure(finalAnswer, style);

  let markingPoints: Array<{ point: string; marks: number }> = [];

  if (Array.isArray(parsed?.marking_points) && parsed.marking_points.length > 0) {
    const filtered = parsed.marking_points.filter((item: any) => {
      const text = typeof item === "string" ? item : String(item?.point || "");
      // Reject generic fallback phrases.
      const isGeneric = /key concept|exam sources|refer to source|marking criterion/i.test(text);
      return text.trim() && !isGeneric;
    });

    if (filtered.length > 0) {
      markingPoints = filtered.map((item: any) =>
        typeof item === "string"
          ? { point: item, marks: 1 }
          : {
              point: String(item.point || item),
              marks: Math.max(1, Math.min(3, Number(item.marks) || 1)),
            }
      );
    }
  }

  if (markingPoints.length === 0 && options.fallbackMarkingPoints?.length) {
    markingPoints = options.fallbackMarkingPoints
      .map((item) => ({
        point: String(item.point || "").trim(),
        marks: Math.max(1, Math.min(5, Number(item.marks) || 1)),
      }))
      .filter((item) => item.point.length > 0);
  }

  if (expectedMarks) {
    if (expectedMarks === 1) {
      if (markingPoints.length === 0) {
        markingPoints = [
          { point: "Any one valid point from the mark scheme gains 1 mark.", marks: 1 },
        ];
      } else {
        markingPoints = [{ point: markingPoints[0].point, marks: 1 }];
      }
    } else {
      const cap =
        style.responseShape === "directed_writing" || expectedMarks >= 20
          ? 8
          : expectedMarks >= 10
            ? 6
            : 5;

      markingPoints = markingPoints.slice(0, cap).map((item) => ({
        point: item.point,
        marks: Math.max(1, Math.min(3, Number(item.marks) || 1)),
      }));
    }
  }

  if (answerMode === "mark_scheme_only" && expectedMarks === 1) {
    markingPoints = markingPoints.slice(0, 1).map((item) => ({
      point: item.point,
      marks: 1,
    }));
  }

  const commonMistakes: string[] = Array.isArray(parsed?.common_mistakes)
    ? parsed.common_mistakes.filter(
        (m: any) => typeof m === "string" && m.trim()
      )
    : [];

  return { answer: finalAnswer, markingPoints, commonMistakes };
}

interface SourceSummary {
  sourceType: SourceType;
  sourceNote: string;
}

function humanizePaperToken(value?: string): string {
  if (!value) return "?";
  return value
    .replace(/^paper[_\s-]*/i, "")
    .replace(/^variant[_\s-]*/i, "")
    .replace(/_/g, " ")
    .trim();
}

function humanizeSessionToken(session?: string): string {
  if (!session) return "Session unknown";
  return session
    .replace(/_/g, "/")
    .replace(/\s+/g, " ")
    .trim();
}

function buildQuestionRef(citation: Citation): string {
  const q = (citation.questionNumber || "").trim();
  const sub = (citation.subQuestion || "").trim();
  if (!q && !sub) return "";
  if (q && sub) return `Question ${q} ${sub}`;
  return q ? `Question ${q}` : `Question ${sub}`;
}

function formatCitationReference(c: Citation): string {
  const paper = humanizePaperToken(c.paper);
  const variant = humanizePaperToken(c.variant);
  const session = humanizeSessionToken(c.session);
  const questionRef = buildQuestionRef(c);
  const questionSuffix = questionRef ? `, ${questionRef}` : "";
  return `${c.subject}, ${c.year} ${session}, Paper ${paper}, Variant ${variant}${questionSuffix}`;
}

function buildSourceSummary(
  citations: Citation[],
  nearbyReferences: Citation[] = [],
  reason?: string
): SourceSummary {
  if (citations.length > 0) {
    const primary = citations[0];
    return {
      sourceType: "past_paper",
      sourceNote: `A similar question appeared in ${formatCitationReference(primary)}.`,
    };
  }

  if (nearbyReferences.length > 0) {
    const nearby = nearbyReferences[0];
    return {
      sourceType: "nearby_only",
      sourceNote:
        `I found related past paper material (closest match: ${formatCitationReference(nearby)}), but it was not a strong enough exact match to treat as a direct source.`,
    };
  }

  return {
    sourceType: "none",
    sourceNote:
      reason ||
      "I could not find a reliable past-paper match for this specific question.",
  };
}

function buildMarkSchemeOnlyAnswer(
  retrievalQuestion: string,
  markingPoints: Array<{ point: string; marks: number }>
): string {
  if (!markingPoints.length) {
    return `## Marking Scheme\nNo reliable mark-scheme points were found for this question in retrieved past-paper context.`;
  }

  const totalMarks = markingPoints.reduce(
    (sum, item) => sum + Math.max(1, Number(item.marks) || 1),
    0
  );

  return `## Marking Scheme
Question focus: ${retrievalQuestion}

${markingPoints
  .map((item, idx) => `${idx + 1}. ${item.point} (${Math.max(1, Number(item.marks) || 1)} mark${Math.max(1, Number(item.marks) || 1) > 1 ? "s" : ""})`)
  .join("\n")}

Estimated total: ${totalMarks} mark${totalMarks > 1 ? "s" : ""}`;
}

// -- RETRIEVAL DIAGNOSTICS ---------------------------------------------------
function buildDiagnostics(
  result: RagRetrievalResult | null,
  mode: "grounded" | "nearby" | "general",
  subjectFilter?: string,
  reason?: string
): RetrievalDiagnostics | undefined {
  if (!INCLUDE_STUDENT_RETRIEVAL_DIAGNOSTICS) return undefined;
  return {
    mode,
    subject_filter: subjectFilter,
    match_count: result?.chunks.length,
    best_similarity: result?.bestSimilarity,
    avg_top3: result?.avgTop3,
    reason,
    requested_match_count: subjectFilter ? TOP_K_WITH_FILTERS : TOP_K,
    similarity_threshold: SIMILARITY_THRESHOLD,
  };
}

function logDeveloperTrace(trace: DeveloperTrace): void {
  console.log(`[RAG_TRACE] ${JSON.stringify(trace)}`);
}

// -- GET /rag/subjects -------------------------------------------------------
router.get("/subjects", async (_req: Request, res: Response) => {
  try {
    const { data, error } = await supabase
      .from("past_paper_meta")
      .select("subject")
      .order("subject");

    if (error) return res.status(500).json({ error: error.message });

    const unique = [...new Set((data || []).map((r: any) => r.subject as string))];
    return res.json(unique.map((s) => ({ name: s })));
  } catch {
    return res.status(500).json({ error: "Failed to fetch subjects" });
  }
});

// -- POST /rag/query ---------------------------------------------------------
router.post("/query", async (req: Request, res: Response) => {
  try {
    const { question, filters, history = [] } = req.body as RagQueryRequest;

    if (!question?.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }

    const normalizedQuestion = question.trim();
    const resolved = resolveQuestionForAnswering(normalizedQuestion, history);
    const retrievalQuestion = resolved.retrievalQuestion;
    const style = inferAnswerStyle(normalizedQuestion, resolved.answerMode);
    const intent = classifyIntent(normalizedQuestion);
    let resolvedQuestionForClient: string | undefined = resolved.usedHistoryQuestion
      ? retrievalQuestion
      : undefined;

    // STAGE A - INTENT
    if (intent === "smalltalk") {
      const developerTrace: DeveloperTrace = {
        intent,
        answer_mode: resolved.answerMode,
        should_search_rag: false,
        followup_detected: false,
        decision_reason: "smalltalk_intent_detected",
        history_turns: history.length,
      };
      logDeveloperTrace(developerTrace);

      return res.json({
        type: "smalltalk",
        answer: getSmallTalkResponse(),
        citations: [],
        developer_trace: developerTrace,
      } as RagQueryResponse);
    }

    // STAGE B - RAG DECISION (conversation-aware)
    const ragDecision = evaluateRagDecision(
      normalizedQuestion,
      history,
      resolved.answerMode
    );
    const shouldRunRetrieval = ragDecision.shouldSearch;

    if (!shouldRunRetrieval) {
      console.log("[RAG] Follow-up detected; skipping retrieval and answering from conversation context");
      const direct = await generateDirectAnswer(
        normalizedQuestion,
        history,
        style,
        resolved.answerMode
      );

      if (!direct) {
        return res.status(500).json({ error: "Failed to generate follow-up answer" });
      }

      const developerTrace: DeveloperTrace = {
        intent,
        answer_mode: resolved.answerMode,
        should_search_rag: false,
        followup_detected: ragDecision.followUpDetected,
        decision_reason: ragDecision.reason,
        history_turns: history.length,
      };
      logDeveloperTrace(developerTrace);

      return res.json({
        type: "exam_question",
        answer: direct.answer,
        marking_points: direct.markingPoints,
        common_mistakes: direct.commonMistakes,
        citations: [],
        source_type: "none",
        resolved_question: resolvedQuestionForClient,
        developer_trace: developerTrace,
        retrieval: buildDiagnostics(
          null,
          "general",
          undefined,
          "Follow-up detected; retrieval skipped"
        ),
      } as RagQueryResponse);
    }

    const rewritten = rewriteQueryForRag(
      retrievalQuestion,
      normalizedQuestion,
      history
    );
    const ragQuery = rewritten.query || retrievalQuestion;

    if (rewritten.usedHistory) {
      resolvedQuestionForClient = ragQuery;
      console.log(`[RAG] Rewrote search query with history context: "${ragQuery}"`);
    }

    // STAGE C - SUBJECT INFERENCE
    const subjectKeyword =
      inferSubjectFromQuestion(ragQuery) ||
      inferSubjectFromQuestion(retrievalQuestion) ||
      inferSubjectFromQuestion(normalizedQuestion);
    const subjectFilter =
      filters?.subject ||
      (subjectKeyword ? resolveSubjectName(subjectKeyword) : undefined);
    const yearFilter = filters?.year;

    const developerTrace: DeveloperTrace = {
      intent,
      answer_mode: resolved.answerMode,
      should_search_rag: true,
      followup_detected: ragDecision.followUpDetected,
      decision_reason: ragDecision.reason,
      history_turns: history.length,
      retrieval_query: ragQuery,
      retrieval_query_rewritten: rewritten.usedHistory,
      subject_keyword: subjectKeyword || undefined,
      subject_filter: subjectFilter,
    };

    // STAGE D - RAG RETRIEVAL
    // Always run RAG - with subject filter when one is known, without when it
    // isn't.  Embedding similarity is the authoritative signal; keyword-based
    // subject inference is just an optimisation that widens the candidate pool.
    if (subjectFilter) {
      console.log(
        `[RAG] Retrieving for subject="${subjectFilter}" year=${yearFilter || "any"} (mode=${resolved.answerMode})`
      );
    } else {
      console.log(
        `[RAG] No subject inferred - running unfiltered retrieval (mode=${resolved.answerMode})`
      );
    }
    let ragResult = await ragRetrieval(
      ragQuery,
      subjectFilter,
      yearFilter
    );
    developerTrace.embedding_dimensions = ragResult.embeddingDimensions;

    // When we retrieved without a filter we can still discover which subject
    // the top chunks belong to and use that for logging / source attribution.
    let effectiveSubjectFilter = subjectFilter;
    let inferredSubjectFromResults: string | undefined;
    if (!subjectFilter) {
      const dominantFromGrounded = inferDominantSubject(ragResult.chunks, 8);
      const dominantFromNearby = inferDominantSubject(ragResult.nearbyChunks, 8);
      const inferredSubject = dominantFromGrounded || dominantFromNearby;

      if (inferredSubject) {
        effectiveSubjectFilter = inferredSubject;
        inferredSubjectFromResults = inferredSubject;
        console.log(`[RAG] Subject inferred from retrieval results: "${inferredSubject}"`);

        // If first pass is weak, run one focused pass using the inferred subject.
        if (!hasReliableContext(ragResult, false)) {
          const focused = await ragRetrieval(
            ragQuery,
            inferredSubject,
            yearFilter
          );

          if (
            focused.success &&
            (hasReliableContext(focused, true) ||
              focused.bestSimilarity > ragResult.bestSimilarity)
          ) {
            ragResult = focused;
            developerTrace.embedding_dimensions = ragResult.embeddingDimensions;
            console.log(
              `[RAG] Focused second-pass retrieval improved context (best=${focused.bestSimilarity.toFixed(3)})`
            );
          }
        }
      }
    }

    developerTrace.subject_filter = effectiveSubjectFilter;
    if (inferredSubjectFromResults) {
      developerTrace.inferred_subject_from_results = inferredSubjectFromResults;
    }

    // Build nearby references regardless
    const nearbyRefs: Citation[] =
      ragResult.nearbyChunks.length > 0
        ? buildCitations(
            ragResult.nearbyChunks.slice(0, NEARBY_REFERENCE_LIMIT),
            "nearby"
          )
        : [];

    // Check if retrieval failed completely
    if (/(unavailable|cohere)/i.test(ragResult.error || "")) {
      const source = buildSourceSummary(
        [],
        nearbyRefs,
        "This did not come from past papers because retrieval is temporarily unavailable."
      );
      logDeveloperTrace(developerTrace);
      return res.json({
        type: "exam_question",
        answer: "The AI study assistant is temporarily unavailable. Please try again later.",
        citations: [],
        source_note: source.sourceNote,
        source_type: source.sourceType,
        resolved_question: resolvedQuestionForClient,
        developer_trace: developerTrace,
        nearby_references: nearbyRefs.length > 0 ? nearbyRefs : undefined,
        retrieval: buildDiagnostics(ragResult, "general", effectiveSubjectFilter, ragResult.error),
      } as RagQueryResponse);
    }

    const reliable = hasReliableContext(ragResult, Boolean(effectiveSubjectFilter));
    const recoverable =
      !reliable &&
      resolved.answerMode !== "mark_scheme_only" &&
      hasRecoverableContext(retrievalQuestion, ragResult);

    if (recoverable) {
      const rescueChunks =
        ragResult.chunks.length > 0
          ? ragResult.chunks
          : ragResult.nearbyChunks.slice(0, Math.min(8, ragResult.nearbyChunks.length));

      console.log(
        `[RAG] Recovery path: using nearby context (best=${ragResult.bestSimilarity.toFixed(3)})`
      );

      const rescued = await generateExamAnswer(
        retrievalQuestion,
        rescueChunks,
        history,
        style,
        resolved.answerMode
      );

      if (rescued) {
        const rescueCitations = buildCitations(rescueChunks).slice(0, 1);
        const source = buildSourceSummary(
          rescueCitations,
          nearbyRefs,
          "Answer grounded using nearby past-paper context."
        );

        logDeveloperTrace(developerTrace);
        return res.json({
          type: "exam_question",
          answer: rescued.answer,
          marking_points: rescued.markingPoints,
          common_mistakes: rescued.commonMistakes,
          citations: rescueCitations,
          low_confidence: true,
          nearby_references: nearbyRefs.length > 0 ? nearbyRefs : undefined,
          source_note: source.sourceNote,
          source_type: source.sourceType,
          resolved_question: resolvedQuestionForClient,
          developer_trace: developerTrace,
          retrieval: buildDiagnostics(
            ragResult,
            "nearby",
            effectiveSubjectFilter,
            "Recovered using nearby context"
          ),
        } as RagQueryResponse);
      }
    }

    if (!reliable) {
      if (resolved.answerMode === "mark_scheme_only") {
        const source = buildSourceSummary(
          [],
          nearbyRefs,
          "This did not come from any exact past paper because similarity was below the reliability threshold for marking-scheme extraction."
        );

        logDeveloperTrace(developerTrace);
        return res.json({
          type: "exam_question",
          answer:
            "## Marking Scheme\nI could not find a reliable past-paper match for this question, so I cannot provide an exact marking scheme.",
          marking_points: [
            {
              point:
                "No reliable mark scheme found in retrieved past-paper context for this question.",
              marks: 1,
            },
          ],
          common_mistakes: [],
          citations: [],
          low_confidence: true,
          nearby_references: nearbyRefs.length > 0 ? nearbyRefs : undefined,
          source_note: source.sourceNote,
          source_type: source.sourceType,
          resolved_question: resolvedQuestionForClient,
          developer_trace: developerTrace,
          retrieval: buildDiagnostics(
            ragResult,
            nearbyRefs.length > 0 ? "nearby" : "general",
            effectiveSubjectFilter,
            "Below reliability threshold"
          ),
        } as RagQueryResponse);
      }

      // Fall back to direct LLM
      console.log(
        `[RAG] Context not reliable (best=${ragResult.bestSimilarity.toFixed(3)}) - direct answer`
      );
      const direct = await generateDirectAnswer(
        retrievalQuestion,
        history,
        style,
        resolved.answerMode
      );
      if (direct) {
        const source = buildSourceSummary([], nearbyRefs);
        logDeveloperTrace(developerTrace);
        return res.json({
          type: "exam_question",
          answer: direct.answer,
          marking_points: direct.markingPoints,
          common_mistakes: direct.commonMistakes,
          citations: [],
          low_confidence: true,
          nearby_references: nearbyRefs.length > 0 ? nearbyRefs : undefined,
          source_note: source.sourceNote,
          source_type: source.sourceType,
          resolved_question: resolvedQuestionForClient,
          developer_trace: developerTrace,
          retrieval: buildDiagnostics(
            ragResult,
            nearbyRefs.length > 0 ? "nearby" : "general",
            effectiveSubjectFilter,
            "Below reliability threshold"
          ),
        } as RagQueryResponse);
      }
      return res.status(500).json({ error: "Failed to generate answer" });
    }

    // STAGE D - LLM WITH CONTEXT
    const examAnswer = await generateExamAnswer(
      retrievalQuestion,
      ragResult.chunks,
      history,
      style,
      resolved.answerMode
    );

    if (!examAnswer) {
      const direct = await generateDirectAnswer(
        retrievalQuestion,
        history,
        style,
        resolved.answerMode
      );
      if (direct) {
        if (resolved.answerMode === "mark_scheme_only") {
          direct.answer = buildMarkSchemeOnlyAnswer(
            retrievalQuestion,
            direct.markingPoints
          );
          direct.commonMistakes = [];
        }

        const source = buildSourceSummary([], nearbyRefs);

        logDeveloperTrace(developerTrace);
        return res.json({
          type: "exam_question",
          answer: direct.answer,
          marking_points: direct.markingPoints,
          common_mistakes: direct.commonMistakes,
          citations: [],
          low_confidence: true,
          nearby_references: nearbyRefs.length > 0 ? nearbyRefs : undefined,
          source_note: source.sourceNote,
          source_type: source.sourceType,
          resolved_question: resolvedQuestionForClient,
          developer_trace: developerTrace,
          retrieval: buildDiagnostics(
            ragResult,
            "nearby",
            effectiveSubjectFilter,
            "Answer synthesis failed"
          ),
        } as RagQueryResponse);
      }
      return res.status(500).json({ error: "Failed to generate answer" });
    }

    const citations = buildCitations(ragResult.chunks).slice(0, 1);

    if (resolved.answerMode === "mark_scheme_only") {
      examAnswer.answer = buildMarkSchemeOnlyAnswer(
        retrievalQuestion,
        examAnswer.markingPoints
      );
      examAnswer.commonMistakes = [];
    }

    const source = buildSourceSummary(citations, nearbyRefs);

    const avgSim =
      ragResult.scores.reduce((s, v) => s + v, 0) / (ragResult.scores.length || 1);

    logDeveloperTrace(developerTrace);
    return res.json({
      type: "exam_question",
      answer: examAnswer.answer,
      marking_points: examAnswer.markingPoints,
      common_mistakes: examAnswer.commonMistakes,
      citations,
      confidence_score: Math.min(1, Math.max(0.1, avgSim)),
      low_confidence: avgSim < 0.4,
      nearby_references: nearbyRefs.length > 0 ? nearbyRefs : undefined,
      source_note: source.sourceNote,
      source_type: source.sourceType,
      resolved_question: resolvedQuestionForClient,
      developer_trace: developerTrace,
      retrieval: buildDiagnostics(ragResult, "grounded", effectiveSubjectFilter, "Grounded answer"),
    } as RagQueryResponse);
  } catch (error) {
    console.error("RAG query error:", error);
    return res
      .status(500)
      .json({ error: error instanceof Error ? error.message : "Unknown error" });
  }
});

export default router;