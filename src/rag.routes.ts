import { Router, Request, Response } from "express";
import { supabase } from "./lib/supabase";
import Groq from "groq-sdk";

const router = Router();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const GROQ_MODEL = process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

const OLLAMA_URL = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/$/, "");
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "bge-m3";
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY || "";
const HF_EMBED_URL = "https://router.huggingface.co/models/BAAI/bge-m3";

const SIMILARITY_THRESHOLD = 0.40;
const TOP_K = 16;
const MAX_CHUNKS_PER_FILE = 4;

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

// Comprehensive known CAIE / Pakistan board subject codes → names
// Keys include both standard CAIE codes and common custom numeric codes
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
  // Common custom numeric systems used in Pakistani schools
  "1001": "Physics", "1002": "Biology", "1003": "Mathematics",
  "1004": "Islamiyat", "1005": "Urdu", "1006": "English Language",
  "1007": "Computer Science", "1008": "Economics",
  "1009": "History", "1010": "Geography", "1011": "Chemistry",
  "1012": "Pakistan Studies", "1013": "Islamiyat",
  "1014": "Biology", "1015": "Pakistan Studies",
  "1016": "Physics", "1017": "Chemistry", "1018": "Mathematics",
  "1019": "English Language", "1020": "Computer Science",
};

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

  const { data, error } = await supabase
    .from("subjects")
    .select("id, code, level")
    .order("code");

  if (error || !data) {
    console.error("Cannot fetch subjects from DB:", error?.message);
    return [];
  }

  const customNames = parseCustomSubjectNames();

  subjectCache = data.map((s: any) => ({
    id: s.id as string,
    code: s.code as string,
    level: s.level as string,
    name: customNames[s.code] || KNOWN_CODE_NAMES[s.code] || `Subject ${s.code}`,
  }));

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

  const q = name.toLowerCase().trim();

  // Exact name match
  const exact = subjects.find((s) => s.name.toLowerCase() === q);
  if (exact) return exact;

  // Partial name match
  const partial = subjects.find(
    (s) => s.name.toLowerCase().includes(q) || q.includes(s.name.toLowerCase())
  );
  if (partial) return partial;

  // Code match (user typed the code directly)
  const byCode = subjects.find((s) => s.code.toLowerCase() === q);
  if (byCode) return byCode;

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
  paperFileId: string;
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
  error?: string;
}

interface RelatedQuestion {
  type: "exact" | "similar";
  text: string;
  source: { subject: string; year: number; session: string; paper: string; file_type: string };
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
}

// ============================================================================
// EMBEDDING — HuggingFace primary, Ollama fallback
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

async function getEmbedding(text: string): Promise<number[]> {
  if (HF_API_KEY) {
    try {
      console.log("[Embeddings] Calling HuggingFace with key:", HF_API_KEY.substring(0, 10) + "...");
      const res = await fetch(HF_EMBED_URL, {
        method: "POST",
        headers: { "Authorization": `Bearer ${HF_API_KEY}`, "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: text }),
        signal: AbortSignal.timeout(20000),
      });
      if (res.ok) {
        const data = await res.json() as any;

        let embedding: number[];
        if (Array.isArray(data[0]?.[0])) {
          // 3D tensor [batch, seq_len, hidden] — take batch 0, mean-pool over seq
          embedding = meanPoolTokens(data[0] as number[][]);
        } else if (Array.isArray(data[0])) {
          // 2D tensor [seq_len, hidden] — mean-pool over seq
          embedding = meanPoolTokens(data as number[][]);
        } else {
          // 1D flat vector [hidden] — use directly
          embedding = data as number[];
        }

        if (Array.isArray(embedding) && embedding.length > 0) return embedding;
      } else {
        const errorText = await res.text();
        console.warn(`HuggingFace embedding HTTP ${res.status}: ${errorText}`);
      }
    } catch (err) { console.warn("HuggingFace embedding error:", err); }
  } else {
    console.warn("[Embeddings] HUGGINGFACE_API_KEY not set in environment variables. Falling back to Ollama.");
  }

  try {
    const res = await fetch(`${OLLAMA_URL}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: OLLAMA_MODEL, prompt: text }),
      signal: AbortSignal.timeout(20000),
    });
    if (res.ok) {
      const data = await res.json() as any;
      if (Array.isArray(data?.embedding) && data.embedding.length > 0) return data.embedding;
    }
  } catch (err) { console.warn("Ollama embedding error:", err); }

  throw new Error("Embedding service unavailable. Set HUGGINGFACE_API_KEY or start Ollama.");
}

// ============================================================================
// STAGE A: INTENT CLASSIFICATION
// ============================================================================

function classifyIntent(question: string): ClassificationResult {
  const q = question.toLowerCase().trim();

  const smalltalkPatterns = [
    /^(hello|hi|hey|thanks|thank you|ok|okay|yes|no|sure|good|great|nice|cool|bye|goodbye)$/,
    /^[\?\!\.]{1,3}$/, /^(lol|haha|hmm|umm|err)$/,
  ];
  if (smalltalkPatterns.some((p) => p.test(q))) return { intent: "smalltalk" };

  const fileTypeIndicators = ["qp", "ms", "er", "gt"];
  const yearMatch = q.match(/\b(20\d{2})\b/);
  const paperMatch = q.match(/\b(p1|p2|p3)\b/i);

  // Common subject keywords
  const subjectKeywords = [
    "computer science", "english language", "english literature",
    "pakistan studies", "pakistan study",
    "chemistry", "physics", "biology", "mathematics", "maths", "math",
    "islamiyat", "urdu", "english", "computer", "economics",
    "history", "geography", "accounting", "commerce", "sociology",
    "additional mathematics", "add maths",
  ].sort((a, b) => b.length - a.length);

  let detectedSubjectKeyword: string | undefined;
  for (const kw of subjectKeywords) {
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

async function ragRetrieval(question: string, filters?: any): Promise<RagRetrievalResult> {
  try {
    let questionEmbedding: number[];
    try {
      questionEmbedding = await getEmbedding(question);
    } catch (embedErr: any) {
      return { success: false, groupedResults: [], rawSimilarityScores: [], error: embedErr.message };
    }

    const searchParams: Record<string, any> = {
      query_embedding: questionEmbedding,
      match_count: TOP_K,
    };
    if (filters?.subject) searchParams.filter_subject_code = filters.subject;
    if (filters?.year) searchParams.filter_year = filters.year;
    if (filters?.file_type) searchParams.filter_file_type = filters.file_type;
    if (filters?.level) searchParams.filter_level = filters.level;

    const { data: searchResults, error: searchErr } = await supabase.rpc("rag_search", searchParams);

    if (searchErr) {
      console.error("rag_search RPC error:", searchErr);
      return { success: false, groupedResults: [], rawSimilarityScores: [], error: searchErr.message };
    }

    if (!searchResults || searchResults.length === 0) {
      return { success: false, groupedResults: [], rawSimilarityScores: [], error: "No results found in database" };
    }

    const filteredResults = (searchResults as any[]).filter(
      (r) => (r.similarity as number) >= SIMILARITY_THRESHOLD
    );

    if (filteredResults.length === 0) {
      return {
        success: false, groupedResults: [], rawSimilarityScores: [],
        error: `No results above threshold. Best similarity was ${(searchResults[0] as any).similarity?.toFixed(3)}.`,
      };
    }

    const rawScores = filteredResults.map((r: any) => r.similarity as number);
    const chunkIds = filteredResults.map((r: any) => r.chunk_id);

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
      return { success: false, groupedResults: [], rawSimilarityScores: rawScores, error: "Failed to fetch chunk metadata" };
    }

    const similarityMap = new Map<string, number>(
      filteredResults.map((r: any) => [r.chunk_id as string, r.similarity as number])
    );

    const grouped = new Map<string, GroupedResult>();
    const sortedChunks = [...enrichedChunks].sort(
      (a: any, b: any) => (similarityMap.get(b.id) || 0) - (similarityMap.get(a.id) || 0)
    );

    for (const chunk of sortedChunks as any[]) {
      const fileId = chunk.paper_file_id;
      const pf = chunk.paper_files;
      const paper = pf?.papers;
      const subject = paper?.subjects;

      if (!grouped.has(fileId)) {
        grouped.set(fileId, {
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
      const group = grouped.get(fileId)!;
      if (group.chunks.length < MAX_CHUNKS_PER_FILE) {
        group.chunks.push({
          id: chunk.id, content: chunk.content,
          chunkIndex: chunk.chunk_index,
          similarity: similarityMap.get(chunk.id) || 0,
        });
      }
    }

    return { success: true, groupedResults: Array.from(grouped.values()), rawSimilarityScores: rawScores };
  } catch (err: any) {
    return { success: false, groupedResults: [], rawSimilarityScores: [], error: err?.message || "Retrieval error" };
  }
}

// ============================================================================
// STAGE C: LLM ANSWER GENERATION
// ============================================================================

async function generateExamAnswer(
  question: string,
  ragResult: RagRetrievalResult,
  history: HistoryMessage[] = []
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
- Answer the question directly and concisely. Use Cambridge mark-scheme language.
- Do NOT mention or guess any paper codes, series numbers (e.g. "1123", "2058"), or exam codes in your answer. The citation system identifies sources automatically.
- If the student asks "which paper is this from" or "what paper number", tell them to look at the source citations shown below your answer — do NOT guess codes yourself.
- Do not say the context is insufficient — always give your best answer from what is provided.
- No waffle. Be exam-focused and precise.

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "answer": "Direct 2-4 sentence answer to the question",
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
      max_tokens: 1200,
      temperature: 0.15,
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
        answer: responseText.substring(0, 500) || "Unable to generate a structured answer.",
        marking_points: [{ point: "Refer to source documents", marks: 1 }],
        common_mistakes: ["Review the exam context above"],
      };
    }

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

    const citations: Citation[] = sortedGroups
      .flatMap((group) =>
        group.chunks.map((chunk) => ({
          subject: group.subject,
          subjectName: resolveSubjectName(group.subject),
          year: group.year,
          session: group.session,
          paper: group.paper,
          file_type: group.filetype,
          storage_path: group.storagePath,
          chunk_index: chunk.chunkIndex,
          similarity: chunk.similarity,
        }))
      )
      .sort((a, b) => b.similarity - a.similarity);

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
        },
        similarity: best.chunk.similarity,
      };
    }

    return {
      answer: String(parsed.answer),
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
  history: HistoryMessage[] = []
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
- Be concise and educational.

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "answer": "Clear explanation of the concept",
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
      max_tokens: 1000,
      temperature: 0.2,
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
        answer: responseText.substring(0, 500) || "Unable to generate an answer.",
        marking_points: [],
        common_mistakes: [],
      };
    }

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

    return { answer: String(parsed.answer), markingPoints, commonMistakes };
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
    const ragResult = await ragRetrieval(question, filters);

    if (!ragResult.success) {
      if (ragResult.error?.includes("unavailable")) {
        return res.json({
          type: "exam_question",
          answer: "The AI study assistant is temporarily unavailable. Please try again later.",
          citations: [],
        } as RagQueryResponse);
      }
      // RAG found nothing — fall back to direct LLM answer
      const directAnswer = await generateDirectAnswer(question, history);
      if (directAnswer) {
        return res.json({
          type: "exam_question",
          answer: directAnswer.answer,
          marking_points: directAnswer.markingPoints,
          common_mistakes: directAnswer.commonMistakes,
          citations: [],
        } as RagQueryResponse);
      }
      return res.json({
        type: "exam_question",
        answer: "I couldn't generate an answer right now. Please try again.",
        citations: [],
      } as RagQueryResponse);
    }

    const examAnswer = await generateExamAnswer(question, ragResult, history);

    if (!examAnswer) {
      const fallbackCitations: Citation[] = ragResult.groupedResults
        .flatMap((g) => g.chunks.map((c) => ({
          subject: g.subject, subjectName: resolveSubjectName(g.subject),
          year: g.year, session: g.session, paper: g.paper,
          file_type: g.filetype, storage_path: g.storagePath,
          chunk_index: c.chunkIndex, similarity: c.similarity,
        })))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5);

      return res.json({
        type: "exam_question",
        answer: ragResult.groupedResults.flatMap((g) => g.chunks.map((c) => c.content)).join("\n\n").substring(0, 800),
        citations: fallbackCitations,
        confidence_score: Math.min(1, ragResult.rawSimilarityScores[0] || 0),
        coverage_percentage: Math.min(100, (ragResult.groupedResults.length / 8) * 100),
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
    } as RagQueryResponse);

  } catch (error) {
    console.error("RAG query error:", error);
    return res.status(500).json({ error: error instanceof Error ? error.message : "Unknown error" });
  }
});

export default router;
