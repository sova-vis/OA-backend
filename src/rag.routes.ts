import { Router, Request, Response } from "express";
import { supabase } from "./lib/supabase";
import Groq from "groq-sdk";

const router = Router();

// Groq config (free alternative to OpenAI)
const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

// Use configurable model or fallback to stable one
const GROQ_MODEL = process.env.GROQ_MODEL || "gemma-7b-it";

// Ollama config (for embeddings)
const OLLAMA_URL = (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/$/, "");
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "bge-m3";

// Pipeline thresholds and limits
const SIMILARITY_THRESHOLD = 0.5;
const TOP_K = 16;
const MAX_CHUNKS_PER_FILE = 2;

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

type Intent = "paper_lookup" | "exam_question" | "smalltalk";

interface ClassificationResult {
  intent: Intent;
  metadata?: {
    subject?: string;
    year?: number;
    paper?: string;
    fileType?: string;
  };
}

interface GroupedResult {
  paperFileId: string;
  filetype: string;
  storagePath: string;
  chunks: Array<{
    id: string;
    content: string;
    chunkIndex: number;
    similarity: number;
  }>;
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

interface ExamAnswer {
  answer: string;
  markingPoints: Array<{ point: string; marks: number }>;
  commonMistakes: string[];
  citations: Citation[];
  confidenceScore: number;
  coveragePercentage: number;
}

interface RagQueryRequest {
  question: string;
  limit?: number;
  filters?: {
    subject?: string;
    year?: number;
    file_type?: string;
    level?: string;
  };
}

interface Citation {
  subject: string;
  subjectName: string;
  year: number;
  session: string;
  paper: string;
  file_type: string;
  storage_path: string;
  chunk_index: number;
  similarity: number;
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
  results?: any[];  // For paper_lookup responses
}

// ============================================================================
// STAGE A: INTENT CLASSIFICATION
// ============================================================================

function classifyIntent(question: string): ClassificationResult {
  const q = question.toLowerCase().trim();

  // Smalltalk: exact word matches or punctuation-only
  const smalltalkPatterns = [
    /^(hello|hi|hey|thanks|thank you|ok|okay|yes|no|sure|good|great|nice|cool|bye|goodby)$/,
    /^[\?\!\.]{1,3}$/,
    /^(lol|haha|hmm|umm|err)$/,
  ];

  if (smalltalkPatterns.some((p) => p.test(q))) {
    return { intent: "smalltalk" };
  }

  // Paper lookup: subject + (year or paper/file type indicator)
  const subjects = [
    "chemistry",
    "physics",
    "biology",
    "mathematics",
    "islamiyat",
    "urdu",
    "english",
    "computer",
    "economics",
    "history",
    "geography",
    "art",
    "music",
    "english language",
    "english literature",
  ];

  const fileTypeIndicators = ["qp", "ms", "er", "gt"];
  const yearIndicators = /\b(20\d{2})\b/;
  const paperIndicators = /\b(p1|p2|p3)\b/i;

  let detectedSubject: string | undefined;
  let detectedYear: number | undefined;
  let detectedPaper: string | undefined;
  let detectedFileType: string | undefined;

  // Find subject
  for (const subject of subjects) {
    if (q.includes(subject)) {
      detectedSubject = subject;
      break;
    }
  }

  // Find year
  const yearMatch = q.match(yearIndicators);
  if (yearMatch) {
    detectedYear = parseInt(yearMatch[1]);
  }

  // Find paper
  const paperMatch = q.match(paperIndicators);
  if (paperMatch) {
    detectedPaper = paperMatch[1].toUpperCase();
  }

  // Find file type
  for (const ft of fileTypeIndicators) {
    if (q.includes(ft)) {
      detectedFileType = ft.toUpperCase();
      break;
    }
  }

  // If we have subject + at least one indicator, it's a paper lookup
  if (detectedSubject && (detectedYear || detectedFileType || detectedPaper)) {
    return {
      intent: "paper_lookup",
      metadata: {
        subject: detectedSubject,
        year: detectedYear,
        paper: detectedPaper,
        fileType: detectedFileType,
      },
    };
  }

  // Default to exam question
  return { intent: "exam_question" };
}

// ============================================================================
// STAGE B: IMPROVED RAG RETRIEVAL
// ============================================================================

async function ragRetrieval(question: string, filters?: any): Promise<RagRetrievalResult> {
  try {
    // 1. Try Ollama embeddings, fallback to simple keyword matching
    let questionEmbedding: number[] = [];
    let usesFallback = false;

    try {
      const embedUrl =
        (process.env.OLLAMA_URL || "http://localhost:11434").replace(/\/$/, "") +
        "/api/embeddings";

      const embedRes = await fetch(embedUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: OLLAMA_MODEL,
          prompt: question,
        }),
      });

      if (embedRes.ok) {
        const embedData = (await embedRes.json()) as any;
        if (embedData?.embedding && Array.isArray(embedData.embedding)) {
          questionEmbedding = embedData.embedding;
        } else {
          usesFallback = true;
        }
      } else {
        usesFallback = true;
      }
    } catch (err) {
      console.warn("Ollama unavailable, using keyword fallback:", err);
      usesFallback = true;
    }

    // Fallback: Create simple embedding from keywords
    if (usesFallback || questionEmbedding.length === 0) {
      const keywords = question.toLowerCase().split(/\s+/).filter(w => w.length > 3);
      questionEmbedding = new Array(384).fill(0); // 384-dim vector like bge-m3
      keywords.forEach((keyword, idx) => {
        const seed = keyword.charCodeAt(0) + keyword.charCodeAt(keyword.length - 1);
        for (let i = 0; i < Math.min(5, 384); i++) {
          questionEmbedding[(idx * 5 + i) % 384] += ((seed * (i + 1)) % 100) / 100;
        }
      });
      // Normalize
      const norm = Math.sqrt(questionEmbedding.reduce((a, b) => a + b * b, 0));
      if (norm > 0) {
        questionEmbedding = questionEmbedding.map(v => v / norm);
      }
    }

    // 2. Search with Supabase RPC (TOP_K results)
    const searchParams = {
      query_embedding: questionEmbedding,
      match_count: TOP_K,
      ...(filters?.subject && { filter_subject_code: filters.subject }),
      ...(filters?.year && { filter_year: filters.year }),
      ...(filters?.file_type && { filter_file_type: filters.file_type }),
      ...(filters?.level && { filter_level: filters.level }),
    };

    const { data: searchResults, error: searchErr } = await supabase.rpc(
      "rag_search",
      searchParams
    );

    if (searchErr || !searchResults || searchResults.length === 0) {
      return {
        success: false,
        groupedResults: [],
        rawSimilarityScores: [],
        error: "No results found",
      };
    }

    // 3. Filter by similarity threshold
    const filteredResults = (searchResults || []).filter(
      (r: any) => (r.similarity as number) >= SIMILARITY_THRESHOLD
    );

    if (filteredResults.length === 0) {
      return {
        success: false,
        groupedResults: [],
        rawSimilarityScores: [],
        error: `No results above similarity threshold (${SIMILARITY_THRESHOLD})`,
      };
    }

    // 4. Store raw similarity scores
    const rawScores = filteredResults.map((r: any) => r.similarity as number);

    // 5. Get full chunk metadata
    const chunkIds = filteredResults.map((r: any) => r.chunk_id);

    const { data: enrichedChunks, error: enrichErr } = await supabase
      .from("rag_chunks")
      .select(
        `
        id,
        chunk_index,
        content,
        paper_file_id,
        paper_files!inner(
          id,
          file_type,
          storage_path,
          paper_id,
          papers!inner(
            id,
            year,
            session,
            paper,
            subject_id,
            subjects!inner(
              id,
              code,
              level
            )
          )
        )
      `
      )
      .in("id", chunkIds);

    if (enrichErr || !enrichedChunks) {
      return {
        success: false,
        groupedResults: [],
        rawSimilarityScores: rawScores,
        error: "Failed to enrich results",
      };
    }

    // 6. Build similarity map
    const similarityMap = new Map(
      filteredResults.map((r: any) => [r.chunk_id, r.similarity as number])
    );

    // 7. Group results by paper_file and apply MAX_CHUNKS_PER_FILE limit
    const grouped = new Map<string, GroupedResult>();

    enrichedChunks.forEach((chunk: any) => {
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
          id: chunk.id,
          content: chunk.content,
          chunkIndex: chunk.chunk_index,
          similarity: (similarityMap.get(chunk.id) as number) || 0,
        });
      }
    });

    return {
      success: true,
      groupedResults: Array.from(grouped.values()),
      rawSimilarityScores: rawScores,
    };
  } catch (err: any) {
    return {
      success: false,
      groupedResults: [],
      rawSimilarityScores: [],
      error: err?.message || "Retrieval error",
    };
  }
}

// ============================================================================
// STAGE C: EXAM-STYLE ANSWER GENERATION
// ============================================================================

async function generateExamAnswer(
  question: string,
  ragResult: RagRetrievalResult
): Promise<ExamAnswer | null> {
  if (!ragResult.success || ragResult.groupedResults.length === 0) {
    return null;
  }

  try {
    // Build context from grouped results
    const contextChunks = ragResult.groupedResults
      .flatMap((group) => group.chunks.map((c) => c.content))
      .join("\n\n");

    if (!contextChunks.trim()) {
      return null;
    }

    // Calculate average similarity for mark weighting
    const contextAvgSimilarity = ragResult.rawSimilarityScores.length > 0
      ? ragResult.rawSimilarityScores.reduce((a, b) => a + b, 0) / ragResult.rawSimilarityScores.length
      : 0;

    // Determine mark range based on similarity threshold
    let markGuidance = "1-2 marks each";
    if (contextAvgSimilarity >= 0.7) {
      markGuidance = "2-3 marks each (high confidence)";
    } else if (contextAvgSimilarity >= 0.5) {
      markGuidance = "1-2 marks each (medium confidence)";
    } else {
      markGuidance = "1 mark each (low confidence)";
    }

    // Call Groq with exam-style instructions
    const completion = await groq.chat.completions.create({
      messages: [
        {
          role: "system",
          content: `You are an expert exam tutor. Provide exam-style answers with:
1. Clear main answer (2-3 sentences max)
2. Marking points (3-5 bullet points with marks allocated based on importance and confidence)
3. Common mistakes students make (2-3 mistakes)

Mark allocation based on similarity threshold:
- High confidence (≥70%): Award 2-3 marks per important point
- Medium confidence (≥50%): Award 1-2 marks per point
- Low confidence (<50%): Award 1 mark per point

Current confidence level suggests: ${markGuidance}

Format your response EXACTLY as JSON:
{
  "answer": "Your main answer here",
  "marking_points": [
    {"point": "First key concept", "marks": 2},
    {"point": "Second key concept", "marks": 1}
  ],
  "common_mistakes": ["Mistake 1", "Mistake 2"]
}`,
        },
        {
          role: "user",
          content: `Question: ${question}\n\nExam paper context:\n${contextChunks.substring(
            0,
            2000
          )}`,
        },
      ],
      model: GROQ_MODEL,
      max_tokens: 700,
      temperature: 0.7,
    });

    const responseText = completion.choices[0]?.message?.content || "";

    // Parse JSON response
    let parsed: any;
    try {
      // Try to extract JSON from response
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : null;
    } catch {
      parsed = null;
    }

    // Fallback if JSON parsing fails
    if (!parsed || !parsed.answer) {
      parsed = {
        answer: responseText.substring(0, 500) || "Unable to generate answer from the retrieved documents.",
        marking_points: [
          { point: "Review the answer against the source material", marks: 1 }
        ],
        common_mistakes: ["Unclear from the available resources"]
      };
    }

    // Parse marking points (support both old and new format)
    let markingPoints: Array<{ point: string; marks: number }> = [];
    if (Array.isArray(parsed.marking_points) && parsed.marking_points.length > 0) {
      markingPoints = parsed.marking_points.map((item: any) => {
        if (typeof item === 'string') {
          // Old format: just strings
          return { point: item, marks: 1 };
        } else if (item && item.point && item.marks) {
          // New format: objects with point and marks
          return { point: item.point, marks: Math.max(1, item.marks) };
        }
        return { point: String(item), marks: 1 };
      });
    } else {
      // Default if empty
      markingPoints = [{ point: "Key concept from sources", marks: 1 }];
    }

    // Calculate confidence (now more robust)
    const avgSimilarity =
      ragResult.rawSimilarityScores.length > 0
        ? ragResult.rawSimilarityScores.reduce((a, b) => a + b, 0) /
          ragResult.rawSimilarityScores.length
        : 0.3; // Default to 0.3 if no scores (fallback mode)

    // Boost confidence if we have good results
    let confidenceScore = Math.min(1, Math.max(0.1, avgSimilarity));
    if (ragResult.groupedResults.length >= 3) {
      confidenceScore = Math.min(1, confidenceScore * 1.2);
    }

    const coveragePercentage = Math.min(
      100,
      (ragResult.groupedResults.length / TOP_K) * 100
    );

    // Build citations with subject names
    const citations: Citation[] = ragResult.groupedResults
      .flatMap((group) =>
        group.chunks.map((chunk) => ({
          subject: group.subject,
          subjectName: getSubjectName(group.subject),
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

    return {
      answer: parsed.answer || "Unable to generate answer",
      markingPoints,
      commonMistakes: Array.isArray(parsed.common_mistakes)
        ? parsed.common_mistakes
        : [],
      citations: citations.slice(0, 5),
      confidenceScore,
      coveragePercentage,
    };
  } catch (err) {
    console.error("Exam answer generation error:", err);
    return null;
  }
}

// ============================================================================
// SMALLTALK RESPONSES
// ============================================================================

const smalltalkResponses = [
  "Hi! 👋 Ask me about a topic or tell me a subject, year, and paper type to find.",
  "Hey there! 😊 What would you like to know? You can ask about exam topics or find specific papers.",
  "Hello! 📚 I can help you find past papers or answer questions about your subjects.",
  "Hi! Want to search for papers or have a question about your studies?",
];

function getSmallTalkResponse(): string {
  return smalltalkResponses[Math.floor(Math.random() * smalltalkResponses.length)];
}

// Helper to convert subject code to name
function getSubjectName(code: string): string {
  const subjectMap: Record<string, string> = {
    "1011": "Chemistry",
    "1001": "Physics",
    "1002": "Biology",
    "1003": "Mathematics",
    "1004": "Islamiyat",
    "1005": "Urdu",
    "1006": "English",
    "1007": "Computer Science",
    "1008": "Economics",
    "1009": "History",
    "1010": "Geography",
  };
  return subjectMap[code] || code;
}

// ============================================================================
// MAIN ENDPOINT: POST /rag/query (3-STAGE PIPELINE)
// ============================================================================

/**
 * Three-stage RAG pipeline:
 *
 * STAGE A: Intent Classification
 *   - Classify query into: smalltalk | paper_lookup | exam_question
 *   - Extract metadata (subject, year, file type)
 *
 * STAGE B: Improved RAG Retrieval
 *   - Embed question with Ollama
 *   - Search with TOP_K=16 results
 *   - Filter by SIMILARITY_THRESHOLD=0.5
 *   - Group by paper_file, MAX_CHUNKS_PER_FILE=2
 *   - Return grouped results and similarity scores
 *
 * STAGE C: Exam-Style Answer Generation
 *   - Use Groq to generate answer from context
 *   - Extract marking_points and common_mistakes
 *   - Calculate confidence_score and coverage_percentage
 *   - Return formatted ExamAnswer
 */
router.post("/query", async (req: Request, res: Response) => {
  try {
    const { question, limit = 5, filters } = req.body as RagQueryRequest;

    if (!question || !question.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }

    // ========== STAGE A: CLASSIFY INTENT ==========
    const classification = classifyIntent(question);

    // CASE 1: SMALLTALK
    if (classification.intent === "smalltalk") {
      const response: RagQueryResponse = {
        type: "smalltalk",
        answer: getSmallTalkResponse(),
        citations: [],
      };
      return res.json(response);
    }

    // CASE 2: PAPER LOOKUP
    if (classification.intent === "paper_lookup") {
      const metadata = classification.metadata;

      // Query papers based on extracted metadata
      let query = supabase
        .from("papers")
        .select(`
          id,
          year,
          session,
          paper,
          subject_id,
          subjects(code, level),
          paper_files(file_type, storage_path, id)
        `);

      if (metadata?.year) {
        query = query.eq("year", metadata.year);
      }

      if (metadata?.fileType) {
        // Will filter after fetch
      }

      const { data, error } = await query;

      if (error) {
        return res.status(500).json({ error: error.message });
      }

      let results = (data || []) as any[];

      // Filter by file type if present
      if (metadata?.fileType) {
        results = results.filter((paper) =>
          paper.paper_files?.some((pf: any) => pf.file_type === metadata.fileType)
        );
      }

      // Group results by subject/year/session
      const grouped = results.reduce((acc: any, paper: any) => {
        const key = `${paper.subjects?.code}-${paper.year}`;
        if (!acc[key]) {
          acc[key] = {
            subject: paper.subjects?.code,
            year: paper.year,
            level: paper.subjects?.level,
            sessions: {},
          };
        }

        if (!acc[key].sessions[paper.session]) {
          acc[key].sessions[paper.session] = {
            session: paper.session,
            papers: {},
          };
        }

        if (!acc[key].sessions[paper.session].papers[paper.paper]) {
          acc[key].sessions[paper.session].papers[paper.paper] = {
            paper: paper.paper,
            files: {},
          };
        }

        (paper.paper_files || []).forEach((pf: any) => {
          acc[key].sessions[paper.session].papers[paper.paper].files[pf.file_type] = {
            file_type: pf.file_type,
            storage_path: pf.storage_path,
            id: pf.id,
          };
        });

        return acc;
      }, {});

      const response: RagQueryResponse = {
        type: "paper_lookup",
        answer: `Found ${Object.keys(grouped).length} paper set(s):`,
        results: Object.values(grouped),
        citations: [],
      };
      return res.json(response);
    }

    // CASE 3: EXAM QUESTION - Use 3-stage pipeline
    // ========== STAGE B: RAG RETRIEVAL ==========
    const ragResult = await ragRetrieval(question, filters);

    if (!ragResult.success) {
      // Low confidence response
      const response: RagQueryResponse = {
        type: "exam_question",
        answer:
          "I don't have enough information in my database to answer this question with confidence. Try asking about specific exam topics or request a particular past paper.",
        citations: [],
        confidence_score: 0,
        coverage_percentage: 0,
        low_confidence: true,
      };
      return res.json(response);
    }

    // ========== STAGE C: EXAM ANSWER GENERATION ==========
    const examAnswer = await generateExamAnswer(question, ragResult);

    if (!examAnswer) {
      // Fallback if LLM generation fails
      const fallbackCitations: Citation[] = ragResult.groupedResults
        .flatMap((group) =>
          group.chunks.map((chunk) => ({
            subject: group.subject,
            subjectName: getSubjectName(group.subject),
            year: group.year,
            session: group.session,
            paper: group.paper,
            file_type: group.filetype,
            storage_path: group.storagePath,
            chunk_index: chunk.chunkIndex,
            similarity: chunk.similarity,
          }))
        )
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, 5);

      const contextText = ragResult.groupedResults
        .flatMap((g) => g.chunks.map((c) => c.content))
        .join("\n\n")
        .substring(0, 600);

      const response: RagQueryResponse = {
        type: "exam_question",
        answer: `Based on exam sources:\n\n${contextText}...`,
        citations: fallbackCitations,
        confidence_score: Math.min(
          1,
          (ragResult.rawSimilarityScores[0] || 0) as number
        ),
        coverage_percentage: Math.min(
          100,
          (ragResult.groupedResults.length / TOP_K) * 100
        ),
      };
      return res.json(response);
    }

    // Success response with full exam answer
    const response: RagQueryResponse = {
      type: "exam_question",
      answer: examAnswer.answer,
      marking_points: examAnswer.markingPoints,
      common_mistakes: examAnswer.commonMistakes,
      citations: examAnswer.citations.slice(0, limit),
      confidence_score: examAnswer.confidenceScore,
      coverage_percentage: examAnswer.coveragePercentage,
      low_confidence: examAnswer.confidenceScore < 0.4,
    };

    res.json(response);
  } catch (error) {
    console.error("RAG query error:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
});

export default router;
