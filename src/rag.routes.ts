import { Router, Request, Response } from "express";
import multer from "multer";
import { grokEnabled, grokChatJson, grokVisionModel, grokErrorMessage } from "./lib/grok";
import { generate, questionPreview, callLLMWithFallback } from "./lib/askai/generate";
import { AskAiIndexUnavailable, type Occurrence } from "./lib/askai/retrieve";

// Ask AI runs IN-PROCESS now (no separate Python service): /query embeds the
// question with bge-base, searches the pgvector index in Supabase (the
// match_ask_ai_chunks RPC), and generates the answer via the LLM — see
// src/lib/askai/*. /ask-image is a separate photo/diagram Q&A via Grok vision.

const router = Router();
const visionUpload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });

function buildCitation(o: Occurrence) {
  return {
    subject: o.subject, year: o.year, session: o.session, paper: o.paper,
    variant: o.variant, questionNumber: o.question_number,
    topicGeneral: o.topic, topicSyllabus: null,
    preview: questionPreview(o.question_text || ""),
    pageImageUrl: null, // DB-sourced questions have no PDF page image
  };
}

// Subjects covered by the chatbot's vector store (see scripts/retrieve.py's
// SUBJECTS list in the Past-Paper Chatbot project) - hardcoded here since
// that service has no /subjects endpoint of its own.
const SUBJECTS = [
  "Accounting", "Additional Maths", "Art and Design", "Biology", "Business Studies",
  "Chemistry", "Commerce", "Computer Science", "Economics", "English",
  "Environmental Management", "Geography", "History", "Islamiyat", "Mathematics",
  "Pakistan Studies", "Physics", "Religious Studies", "Sociology", "Statistics",
];

router.get("/subjects", async (_req: Request, res: Response) => {
  return res.json(SUBJECTS.map((s) => ({ name: s })));
});

router.post("/query", async (req: Request, res: Response) => {
  try {
    const { question, mode, subject, level } = req.body as {
      question?: string; mode?: string; subject?: string; level?: string;
    };

    if (!question?.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }

    const { answer, result } = await generate(question, {
      subject: subject || null, level: level || null, mode: mode || null,
    });
    const occurrences = result.occurrences || [];
    return res.json({
      type: "exam_question",
      mode: result.intent === "paper_lookup" ? "find" : "ask",
      answer,
      citations: occurrences.map(buildCitation),
      source_type: occurrences.length ? "past_paper" : "none",
      subject: subject || result.hits[0]?.metadata.subject || undefined,
    });
  } catch (err: any) {
    if (err instanceof AskAiIndexUnavailable) {
      console.error("[RAG] index unavailable:", err.message);
      return res.status(503).json({ error: "Ask AI is still being set up — the question index isn't ready yet." });
    }
    console.error("[RAG] query failed:", err?.message || err);
    return res.status(500).json({ error: "Ask AI couldn't answer right now. Please try again." });
  }
});

/**
 * POST /rag/explain-mcq — one quick LLM call explaining why an MCQ answer is
 * wrong, shown inline under the question (no navigation to the Ask AI page).
 */
router.post("/explain-mcq", async (req: Request, res: Response) => {
  try {
    const { questionText, options, correctAnswer, studentAnswer } = req.body as {
      questionText?: string; options?: Array<{ label?: string; text?: string }>;
      correctAnswer?: string; studentAnswer?: string;
    };
    if (!questionText?.trim() || !correctAnswer) {
      return res.status(400).json({ error: "Missing question or correct answer" });
    }
    const optionsText = Array.isArray(options)
      ? options.map((o) => `${o.label}. ${o.text}`).join("\n")
      : "";
    const system =
      "You are a concise Cambridge O/A Level tutor. In 2-4 short sentences or a few " +
      "'- ' Markdown bullets, explain why the correct option is right and, if the " +
      "student picked a different one, why theirs is wrong. Plain text / Markdown " +
      "only — never LaTeX; write any formula in plain text (e.g. '6CO2 + 6H2O -> ...'). " +
      "Be specific to this question; don't restate the whole question.";
    const wrong = studentAnswer && studentAnswer !== correctAnswer;
    const user =
      `Question: ${questionText}\n${optionsText ? "Options:\n" + optionsText + "\n" : ""}` +
      `Correct answer: ${correctAnswer}\nStudent chose: ${studentAnswer || "(none)"}\n\n` +
      `Explain why ${correctAnswer} is correct${wrong ? ` and why ${studentAnswer} is wrong` : ""}.`;
    const answer = await callLLMWithFallback(system, user);
    return res.json({ answer });
  } catch (err: any) {
    console.error("[RAG] explain-mcq failed:", err?.message || err);
    return res.status(500).json({ error: "Couldn't explain that right now. Please try again." });
  }
});

/**
 * POST /rag/ask-image  (multipart) — Ask AI with an attached image.
 * Diagrams, graphs, circuits, chemical structures or a photographed question are
 * read by Grok vision (grok-4.5) and answered as an O/A-Level tutor.
 */
router.post("/ask-image", visionUpload.single("image"), async (req: Request, res: Response) => {
  try {
    const question = String(req.body?.question || "").trim() || "Read the attached image and answer any question in it, explaining clearly.";
    const subject = String(req.body?.subject || "").trim();
    const file = req.file;
    if (!file || !/^image\//.test(file.mimetype)) {
      return res.status(400).json({ error: "Attach an image (JPG or PNG)." });
    }
    if (!grokEnabled()) {
      return res.status(503).json({ error: "AI image reading is not configured. Add XAI_API_KEY in OA-backend/.env." });
    }
    const system = [
      "You are a friendly, precise Cambridge O/A Level tutor.",
      "The student has attached an image — a diagram, graph, circuit, chemical structure, or a photographed exam question.",
      "Read it carefully and answer clearly and correctly at O/A-Level depth. If it's a question, solve it and show the key steps.",
      "Return JSON ONLY: { \"answer\": string }.",
    ].join(" ");
    const user = `${subject ? "Subject: " + subject + ". " : ""}${question}`;
    const parsed = await grokChatJson({
      system, user,
      images: [{ base64: file.buffer.toString("base64"), mimeType: file.mimetype }],
      model: grokVisionModel(), temperature: 0.2, maxTokens: 1300, timeoutMs: 90_000,
    });
    return res.json({ type: "image_answer", answer: String(parsed.answer || "") });
  } catch (error) {
    const err = error as { status?: number; message?: string };
    console.error("[RAG] ask-image error", JSON.stringify({ status: err?.status ?? null, message: err?.message ?? String(error) }));
    return res.status(500).json({ error: grokErrorMessage(error) });
  }
});

export default router;
