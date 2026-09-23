import { Router, Request, Response } from "express";
import multer from "multer";
import { grokEnabled, grokChatJson, grokVisionModel, grokErrorMessage } from "./lib/grok";
import { askAi, normalizeMath, plainMath } from "./lib/askai/generate";
import { chatText, diagnoseProviders } from "./lib/askai/llm";
import { ALL_SUBJECTS, AskAiIndexUnavailable } from "./lib/askai/retrieve";
import { supabase } from "./lib/supabase";
import type { AuthenticatedRequest } from "./lib/clerkAuth";

// Ask AI runs IN-PROCESS (no separate Python service) and LLM-first: /query plans
// the request with the LLM (search or not, how to search), embeds with bge-base,
// searches the pgvector index in Supabase (match_ask_ai_chunks RPC), has the LLM
// rank the matches (best / conceptual / related), then answers — see
// src/lib/askai/*. /ask-image is a separate photo/diagram Q&A via Grok vision.

const router = Router();
const visionUpload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });

router.get("/subjects", async (_req: Request, res: Response) => {
  return res.json(ALL_SUBJECTS.map((s) => ({ name: s })));
});

/**
 * GET /rag/diag — staff-only (teacher/admin profile): pings every configured AI
 * provider and reports what it answers, so a dead key or retired model id on
 * the deployed box can be diagnosed without log access. No secrets in the output.
 */
router.get("/diag", async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    const prof = clerkId
      ? await supabase.from("profiles").select("role").eq("clerk_id", clerkId).maybeSingle()
      : null;
    const role = prof?.data?.role;
    if (role !== "teacher" && role !== "admin") return res.status(403).json({ error: "forbidden" });
    const providers = await diagnoseProviders();
    return res.json({
      configured: { groq: Boolean(process.env.GROQ_API_KEY?.trim()), groqModel: process.env.GROQ_MODEL?.trim() || null, xai: Boolean((process.env.XAI_API_KEY || process.env.GROK_API_KEY || "").trim()) },
      providers,
    });
  } catch (err: any) {
    return res.status(500).json({ error: String(err?.message || err).slice(0, 200) });
  }
});

router.post("/query", async (req: Request, res: Response) => {
  try {
    const { question, mode, subject, level, history } = req.body as {
      question?: string; mode?: string; subject?: string; level?: string; history?: unknown;
    };
    if (!question?.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }
    const out = await askAi({
      query: question.slice(0, 4000), mode: mode === "find" ? "find" : "ask",
      subject: subject || null, level: level || null, history,
    });
    return res.json({ ...out, source_type: out.citations.length ? "past_paper" : "none" });
  } catch (err: any) {
    if (err instanceof AskAiIndexUnavailable) {
      console.error("[RAG] index unavailable:", err.message);
      return res.status(503).json({ error: "Ask AI is still being set up — the question index isn't ready yet." });
    }
    const msg = String(err?.message || err);
    console.error("[RAG] query failed:", msg);
    if (/AI providers? (are|is) currently unavailable|No AI provider/i.test(msg)) {
      return res.status(503).json({ error: "The AI model is busy or unavailable right now — please try again in a moment." });
    }
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
    const answer = plainMath(await chatText(system, user, { tier: "smart", maxTokens: 600, temperature: 0.2 }));
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
      "Formatting: Markdown; mathematics as LaTeX between dollar signs — $...$ inline or $$...$$ on its own single line for a key step.",
      "Dollar signs are ONLY for real mathematical expressions (fractions, powers, roots, subscripts). NEVER wrap a plain number or ordinary word in dollar signs — write 'the answer is 3', not 'the answer is $3$'.",
      "Chemical equations in plain text with -> arrows, not LaTeX.",
      "Return JSON ONLY: { \"answer\": string }.",
    ].join(" ");
    const user = `${subject ? "Subject: " + subject + ". " : ""}${question}`;
    const parsed = await grokChatJson({
      system, user,
      images: [{ base64: file.buffer.toString("base64"), mimeType: file.mimetype }],
      model: grokVisionModel(), temperature: 0.2, maxTokens: 1300, timeoutMs: 90_000,
    });
    return res.json({ type: "image_answer", answer: normalizeMath(String(parsed.answer || "")) });
  } catch (error) {
    const err = error as { status?: number; message?: string };
    console.error("[RAG] ask-image error", JSON.stringify({ status: err?.status ?? null, message: err?.message ?? String(error) }));
    return res.status(500).json({ error: grokErrorMessage(error) });
  }
});

export default router;
