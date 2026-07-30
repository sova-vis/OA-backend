import { Router, Request, Response } from "express";
import multer from "multer";
import { grokEnabled, grokChatJson, grokVisionModel, grokErrorMessage } from "./lib/grok";

// The main "Ask AI" endpoints (/subjects, /query) are a thin proxy to the
// Past-Paper Chatbot's own Python service, which already does retrieval +
// generation + Drive-hosted page images - replacing the previous ~4000-line
// implementation that lived directly in this file. /ask-image is a separate,
// self-contained feature (photo/diagram Q&A via Grok vision) that doesn't
// touch the /query pipeline at all, so it's kept as-is alongside the proxy.

const router = Router();
const visionUpload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });

const CHATBOT_SERVICE_URL = (process.env.CHATBOT_SERVICE_URL || "http://127.0.0.1:8002").replace(/\/$/, "");

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
    const { question, history, mode, subject } = req.body as {
      question?: string; history?: unknown; mode?: string; subject?: string;
    };

    if (!question?.trim()) {
      return res.status(400).json({ error: "Question is required" });
    }

    const response = await fetch(`${CHATBOT_SERVICE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, mode, subject }),
      // Ask mode now walks through 3 full worked past-paper answers on top
      // of the explanation, which can genuinely take 45-60s+ from the LLM -
      // 60s was cutting it dangerously close, causing intermittent failures.
      signal: AbortSignal.timeout(120000),
    });

    if (!response.ok) {
      const body = await response.text();
      console.error(`[RAG] Chatbot service returned ${response.status}: ${body.slice(0, 300)}`);
      return res.status(502).json({ error: "Chatbot service error" });
    }

    const data = await response.json();
    return res.json(data);
  } catch (err: any) {
    console.error("[RAG] Failed to reach chatbot service:", err?.message || err);
    return res.status(502).json({
      error: "Could not reach the chatbot service. Is it running on " + CHATBOT_SERVICE_URL + "?",
    });
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
