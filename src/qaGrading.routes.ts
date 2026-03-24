import { Router, Request, Response } from 'express';
import Groq from 'groq-sdk';
import multer from 'multer';

type GradeLabel = 'fully_correct' | 'partially_correct' | 'weak';
type StatusLabel = 'accepted' | 'review_required' | 'failed';
type GradingSource = 'grok' | 'deterministic' | 'pipeline';

interface QaGradingRequest {
  question?: string;
  student_answer?: string;
  marking_scheme_answer?: string;
  subject?: string;
  year?: number;
  session?: string;
  paper?: string;
  variant?: string;
  question_id?: string;
  debug?: boolean;
}

interface QaGradingResponse {
  status: StatusLabel;
  score: number;
  score_percent: number;
  grade_label: GradeLabel;
  feedback: string;
  expected_points: string[];
  missing_points: string[];
  student_option: string | null;
  correct_option: string | null;
  grading_source: GradingSource;
  grading_model: string | null;
  question: string;
  student_answer: string;
  marking_scheme_answer: string;
  subject: string | null;
  timestamp: string;
}

interface ServiceEvaluateResponse {
  status?: StatusLabel;
  score?: number;
  score_percent?: number;
  grade_label?: GradeLabel;
  feedback?: string;
  expected_points?: string[];
  missing_points?: string[];
  student_answer?: string;
  marking_scheme_answer?: string;
}

interface QaModeAPreviewResponse {
  request_id: string;
  extracted_question_text: string;
  extracted_student_answer: string;
  normalized_question_text: string;
  normalized_student_answer: string;
  vision_confidence: number;
  vision_warnings: string[];
  match_confidence: number;
  top1_top2_margin: number;
  auto_accept_eligible: boolean;
  auto_accept_reason: string;
  top_alternatives: Array<{
    question_id: string;
    match_confidence: number;
    question_text: string;
    source_paper_reference: string;
  }>;
  debug_run_id?: string | null;
}

interface QaModeAConfirmRequest {
  question_text: string;
  student_answer?: string;
  subject?: string;
  year?: number;
  session?: string;
  paper?: string;
  variant?: string;
  question_id?: string;
  debug?: boolean;
}

interface EvalResult {
  score: number;
  score_percent: number;
  grade_label: GradeLabel;
  feedback: string;
  expected_points: string[];
  missing_points: string[];
  student_option: string | null;
  correct_option: string | null;
  grading_source: GradingSource;
  grading_model: string | null;
}

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
});

const groqApiKey = (process.env.GROQ_API_KEY || '').trim();
const groqModel = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';
const groq = groqApiKey ? new Groq({ apiKey: groqApiKey }) : null;
const qaServiceUrl = (process.env.QA_GRADING_SERVICE_URL || 'http://127.0.0.1:8001').trim();
const nodeEnv = (process.env.NODE_ENV || '').trim().toLowerCase();

function serviceUrl(path: string): string {
  const base = qaServiceUrl.endsWith('/') ? qaServiceUrl.slice(0, -1) : qaServiceUrl;
  return `${base}${path}`;
}

function toServiceConnectionMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  const lowered = message.toLowerCase();
  if (lowered.includes('econnrefused') || lowered.includes('fetch failed')) {
    const isLocalhostUrl =
      qaServiceUrl.includes('127.0.0.1') ||
      qaServiceUrl.includes('localhost');

    if (nodeEnv === 'production' && isLocalhostUrl) {
      return [
        `Cannot connect to Q/A grading service at ${qaServiceUrl}.`,
        'In production, localhost points to the same backend container only.',
        'Set QA_GRADING_SERVICE_URL to a deployed Subject-Grading service URL, or deploy backend with an embedded sidecar process that is actually running on that port.',
      ].join(' ');
    }

    return `Cannot connect to Q/A grading service at ${qaServiceUrl}. Start it with: cd OA-backend ; npm run qa-grading:service`;
  }
  return message;
}

async function proxyJson<TRequest, TResponse>(path: string, payload: TRequest): Promise<TResponse> {
  const response = await fetch(serviceUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const text = await response.text();
  let data: unknown = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Invalid JSON from QA grading service: ${text.slice(0, 300)}`);
  }

  if (!response.ok) {
    const errorMessage =
      typeof (data as { detail?: unknown; error?: unknown }).detail === 'string'
        ? ((data as { detail: string }).detail || '').trim()
        : typeof (data as { error?: unknown }).error === 'string'
          ? ((data as { error: string }).error || '').trim()
          : `QA grading service request failed with status ${response.status}`;
    throw new Error(errorMessage || `QA grading service request failed with status ${response.status}`);
  }

  return data as TResponse;
}

function parseIntegerLike(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isInteger(value)) return value;
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed) return undefined;
  const parsed = Number(trimmed);
  if (!Number.isInteger(parsed)) return undefined;
  return parsed;
}

function normalizeServiceEvaluateResponse(
  response: ServiceEvaluateResponse,
  request: {
    question: string;
    studentAnswer: string;
    markingSchemeAnswer: string;
    subject: string | null;
  }
): QaGradingResponse {
  const rawScore = Number(response.score);
  const score = Number.isFinite(rawScore) ? Math.max(0, Math.min(1, rawScore)) : 0;

  const scorePercent =
    Number.isFinite(Number(response.score_percent)) && typeof response.score_percent === 'number'
      ? response.score_percent
      : Number((score * 100).toFixed(2));

  const gradeLabel: GradeLabel =
    response.grade_label === 'fully_correct' ||
    response.grade_label === 'partially_correct' ||
    response.grade_label === 'weak'
      ? response.grade_label
      : gradeFromScore(score);

  const status: StatusLabel =
    response.status === 'accepted' || response.status === 'review_required' || response.status === 'failed'
      ? response.status
      : statusFromScore(score);

  return {
    status,
    score,
    score_percent: scorePercent,
    grade_label: gradeLabel,
    feedback: response.feedback || 'Grading completed from QA pipeline service.',
    expected_points: Array.isArray(response.expected_points) ? response.expected_points : [],
    missing_points: Array.isArray(response.missing_points) ? response.missing_points : [],
    student_option: null,
    correct_option: null,
    grading_source: 'pipeline',
    grading_model: null,
    question: request.question,
    student_answer: response.student_answer || request.studentAnswer,
    marking_scheme_answer: response.marking_scheme_answer || request.markingSchemeAnswer,
    subject: request.subject,
    timestamp: new Date().toISOString(),
  };
}

const MCQ_RE = /^\s*([ABCD])\s*$/i;
const MCQ_TOKEN_RE = /\b([ABCD])\b/i;
const MCQ_SCHEME_PATTERNS = [
  /\bcorrect(?:\s+answer|\s+option)?\s*(?:is|:)?\s*([ABCD])\b/i,
  /\banswer\s*[:\-]\s*([ABCD])\b/i,
  /\boption\s*([ABCD])\s*(?:is\s+correct|correct)\b/i,
];
const SPLIT_RE = /(?:\n+|[.;:])/;
const NON_ALNUM_RE = /[^a-z0-9\s]+/gi;
const SPACE_RE = /\s+/g;

const STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'that',
  'the',
  'to',
  'with',
]);

function normalizeText(text: string): string {
  const lowered = (text || '').toLowerCase();
  const cleaned = lowered.replace(NON_ALNUM_RE, ' ');
  return cleaned.replace(SPACE_RE, ' ').trim();
}

function tokenize(text: string): string[] {
  const normalized = normalizeText(text);
  if (!normalized) return [];
  return normalized.split(' ').filter((token) => token && !STOPWORDS.has(token));
}

function extractExpectedPoints(markingSchemeAnswer: string): string[] {
  const chunks = markingSchemeAnswer
    .split(SPLIT_RE)
    .map((part) => part.trim())
    .filter((part) => part.length >= 4);

  if (chunks.length > 0) {
    return chunks.slice(0, 10);
  }

  if (markingSchemeAnswer.trim()) {
    return [markingSchemeAnswer.trim()];
  }

  return [];
}

function extractStudentOption(studentAnswer: string): string | null {
  const text = (studentAnswer || '').trim().toUpperCase();
  if (!text) return null;

  const exact = text.match(MCQ_RE);
  if (exact?.[1]) {
    return exact[1].toUpperCase();
  }

  const token = text.match(MCQ_TOKEN_RE);
  if (token?.[1]) {
    return token[1].toUpperCase();
  }

  return null;
}

function extractSchemeOption(markingSchemeAnswer: string): string | null {
  const text = (markingSchemeAnswer || '').trim().toUpperCase();
  if (!text) return null;

  const exact = text.match(MCQ_RE);
  if (exact?.[1]) {
    return exact[1].toUpperCase();
  }

  for (const pattern of MCQ_SCHEME_PATTERNS) {
    const hit = text.match(pattern);
    if (hit?.[1]) {
      return hit[1].toUpperCase();
    }
  }

  return null;
}

function isMcq(markingSchemeAnswer: string): boolean {
  return extractSchemeOption(markingSchemeAnswer) !== null;
}

function gradeFromScore(score: number): GradeLabel {
  if (score >= 0.85) return 'fully_correct';
  if (score >= 0.45) return 'partially_correct';
  return 'weak';
}

function statusFromScore(score: number): StatusLabel {
  if (score >= 0.75) return 'accepted';
  if (score >= 0.4) return 'review_required';
  return 'failed';
}

function buildFeedback(score: number, expectedPoints: string[], missingPoints: string[]): string {
  if (score >= 0.85) {
    return 'Great work. Your answer aligns strongly with the marking scheme.';
  }

  if (score >= 0.45) {
    if (missingPoints.length > 0) {
      return `Good attempt. Add these points for a stronger answer: ${missingPoints
        .slice(0, 3)
        .join('; ')}.`;
    }
    return 'Good attempt. Your answer is partially aligned with the marking scheme.';
  }

  if (expectedPoints.length > 0) {
    return `Your answer is currently weak against the marking scheme. Focus on: ${expectedPoints
      .slice(0, 3)
      .join('; ')}.`;
  }

  return 'Your answer is currently weak against the marking scheme.';
}

function evaluateDeterministic(studentAnswer: string, markingSchemeAnswer: string): EvalResult {
  const correctOption = extractSchemeOption(markingSchemeAnswer);
  const studentOption = extractStudentOption(studentAnswer);

  if (isMcq(markingSchemeAnswer) && correctOption) {
    const isCorrect = studentOption === correctOption;
    const score = isCorrect ? 1 : 0;
    const expectedPoints = [`Correct option: ${correctOption}`];
    const missingPoints = isCorrect ? [] : [`Select option ${correctOption}.`];

    return {
      score,
      score_percent: Number((score * 100).toFixed(2)),
      grade_label: isCorrect ? 'fully_correct' : 'weak',
      feedback: isCorrect
        ? 'Correct choice. This matches the marking scheme.'
        : `Incorrect option. The expected option is ${correctOption}.`,
      expected_points: expectedPoints,
      missing_points: missingPoints,
      student_option: studentOption,
      correct_option: correctOption,
      grading_source: 'deterministic',
      grading_model: null,
    };
  }

  const expectedPoints = extractExpectedPoints(markingSchemeAnswer);
  if (expectedPoints.length === 0) {
    return {
      score: 0,
      score_percent: 0,
      grade_label: 'weak',
      feedback: 'Marking scheme answer is required for grading.',
      expected_points: [],
      missing_points: [],
      student_option: studentOption,
      correct_option: correctOption,
      grading_source: 'deterministic',
      grading_model: null,
    };
  }

  const studentTokens = tokenize(studentAnswer);
  const matchedPoints: string[] = [];
  const missingPoints: string[] = [];

  for (const point of expectedPoints) {
    const pointTokens = tokenize(point);
    if (pointTokens.length === 0) continue;

    const pointSet = new Set(pointTokens);
    const studentSet = new Set(studentTokens);
    const overlapSize = Array.from(pointSet).filter((token) => studentSet.has(token)).length;
    const coverage = overlapSize / pointSet.size;

    if (coverage >= 0.5) {
      matchedPoints.push(point);
    } else {
      missingPoints.push(point);
    }
  }

  const score = expectedPoints.length > 0 ? matchedPoints.length / expectedPoints.length : 0;
  const gradeLabel = gradeFromScore(score);

  return {
    score: Number(score.toFixed(4)),
    score_percent: Number((score * 100).toFixed(2)),
    grade_label: gradeLabel,
    feedback: buildFeedback(score, expectedPoints, missingPoints),
    expected_points: expectedPoints,
    missing_points: missingPoints,
    student_option: studentOption,
    correct_option: correctOption,
    grading_source: 'deterministic',
    grading_model: null,
  };
}

function parseJsonObject(raw: string): Record<string, unknown> | null {
  const text = (raw || '').trim();
  if (!text) return null;

  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Continue with object extraction fallback.
  }

  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try {
      const parsed = JSON.parse(text.slice(start, end + 1));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      return null;
    }
  }

  return null;
}

async function evaluateWithGroq(
  question: string,
  studentAnswer: string,
  markingSchemeAnswer: string,
  fallback: EvalResult
): Promise<EvalResult> {
  if (!groq) {
    return fallback;
  }

  try {
    const systemPrompt = [
      'You are an O/A Levels exam answer grader.',
      'Grade strictly against the supplied marking scheme.',
      'Return JSON only with these keys: score, grade_label, feedback, expected_points, missing_points, student_option, correct_option.',
      'score must be between 0 and 1.',
      'grade_label must be one of fully_correct, partially_correct, weak.',
      'expected_points and missing_points must be arrays of short strings.',
      'For MCQ, use a strict binary score using the correct option.',
    ].join(' ');

    const completion = await groq.chat.completions.create({
      model: groqModel,
      temperature: 0,
      messages: [
        { role: 'system', content: systemPrompt },
        {
          role: 'user',
          content: JSON.stringify({
            question,
            student_answer: studentAnswer,
            marking_scheme_answer: markingSchemeAnswer,
          }),
        },
      ],
    });

    const content = completion.choices?.[0]?.message?.content || '';
    const parsed = parseJsonObject(typeof content === 'string' ? content : '');
    if (!parsed) {
      return fallback;
    }

    const rawScore = Number(parsed.score);
    const clampedScore = Number.isFinite(rawScore)
      ? Math.max(0, Math.min(1, rawScore))
      : fallback.score;

    const rawLabel = String(parsed.grade_label || '').trim().toLowerCase();
    const gradeLabel: GradeLabel =
      rawLabel === 'fully_correct' || rawLabel === 'partially_correct' || rawLabel === 'weak'
        ? (rawLabel as GradeLabel)
        : gradeFromScore(clampedScore);

    const expectedPoints = Array.isArray(parsed.expected_points)
      ? parsed.expected_points.map((item) => String(item).trim()).filter(Boolean).slice(0, 10)
      : fallback.expected_points;

    const missingPoints = Array.isArray(parsed.missing_points)
      ? parsed.missing_points.map((item) => String(item).trim()).filter(Boolean).slice(0, 10)
      : fallback.missing_points;

    const studentOption =
      typeof parsed.student_option === 'string' && parsed.student_option.trim()
        ? parsed.student_option.trim().toUpperCase()
        : fallback.student_option;

    const correctOption =
      typeof parsed.correct_option === 'string' && parsed.correct_option.trim()
        ? parsed.correct_option.trim().toUpperCase()
        : fallback.correct_option;

    const feedback =
      typeof parsed.feedback === 'string' && parsed.feedback.trim()
        ? parsed.feedback.trim()
        : buildFeedback(clampedScore, expectedPoints, missingPoints);

    return {
      score: Number(clampedScore.toFixed(4)),
      score_percent: Number((clampedScore * 100).toFixed(2)),
      grade_label: gradeLabel,
      feedback,
      expected_points: expectedPoints,
      missing_points: missingPoints,
      student_option: studentOption,
      correct_option: correctOption,
      grading_source: 'grok',
      grading_model: groqModel,
    };
  } catch {
    return fallback;
  }
}

router.get('/health', (_req: Request, res: Response) => {
  return res.json({
    status: 'ok',
    route: 'qa-grading',
    groq_enabled: Boolean(groq),
    qa_service_url: qaServiceUrl,
  });
});

router.get('/ready', async (_req: Request, res: Response) => {
  try {
    const response = await fetch(serviceUrl('/oa-level/ready'));
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(503).json({
      ready: false,
      warmup_done: false,
      error: toServiceConnectionMessage(error),
    });
  }
});

router.post('/evaluate', async (req: Request, res: Response) => {
  try {
    const body = (req.body ?? {}) as QaGradingRequest;
    const question = (body.question || '').trim();
    const studentAnswer = (body.student_answer || '').trim();
    const markingSchemeAnswer = (body.marking_scheme_answer || '').trim();
    const subject = (body.subject || '').trim() || null;

    if (!question) {
      return res.status(400).json({ error: 'question is required' });
    }

    if (!studentAnswer) {
      return res.status(400).json({ error: 'student_answer is required' });
    }

    // Prefer full pipeline service when available.
    try {
      const serviceResponse = await proxyJson<QaGradingRequest, ServiceEvaluateResponse>('/oa-level/evaluate', {
        question,
        student_answer: studentAnswer,
        subject: subject || undefined,
        year: parseIntegerLike(body.year),
        session: typeof body.session === 'string' ? body.session.trim() || undefined : undefined,
        paper: typeof body.paper === 'string' ? body.paper.trim() || undefined : undefined,
        variant: typeof body.variant === 'string' ? body.variant.trim() || undefined : undefined,
        question_id: typeof body.question_id === 'string' ? body.question_id.trim() || undefined : undefined,
        debug: Boolean(body.debug),
      });

      return res.json(
        normalizeServiceEvaluateResponse(serviceResponse, {
          question,
          studentAnswer,
          markingSchemeAnswer,
          subject,
        })
      );
    } catch (serviceError) {
      console.warn('Q/A typed proxy failed, using local fallback:', serviceError);
    }

    if (!markingSchemeAnswer) {
      return res.status(400).json({
        error:
          'marking_scheme_answer is required for local grading fallback. Keep QA grading service running for full pipeline grading without manual mark scheme.',
      });
    }

    const deterministic = evaluateDeterministic(studentAnswer, markingSchemeAnswer);
    const graded = await evaluateWithGroq(
      question,
      studentAnswer,
      markingSchemeAnswer,
      deterministic
    );

    const response: QaGradingResponse = {
      status: statusFromScore(graded.score),
      score: graded.score,
      score_percent: graded.score_percent,
      grade_label: graded.grade_label,
      feedback: graded.feedback,
      expected_points: graded.expected_points,
      missing_points: graded.missing_points,
      student_option: graded.student_option,
      correct_option: graded.correct_option,
      grading_source: graded.grading_source,
      grading_model: graded.grading_model,
      question,
      student_answer: studentAnswer,
      marking_scheme_answer: markingSchemeAnswer,
      subject,
      timestamp: new Date().toISOString(),
    };

    return res.json(response);
  } catch (error) {
    console.error('Q/A grading error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

router.post('/evaluate-from-image/preview', upload.single('file'), async (req: Request, res: Response) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'file is required' });
    }

    const formData = new FormData();
    const mimeType = (file.mimetype || 'application/octet-stream').trim();
    const blob = new Blob([file.buffer], { type: mimeType });
    formData.append('file', blob, file.originalname || 'upload.bin');

    const body = (req.body ?? {}) as Record<string, unknown>;
    const passthroughFields = ['subject', 'session', 'paper', 'variant', 'question_id', 'debug', 'page_number'];
    for (const field of passthroughFields) {
      const value = body[field];
      if (typeof value === 'string' && value.trim()) {
        formData.append(field, value.trim());
      }
    }
    const year = parseIntegerLike(body.year);
    if (typeof year === 'number') {
      formData.append('year', String(year));
    }

    const response = await fetch(serviceUrl('/oa-level/evaluate-from-image/preview'), {
      method: 'POST',
      body: formData,
    });

    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    return res.status(response.status).json(data as QaModeAPreviewResponse | Record<string, unknown>);
  } catch (error) {
    console.error('Q/A grading preview proxy error:', error);
    return res.status(500).json({
      error: toServiceConnectionMessage(error),
    });
  }
});

router.post('/evaluate-from-image/confirm', async (req: Request, res: Response) => {
  try {
    const body = (req.body ?? {}) as QaModeAConfirmRequest;
    if (!body.question_text || !body.question_text.trim()) {
      return res.status(400).json({ error: 'question_text is required' });
    }

    const response = await proxyJson<QaModeAConfirmRequest, ServiceEvaluateResponse>(
      '/oa-level/evaluate-from-image/confirm',
      body
    );

    return res.json(
      normalizeServiceEvaluateResponse(response, {
        question: body.question_text.trim(),
        studentAnswer: (body.student_answer || '').trim(),
        markingSchemeAnswer: '',
        subject: (body.subject || '').trim() || null,
      })
    );
  } catch (error) {
    console.error('Q/A grading confirm proxy error:', error);
    return res.status(500).json({
      error: toServiceConnectionMessage(error),
    });
  }
});

export default router;
