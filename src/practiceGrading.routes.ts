import { Router, Response } from 'express';
import multer from 'multer';
import { AuthenticatedRequest, clerkAuth } from './lib/clerkAuth';
import { grokEnabled, grokVisionModel, grokTextModel, grokErrorMessage, grokHttpStatus, GrokError } from './lib/grok';
import {
  PracticeProgressDoc, GradedQuestion, ExtractionSummary,
  ensureBucket, isValidPaperKey, readDoc, writeDoc, signUploads, loadUploadPages,
} from './lib/practiceStore';
import { GradeQuestion, extractHandwrittenAnswers } from './lib/handwrittenExtraction';
import {
  gradeOne, gradeTyped, gradeHandwritten, gradeExtracted, buildReport, typedAnswersFromGraded,
} from './lib/practiceMarking';
import { PageImage, isSupportedImageType, isPracticeUploadType, rasterizePdfPages } from './lib/pdfPages';
import { SolveMode } from './lib/practiceStore';

/**
 * HTTP surface for practice-paper marking. All scoring lives in
 * lib/practiceMarking (one engine for typed and uploaded attempts); handwritten
 * uploads are transcribed first by lib/handwrittenExtraction.
 *
 *   POST /practice-grading/grade            whole paper, typed or handwritten
 *   POST /practice-grading/grade-one        one question, typed (topic drill)
 *   POST /practice-grading/grade-one-image  one question, from a photo
 */

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 15 * 1024 * 1024 } });

const MAX_QUESTIONS = 80;

const router = Router();

router.post('/grade', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const body = (req.body ?? {}) as {
      paperKey?: string; subject?: string; solveMode?: SolveMode; questions?: GradeQuestion[];
    };
    const paperKey = body.paperKey;
    if (!isValidPaperKey(paperKey)) return res.status(400).json({ error: 'Invalid paper key' });

    const subject = (body.subject || paperKey.split('|')[0] || '').trim();
    const solveMode: SolveMode = body.solveMode === 'handwritten' ? 'handwritten' : 'digital';
    const questions = Array.isArray(body.questions) ? body.questions.slice(0, MAX_QUESTIONS) : [];
    if (questions.length === 0) return res.status(400).json({ error: 'No questions to grade' });

    await ensureBucket();

    // Split into instant (MCQ) and AI (written) work.
    const needsAi = questions.some((q) => q.type !== 'mcq' || solveMode === 'handwritten');
    if (needsAi && !grokEnabled()) {
      return res.status(503).json({ error: grokErrorMessage(new GrokError('', 'no_key')), code: 'no_key' });
    }

    let graded: GradedQuestion[];
    let model = grokTextModel();
    let extraction: ExtractionSummary | undefined;

    if (solveMode === 'handwritten') {
      const doc = await readDoc(clerkId, paperKey);
      const uploads = doc?.uploads ?? [];
      if (uploads.length === 0) {
        return res.status(400).json({ error: 'Upload a photo, scan or PDF of your handwritten answers first.' });
      }

      const loaded = await loadUploadPages(uploads, undefined, { clerkId, paperKey });
      if (loaded.pages.length === 0) {
        // Every file failed. Say which, and why — the previous behaviour told the
        // student to upload something they had already uploaded.
        const storageGone = loaded.skipped.some((reason) => reason.includes('could not be downloaded from storage'));
        const detail = loaded.skipped.length ? ` ${loaded.skipped.join(' ')}` : '';
        return res.status(400).json({
          error: storageGone
            ? `We couldn't read your upload from storage.${detail} Remove the file and upload it again.`
            : `None of your uploaded files could be read.${detail} Upload a clear JPG, PNG or PDF of your answers.`,
          code: 'no_readable_pages',
          skipped: loaded.skipped,
        });
      }

      model = grokVisionModel();
      // Drop any typed answers the client sent — uploaded papers are marked from
      // the transcription only, so a leftover typed draft cannot mix into marks.
      const fromUpload = questions.map((question) => ({
        ...question, studentOption: null, studentParts: {}, studentAnswer: null,
      }));
      const result = await gradeHandwritten(subject, fromUpload, loaded.pages, fromUpload.every((q) => q.type === 'mcq'));
      graded = result.graded;
      // Files we could not use are warnings on the report, not silent zeros.
      extraction = { ...result.extraction, warnings: [...loaded.skipped, ...result.extraction.warnings] };
    } else {
      graded = await gradeTyped(subject, questions);
    }

    const report = buildReport(graded, solveMode, model, extraction);

    // Persist the report onto the progress doc and mark the paper completed.
    // Handwritten attempts also store the transcription in the same answer slots
    // Solve here uses, so the report UI can render both paths identically.
    const segments = paperKey.split('|');
    const existing = await readDoc(clerkId, paperKey);
    const now = new Date().toISOString();
    const extractedAnswers = solveMode === 'handwritten' ? typedAnswersFromGraded(questions, graded) : null;
    const answeredCount = graded.filter((g) => {
      if (g.extractionFlag === 'blank' || g.extractionFlag === 'not_found') return false;
      if (g.marksWithheld) return true;
      if (g.verdict === 'unanswered' && !g.extractedAnswer) return false;
      return true;
    }).length;
    const totalCount = graded.length;
    const doc: PracticeProgressDoc = existing
      ? {
          ...existing, report, status: 'completed', answeredCount, totalCount, updatedAt: now,
          ...(extractedAnswers ? { answers: extractedAnswers } : {}),
        }
      : {
          paperKey, subject, year: segments[1], session: segments[2], paper: segments[3], variant: segments[4],
          isMcq: questions.every((q) => q.type === 'mcq'), solveMode, status: 'completed',
          answers: extractedAnswers ?? { mcq: {}, parts: {} }, uploads: [], report,
          answeredCount, totalCount,
          timerDurationSeconds: 0, timerElapsedSeconds: 0, startedAt: now, updatedAt: now,
        };
    await writeDoc(clerkId, doc);

    return res.json({ report, item: { ...doc, uploads: await signUploads(doc.uploads ?? []) } });
  } catch (error) {
    console.error('Practice grading error:', error);
    const message = grokErrorMessage(error);
    return res.status(grokHttpStatus(error)).json({ error: message });
  }
});

/**
 * POST /practice-grading/grade-one  (Clerk auth)
 * Grade a single question against its own marking scheme (topic-drill mode).
 * Stateless — nothing is persisted.
 */
router.post('/grade-one', clerkAuth, async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const body = (req.body ?? {}) as { subject?: string; question?: GradeQuestion };
    const question = body.question;
    if (!question || !question.id) return res.status(400).json({ error: 'question is required' });
    const subject = (body.subject || '').trim();

    if (question.type !== 'mcq' && !grokEnabled()) {
      return res.status(503).json({ error: grokErrorMessage(new GrokError('', 'no_key')), code: 'no_key' });
    }
    return res.json({ result: await gradeOne(subject, question) });
  } catch (error) {
    console.error('Single-question grading error:', error);
    const message = grokErrorMessage(error);
    return res.status(grokHttpStatus(error)).json({ error: message });
  }
});

/**
 * POST /practice-grading/grade-one-image  (Clerk auth, multipart)
 * Grade one topic-drill question from a photo of a handwritten answer. The photo
 * is transcribed and then marked by the shared scorer, so it agrees with the
 * typed path. Stateless — the image is read in-memory, not stored.
 * Fields: subject, question (JSON string); file: the answer image.
 */
router.post('/grade-one-image', clerkAuth, upload.single('file'), async (req: AuthenticatedRequest, res: Response) => {
  try {
    const clerkId = req.auth?.clerkId;
    if (!clerkId) return res.status(401).json({ error: 'Unauthorized' });

    const subject = String(req.body?.subject || '').trim();
    let question: GradeQuestion;
    try {
      question = JSON.parse(String(req.body?.question || '')) as GradeQuestion;
    } catch {
      return res.status(400).json({ error: 'Invalid question payload' });
    }
    if (!question || !question.id) return res.status(400).json({ error: 'question is required' });

    const file = req.file;
    if (!file) return res.status(400).json({ error: 'No file provided' });
    if (!isPracticeUploadType(file.mimetype, file.originalname) && !isSupportedImageType(file.mimetype)) {
      return res.status(400).json({ error: 'Upload a photo (JPG or PNG) or a PDF of your answer.' });
    }
    if (!file.buffer?.byteLength) return res.status(400).json({ error: 'That file is empty.' });
    if (!grokEnabled()) {
      return res.status(503).json({ error: grokErrorMessage(new GrokError('', 'no_key')), code: 'no_key' });
    }

    const isPdf = (file.mimetype || '').toLowerCase() === 'application/pdf'
      || (file.originalname || '').toLowerCase().endsWith('.pdf');
    let pages: PageImage[];
    if (isPdf) {
      const raster = await rasterizePdfPages(file.buffer, 1, 8, file.originalname);
      if (raster.pages.length === 0) {
        return res.status(400).json({ error: 'That PDF has no readable pages.' });
      }
      pages = raster.pages;
    } else {
      const mime = isSupportedImageType(file.mimetype) ? file.mimetype : 'image/jpeg';
      pages = [{
        base64: file.buffer.toString('base64'), mimeType: mime, page: 1, source: file.originalname,
      }];
    }
    const read = await extractHandwrittenAnswers([question], pages, {
      isMcqPaper: question.type === 'mcq',
      singleQuestion: true,
    });
    const result = await gradeExtracted(subject, question, read.byQuestionId.get(question.id));
    return res.json({ result });
  } catch (error) {
    console.error('Single-question image grading error:', error);
    const message = grokErrorMessage(error);
    return res.status(grokHttpStatus(error)).json({ error: message });
  }
});

router.get('/health', (_req, res: Response) => {
  res.json({ status: 'ok', grok_enabled: grokEnabled(), text_model: grokTextModel(), vision_model: grokVisionModel() });
});

export default router;
