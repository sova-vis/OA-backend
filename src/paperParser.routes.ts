import { Router, Request, Response } from 'express';
import Groq from 'groq-sdk';
import multer from 'multer';
import { PDFParse } from 'pdf-parse';
import { getFileMetadata, getFileStream, listFoldersAndFiles } from './lib/googleDrive';

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

interface ParsedPaperQuestion {
  id: string;
  number: string;
  stimulus: string | null;
  text: string;
  marks: number | null;
  topic: string | null;
  subquestions: Array<{
    id: string;
    label: string;
    stimulus: string | null;
    text: string;
    marks: number | null;
    answer_space_hint: string;
  }>;
  answer_space_hint: string;
  diagrams_or_sources: string[];
}

interface ParsedPaper {
  title: string;
  subject: string | null;
  syllabus_code: string | null;
  year: number | null;
  session: string | null;
  paper: string | null;
  variant: string | null;
  duration: string | null;
  total_marks: number | null;
  instructions: string[];
  source_materials: Array<{
    id: string;
    title: string;
    text: string;
    used_by_question_ids: string[];
  }>;
  sections: Array<{
    title: string;
    instructions: string[];
    question_ids: string[];
  }>;
  questions: ParsedPaperQuestion[];
}

const groqApiKey = (process.env.GROQ_API_KEY || '').trim();
const groqModel = (process.env.GROQ_PAPER_PARSER_MODEL || process.env.GROQ_MODEL || 'llama-3.3-70b-versatile').trim();
const groq = groqApiKey ? new Groq({ apiKey: groqApiKey }) : null;

function parseJsonObject(raw: string): Record<string, unknown> | null {
  const text = raw.trim();
  if (!text) return null;

  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
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

async function streamToBuffer(stream: NodeJS.ReadableStream): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

async function extractPdfText(buffer: Buffer): Promise<string> {
  const parser = new PDFParse({ data: buffer });
  try {
    const result = await parser.getText();
    const pageText = result.pages
      .map((_, index) => {
        const text = result.getPageText(index + 1).trim();
        return text ? `\n\n--- PAGE ${index + 1} ---\n${text}` : '';
      })
      .join('');

    return (pageText || result.text)
      .replace(/\u0000/g, '')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{4,}/g, '\n\n\n')
      .trim();
  } finally {
    await parser.destroy().catch(() => undefined);
  }
}

function normalizeParsedPaper(input: Record<string, unknown>): ParsedPaper {
  const questions = Array.isArray(input.questions) ? input.questions : [];
  return {
    title: typeof input.title === 'string' && input.title.trim() ? input.title.trim() : 'Parsed past paper',
    subject: typeof input.subject === 'string' && input.subject.trim() ? input.subject.trim() : null,
    syllabus_code:
      typeof input.syllabus_code === 'string' && input.syllabus_code.trim()
        ? input.syllabus_code.trim()
        : null,
    year: typeof input.year === 'number' && Number.isInteger(input.year) ? input.year : null,
    session: typeof input.session === 'string' && input.session.trim() ? input.session.trim() : null,
    paper: typeof input.paper === 'string' && input.paper.trim() ? input.paper.trim() : null,
    variant: typeof input.variant === 'string' && input.variant.trim() ? input.variant.trim() : null,
    duration: typeof input.duration === 'string' && input.duration.trim() ? input.duration.trim() : null,
    total_marks:
      typeof input.total_marks === 'number' && Number.isInteger(input.total_marks)
        ? input.total_marks
        : null,
    instructions: Array.isArray(input.instructions)
      ? input.instructions.map((item) => String(item).trim()).filter(Boolean)
      : [],
    source_materials: Array.isArray(input.source_materials)
      ? input.source_materials.map((source: any, index) => ({
          id: typeof source?.id === 'string' && source.id.trim() ? source.id.trim() : `source${index + 1}`,
          title:
            typeof source?.title === 'string' && source.title.trim()
              ? source.title.trim()
              : `Source ${index + 1}`,
          text: typeof source?.text === 'string' ? source.text.trim() : '',
          used_by_question_ids: Array.isArray(source?.used_by_question_ids)
            ? source.used_by_question_ids.map((item: unknown) => String(item).trim()).filter(Boolean)
            : [],
        })).filter((source) => source.text || source.title)
      : [],
    sections: Array.isArray(input.sections)
      ? input.sections.map((section: any) => ({
          title: typeof section?.title === 'string' && section.title.trim() ? section.title.trim() : 'Section',
          instructions: Array.isArray(section?.instructions)
            ? section.instructions.map((item: unknown) => String(item).trim()).filter(Boolean)
            : [],
          question_ids: Array.isArray(section?.question_ids)
            ? section.question_ids.map((item: unknown) => String(item).trim()).filter(Boolean)
            : [],
        }))
      : [],
    questions: questions.map((question: any, index) => ({
      id: typeof question?.id === 'string' && question.id.trim() ? question.id.trim() : `q${index + 1}`,
      number:
        typeof question?.number === 'string' && question.number.trim()
          ? question.number.trim()
          : `${index + 1}`,
      stimulus:
        typeof question?.stimulus === 'string' && question.stimulus.trim()
          ? question.stimulus.trim()
          : null,
      text: typeof question?.text === 'string' ? question.text.trim() : '',
      marks: typeof question?.marks === 'number' && Number.isInteger(question.marks) ? question.marks : null,
      topic: typeof question?.topic === 'string' && question.topic.trim() ? question.topic.trim() : null,
      subquestions: Array.isArray(question?.subquestions)
        ? question.subquestions.map((subquestion: any, subIndex: number) => ({
            id:
              typeof subquestion?.id === 'string' && subquestion.id.trim()
                ? subquestion.id.trim()
                : `q${index + 1}_${subIndex + 1}`,
            label:
              typeof subquestion?.label === 'string' && subquestion.label.trim()
                ? subquestion.label.trim()
                : String.fromCharCode(97 + subIndex),
            stimulus:
              typeof subquestion?.stimulus === 'string' && subquestion.stimulus.trim()
                ? subquestion.stimulus.trim()
                : null,
            text: typeof subquestion?.text === 'string' ? subquestion.text.trim() : '',
            marks:
              typeof subquestion?.marks === 'number' && Number.isInteger(subquestion.marks)
                ? subquestion.marks
                : null,
            answer_space_hint:
              typeof subquestion?.answer_space_hint === 'string'
                ? subquestion.answer_space_hint.trim()
                : '',
          }))
        : [],
      answer_space_hint:
        typeof question?.answer_space_hint === 'string' ? question.answer_space_hint.trim() : '',
      diagrams_or_sources: Array.isArray(question?.diagrams_or_sources)
        ? question.diagrams_or_sources.map((item: unknown) => String(item).trim()).filter(Boolean)
        : [],
    })),
  };
}

async function structurePaperText(params: {
  filename: string;
  text: string;
  companionFilenames?: string[];
}): Promise<ParsedPaper> {
  if (!groq) {
    throw new Error('GROQ_API_KEY is not configured on the backend.');
  }

  const trimmedText = params.text.slice(0, 115000);
  if (trimmedText.length < 80) {
    throw new Error('Could not extract enough readable text from this PDF.');
  }

  const systemPrompt = [
    'You are an exam paper structuring assistant.',
    'Return JSON only.',
    'Do not answer any questions.',
    'Extract the uploaded past paper text into this exact JSON shape:',
    '{ "title": string, "subject": string|null, "syllabus_code": string|null, "year": number|null, "session": string|null, "paper": string|null, "variant": string|null, "duration": string|null, "total_marks": number|null, "instructions": string[], "source_materials": [{ "id": string, "title": string, "text": string, "used_by_question_ids": string[] }], "sections": [{ "title": string, "instructions": string[], "question_ids": string[] }], "questions": [{ "id": string, "number": string, "stimulus": string|null, "text": string, "marks": number|null, "topic": string|null, "subquestions": [{ "id": string, "label": string, "stimulus": string|null, "text": string, "marks": number|null, "answer_space_hint": string }], "answer_space_hint": string, "diagrams_or_sources": string[] }] }',
    'Preserve every question and subquestion in the exact original order. Use null for unknown metadata and empty arrays where appropriate.',
    'Important: full reading passages, inserts, source texts, case studies, articles, letters, reports, and long extracts must be captured in source_materials, not duplicated into every question.',
    'If the paper references Passage 1, Passage 2, Insert, Source A, Figure 1, etc., create a source_materials entry containing the full available text/description and connect it to related question ids.',
    'If a source is referenced but not present in the supplied text, add a source_materials entry with its title and text set to "Source material not included in the selected PDF."',
    'Scenario text that belongs only to one numbered question can go in question stimulus.',
    'If a scenario belongs to several subquestions under one numbered question, put it in the parent question stimulus and keep subquestion stimulus null unless the subquestion has extra context.',
    'If a source/scenario belongs only to one subquestion, put it in that subquestion stimulus.',
    'The question text should contain the actual command/question only; do not lose the scenario.',
    'When a page break splits a question, merge it back into the same question.',
    'Do not combine separate numbered questions. Do not invent questions.',
    'Do not create duplicate entries for the same numbered question. If question 3 has parts (a), (b), (c), create one question with number "3" and put the parts in subquestions.',
    'If the paper labels parts as 3(a), 3(b), 3(c), keep question.number as "3" and subquestion labels as "a", "b", "c".',
    'Use concise answer_space_hint values such as short answer, calculation space, essay response, diagram/table.',
    'If figures, maps, source extracts, diagrams, tables, or graphs are referenced, summarize them in diagrams_or_sources.',
  ].join(' ');

  const completion = await groq.chat.completions.create({
    model: groqModel,
    temperature: 0,
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: systemPrompt },
      {
        role: 'user',
        content: JSON.stringify({
          filename: params.filename,
          companion_filenames: params.companionFilenames ?? [],
          extracted_pdf_text: trimmedText,
        }),
      },
    ],
  });

  const content = completion.choices?.[0]?.message?.content || '';
  const parsed = parseJsonObject(typeof content === 'string' ? content : '');
  if (!parsed) {
    throw new Error('Groq did not return parseable JSON.');
  }

  return normalizeParsedPaper(parsed);
}

async function parseBuffer(buffer: Buffer, filename: string) {
  const text = await extractPdfText(buffer);
  const paper = await structurePaperText({ filename, text });
  return {
    paper,
    raw_json: paper,
    model: groqModel,
    provider: 'groq',
    filename,
    parsed_at: new Date().toISOString(),
  };
}

function scoreCompanionCandidate(selectedName: string, candidateName: string): number {
  const selected = selectedName.toLowerCase();
  const candidate = candidateName.toLowerCase();
  if (selected === candidate) return -100;
  if (!candidate.endsWith('.pdf')) return -100;

  let score = 0;
  if (/(insert|source|passage|reading|text|resource)/i.test(candidate)) score += 10;
  if (/(qp|question)/i.test(candidate)) score -= 4;
  if (/(ms|mark|scheme|er|examiner)/i.test(candidate)) score -= 10;

  const selectedTokens = selected.match(/[a-z]+|\d+/g) ?? [];
  const candidateTokens = new Set(candidate.match(/[a-z]+|\d+/g) ?? []);
  for (const token of selectedTokens) {
    if (token.length >= 2 && candidateTokens.has(token)) score += 1;
  }

  return score;
}

async function collectDrivePaperBundle(fileId: string): Promise<{
  filename: string;
  buffer: Buffer;
  companionFilenames: string[];
}> {
  const metadata = await getFileMetadata(fileId);
  if (!metadata.mimeType?.includes('pdf')) {
    throw new Error('Only PDF past papers are supported.');
  }

  const mainStream = await getFileStream(fileId);
  const buffers = [await streamToBuffer(mainStream)];
  const filenames = [metadata.name || 'past-paper.pdf'];
  const companionFilenames: string[] = [];

  const parentId = metadata.parents?.[0];
  if (parentId) {
    const siblings = await listFoldersAndFiles(parentId);
    const candidates = siblings
      .filter((item) => !item.isFolder && item.id !== fileId && item.mimeType?.includes('pdf'))
      .map((item) => ({ item, score: scoreCompanionCandidate(metadata.name || '', item.name || '') }))
      .filter((candidate) => candidate.score >= 10)
      .sort((a, b) => b.score - a.score)
      .slice(0, 2);

    for (const candidate of candidates) {
      const stream = await getFileStream(candidate.item.id);
      buffers.push(await streamToBuffer(stream));
      filenames.push(candidate.item.name);
      companionFilenames.push(candidate.item.name);
    }
  }

  return {
    filename: filenames.join(' + '),
    buffer: Buffer.concat(buffers),
    companionFilenames,
  };
}

router.post('/parse', upload.single('file'), async (req: Request, res: Response) => {
  try {
    const file = req.file;
    if (!file) {
      return res.status(400).json({ error: 'file is required' });
    }

    const mimeType = file.mimetype || 'application/pdf';
    if (!mimeType.includes('pdf')) {
      return res.status(400).json({ error: 'Only PDF past papers are supported.' });
    }

    const result = await parseBuffer(file.buffer, file.originalname || 'past-paper.pdf');
    return res.json(result);
  } catch (error) {
    console.error('Paper parser upload error:', error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to parse past paper.',
    });
  }
});

router.post('/parse-drive', async (req: Request, res: Response) => {
  try {
    const body = (req.body ?? {}) as { fileId?: unknown };
    const fileId = typeof body.fileId === 'string' ? body.fileId.trim() : '';
    if (!fileId) {
      return res.status(400).json({ error: 'fileId is required' });
    }

    const bundle = await collectDrivePaperBundle(fileId);
    const text = await extractPdfText(bundle.buffer);
    const paper = await structurePaperText({
      filename: bundle.filename,
      text,
      companionFilenames: bundle.companionFilenames,
    });
    const result = {
      paper,
      raw_json: paper,
      model: groqModel,
      provider: 'groq',
      filename: bundle.filename,
      companion_files: bundle.companionFilenames,
      parsed_at: new Date().toISOString(),
    };
    return res.json({
      ...result,
      drive_file_id: fileId,
    });
  } catch (error) {
    console.error('Paper parser Drive error:', error);
    return res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to parse selected past paper.',
    });
  }
});

export default router;
