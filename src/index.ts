import cors from 'cors';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

import 'dotenv/config';
import express, { Express, Request, Response } from 'express';
import authRoutes from './auth.routes';
import adminRoutes from './admin.routes';
import papersRoutes from './papers.routes';
import ragRoutes from './rag.routes';
import contentRoutes from './content.routes';
import qaGradingRoutes from './qaGrading.routes';
import paperTrackingRoutes from './paperTracking.routes';
import practiceProgressRoutes from './practiceProgress.routes';
import uploadCheckRoutes from './uploadCheck.routes';
import insightsRoutes from './insights.routes';
import practiceGradingRoutes from './practiceGrading.routes';
import mentoringRoutes from './mentoring.routes';
import paperParserRoutes from './paperParser.routes';
import classesRoutes from './classes.routes';
import assignmentsRoutes from './assignments.routes';
import customQuestionsRoutes from './customQuestions.routes';
import submissionsRoutes from './submissions.routes';
import reviewRoutes from './review.routes';
import releaseRoutes from './release.routes';
import feedbackRoutes from './feedback.routes';
import teacherInsightsRoutes from './teacherInsights.routes';
import settingsRoutes from './settings.routes';
import institutionRoutes from './institution.routes';
import datesheetRoutes from './datesheet.routes';
import { clerkAuth } from './lib/clerkAuth';
import { rateLimit } from './lib/rateLimit';
import { logConfigReport, serviceReadinessMap } from './lib/configReport';

// per-user throttle for the paid-AI routes (generous — only stops abuse)
const aiLimit = rateLimit({ windowMs: 60_000, max: 40, name: 'ai' });

const app: Express = express();
const PORT = process.env.PORT || 3001;

const oaServiceUrl = (
  process.env.OA_GRADING_SERVICE_URL ||
  process.env.QA_GRADING_SERVICE_URL ||
  'http://127.0.0.1:8001'
).trim();
const autoStartOaServiceSetting = (
  process.env.AUTO_START_OA_GRADING_SERVICE ||
  process.env.AUTO_START_QA_GRADING_SERVICE ||
  ''
).trim().toLowerCase();
const shouldAutoStartOaService = autoStartOaServiceSetting !== 'false';

function isLocalOaServiceUrl(): boolean {
  try {
    const normalized = oaServiceUrl.endsWith('/') ? oaServiceUrl.slice(0, -1) : oaServiceUrl;
    const parsed = new URL(normalized);
    return parsed.hostname === '127.0.0.1' || parsed.hostname === 'localhost';
  } catch {
    return oaServiceUrl.includes('127.0.0.1') || oaServiceUrl.includes('localhost');
  }
}

async function isOaServiceReachable(): Promise<boolean> {
  try {
    const base = oaServiceUrl.endsWith('/') ? oaServiceUrl.slice(0, -1) : oaServiceUrl;
    const response = await fetch(`${base}/oa-level/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureOaServiceSidecar(): Promise<void> {
  if (!shouldAutoStartOaService) {
    console.log('OA grading sidecar auto-start is disabled by AUTO_START_OA_GRADING_SERVICE=false.');
    return;
  }

  if (!isLocalOaServiceUrl()) {
    console.log(`OA grading service is configured as remote (${oaServiceUrl}); skipping local sidecar auto-start.`);
    return;
  }

  const reachable = await isOaServiceReachable();
  if (reachable) {
    console.log(`OA grading sidecar already running at ${oaServiceUrl}`);
    return;
  }

  try {
    const logDir = path.join(process.cwd(), 'logs');
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
    const sidecarLogPath = path.join(logDir, 'oa-grading-sidecar.log');
    const outFd = fs.openSync(sidecarLogPath, 'a');
    const launcherScriptPath = path.join(process.cwd(), 'scripts', 'start-oa-grading-service.js');

    if (!fs.existsSync(launcherScriptPath)) {
      console.log(`OA grading launcher script not found at ${launcherScriptPath}; skipping auto-start.`);
      return;
    }

    let sidecarPort = '8001';
    try {
      const normalized = oaServiceUrl.endsWith('/') ? oaServiceUrl.slice(0, -1) : oaServiceUrl;
      const parsed = new URL(normalized);
      if (parsed.port) {
        sidecarPort = parsed.port;
      }
    } catch {
      // Keep default port if OA_GRADING_SERVICE_URL is malformed.
    }

    const child = spawn(process.execPath, [launcherScriptPath], {
      cwd: process.cwd(),
      detached: true,
      stdio: ['ignore', outFd, outFd],
      windowsHide: true,
      env: {
        ...process.env,
        OA_GRADING_SERVICE_PORT: sidecarPort,
        QA_GRADING_SERVICE_PORT: sidecarPort,
      },
    });

    child.on('error', (error) => {
      console.warn('OA grading sidecar launcher failed:', error);
    });

    child.unref();
    console.log(`Starting OA grading sidecar from launcher: ${launcherScriptPath}`);
    console.log(`Sidecar logs: ${sidecarLogPath}`);
  } catch (error) {
    console.warn('Failed to auto-start OA grading sidecar:', error);
  }
}

// CORS configuration - allow both local development and production
function normalizeOrigin(origin: string): string {
  return origin
    .trim()
    .replace(/^['"]+|['"]+$/g, '')
    .replace(/\/+$/, '');
}

const configuredOrigins = (process.env.FRONTEND_URL || '')
  .split(',')
  .map((origin) => normalizeOrigin(origin))
  .filter(Boolean);

const allowedOrigins = new Set(
  // Production frontend is allow-listed explicitly (belt-and-suspenders) in
  // addition to FRONTEND_URL, so a misconfigured env can never lock out the
  // real site. Add other production origins to FRONTEND_URL (comma-separated).
  ['http://localhost:3000', 'https://oalevels.vercel.app', ...configuredOrigins].map((origin) => normalizeOrigin(origin))
);

function isAllowedOrigin(origin: string): boolean {
  const normalizedOrigin = normalizeOrigin(origin);
  if (allowedOrigins.has(normalizedOrigin)) {
    return true;
  }

  try {
    const parsed = new URL(normalizedOrigin);
    const host = parsed.hostname.toLowerCase();
    const port = Number(parsed.port);

    if (
      process.env.NODE_ENV !== 'production' &&
      (host === 'localhost' || host === '127.0.0.1') &&
      port >= 3000 &&
      port <= 3010
    ) {
      return true;
    }

    // F-03: only THIS project's Vercel preview deployments are allowed —
    // production is allow-listed explicitly above / via FRONTEND_URL. The broad
    // "*.vercel.app" rule was removed so an unrelated app anyone deploys to
    // Vercel can no longer make credentialed calls to this API.
    if (host.endsWith('.sova-vis-projects.vercel.app')) {
      return true;
    }
  } catch {
    // Ignore malformed origins and reject below.
  }

  return false;
}

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || isAllowedOrigin(origin)) {
      callback(null, true);
      return;
    }

    callback(new Error(`CORS blocked for origin: ${origin}`));
  },
  credentials: true,
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Auth API
app.use('/auth', authRoutes);

// Admin API
app.use('/admin', adminRoutes);

// Papers API
app.use('/papers', papersRoutes);

// Content API (navigation/search) — now requires auth (F-04)
app.use('/content', clerkAuth, contentRoutes);

// RAG / Ask-AI — auth + AI rate limit (F-04)
app.use('/rag', clerkAuth, aiLimit, ragRoutes);

// User paper tracking API
app.use('/tracking', paperTrackingRoutes);

// Practice-paper progress API (autosave, timers, handwritten uploads)
app.use('/practice', practiceProgressRoutes);

// AI marking for practice papers (Grok text + vision) — auth in-module + AI limit
app.use('/practice-grading', clerkAuth, aiLimit, practiceGradingRoutes);

// Standalone upload-and-mark flow (Grok vision + annotated PDF)
app.use('/upload-check', clerkAuth, aiLimit, uploadCheckRoutes);

// Phase 1 — performance insights (attempts log for Notebook / Weakness Map)
app.use('/insights', clerkAuth, aiLimit, insightsRoutes);

// Teacher-student meetings and chat API
app.use('/mentoring', mentoringRoutes);

// Teacher Portal — class & enrolment management (auth enforced in-module)
app.use('/classes', classesRoutes);

// Teacher Portal — assignment creation & tracking (auth enforced in-module)
app.use('/assignments', assignmentsRoutes);

// Teacher Portal — custom questions with discrete criteria (§5.4)
app.use('/custom-questions', customQuestionsRoutes);

// Teacher Portal — submissions & marking (feature groups 7, 8)
app.use('/submissions', submissionsRoutes);

// Teacher Portal — grading review (feature group 9)
app.use('/review', reviewRoutes);

// Teacher Portal — result release (§11) and feedback (§10)
app.use('/release', releaseRoutes);
app.use('/feedback', feedbackRoutes);

// Teacher Portal — dashboard (§2) and insights (§12)
app.use('/teacher-insights', teacherInsightsRoutes);

// Teacher Portal — settings (§17) + notifications (§16), institution (§14–18)
app.use('/settings', settingsRoutes);
app.use('/institution', institutionRoutes);

// Propel — exam datesheet (student-facing)
app.use('/datesheet', datesheetRoutes);

// OA / QA grading proxy — now requires auth + AI rate limit (F-04)
app.use('/oa-grading', clerkAuth, aiLimit, qaGradingRoutes);
app.use('/qa-grading', clerkAuth, aiLimit, qaGradingRoutes);

// Past paper structuring — now requires auth + AI rate limit (F-04)
app.use('/paper-parser', clerkAuth, aiLimit, paperParserRoutes);

// Health check — includes subsystem readiness booleans (never secrets) so a
// misconfigured deploy is diagnosable without shell access.
app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString(), services: serviceReadinessMap() });
});

// Root route
app.get('/', (_req: Request, res: Response) => {
  res.send('Welcome to the Propel backend API!');
});

// Start server
app.listen(PORT, () => {
  console.log(`Backend server running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log('OA grading sidecar startup is handled by /qa-grading on-demand checks.');
  logConfigReport();
});

export default app;
