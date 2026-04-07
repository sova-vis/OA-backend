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
import mentoringRoutes from './mentoring.routes';

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
  ['http://localhost:3000', ...configuredOrigins].map((origin) => normalizeOrigin(origin))
);

function isAllowedOrigin(origin: string): boolean {
  const normalizedOrigin = normalizeOrigin(origin);
  if (allowedOrigins.has(normalizedOrigin)) {
    return true;
  }

  // Allow Vercel preview deployments for this project namespace.
  try {
    const parsed = new URL(normalizedOrigin);
    const host = parsed.hostname.toLowerCase();
    if (host.endsWith('.vercel.app')) {
      return true;
    }
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

// Content API (navigation/search)
app.use('/content', contentRoutes);

// RAG API
app.use('/rag', ragRoutes);

// User paper tracking API
app.use('/tracking', paperTrackingRoutes);

// Teacher-student meetings and chat API
app.use('/mentoring', mentoringRoutes);

// OA grading API
app.use('/oa-grading', qaGradingRoutes);
app.use('/qa-grading', qaGradingRoutes);

// Health check
app.get('/health', (_req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
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
});

export default app;
