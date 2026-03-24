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

const app: Express = express();
const PORT = process.env.PORT || 3001;

const qaServiceUrl = (process.env.QA_GRADING_SERVICE_URL || 'http://127.0.0.1:8001').trim();
const autoStartQaServiceSetting = (process.env.AUTO_START_QA_GRADING_SERVICE || '').trim().toLowerCase();
const shouldAutoStartQaService = autoStartQaServiceSetting !== 'false';

async function isQaServiceReachable(): Promise<boolean> {
  try {
    const base = qaServiceUrl.endsWith('/') ? qaServiceUrl.slice(0, -1) : qaServiceUrl;
    const response = await fetch(`${base}/oa-level/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

async function ensureQaServiceSidecar(): Promise<void> {
  if (!shouldAutoStartQaService) {
    console.log('ℹ️ Q/A grading sidecar auto-start is disabled by AUTO_START_QA_GRADING_SERVICE=false.');
    return;
  }

  const reachable = await isQaServiceReachable();
  if (reachable) {
    console.log(`✅ Q/A grading sidecar already running at ${qaServiceUrl}`);
    return;
  }

  try {
    const logDir = path.join(process.cwd(), 'logs');
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }
    const sidecarLogPath = path.join(logDir, 'qa-sidecar.log');
    const outFd = fs.openSync(sidecarLogPath, 'a');
    const launcherScriptPath = path.join(process.cwd(), 'scripts', 'start-qa-grading-service.js');

    if (!fs.existsSync(launcherScriptPath)) {
      console.log(`ℹ️ Q/A grading launcher script not found at ${launcherScriptPath}; skipping auto-start.`);
      return;
    }

    let sidecarPort = '8001';
    try {
      const normalized = qaServiceUrl.endsWith('/') ? qaServiceUrl.slice(0, -1) : qaServiceUrl;
      const parsed = new URL(normalized);
      if (parsed.port) {
        sidecarPort = parsed.port;
      }
    } catch {
      // Keep default port if QA_GRADING_SERVICE_URL is malformed.
    }

    const child = spawn(process.execPath, [launcherScriptPath], {
      cwd: process.cwd(),
      detached: true,
      stdio: ['ignore', outFd, outFd],
      windowsHide: true,
      env: {
        ...process.env,
        QA_GRADING_SERVICE_PORT: sidecarPort,
      },
    });

    child.on('error', (error) => {
      console.warn('⚠️ Q/A grading sidecar launcher failed:', error);
    });

    child.unref();
    console.log(`🚀 Starting Q/A grading sidecar from launcher: ${launcherScriptPath}`);
    console.log(`📝 Sidecar logs: ${sidecarLogPath}`);
  } catch (error) {
    console.warn('⚠️ Failed to auto-start Q/A grading sidecar:', error);
  }
}

// CORS configuration - allow both local development and production
const configuredOrigins = (process.env.FRONTEND_URL || '')
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

const allowedOrigins = Array.from(new Set([
  'http://localhost:3000',
  ...configuredOrigins
]));

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
      return;
    }

    callback(new Error(`CORS blocked for origin: ${origin}`));
  },
  credentials: true
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

// Q/A grading API
app.use('/qa-grading', qaGradingRoutes);

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Root route
app.get('/', (req: Request, res: Response) => {
  res.send('Welcome to the Propel backend API!');
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);

  void ensureQaServiceSidecar();
});

export default app;
