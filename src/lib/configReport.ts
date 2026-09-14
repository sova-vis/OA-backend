/**
 * Boot-time configuration readiness report.
 *
 * After the F-04 lockdown, a missing key no longer fails loudly at boot — it
 * shows up later as a 401 (Clerk) or a 500 (an AI provider) that a student
 * hits. This module reports, at startup and via /health, which subsystems are
 * actually configured, so a bad deploy is obvious in one glance.
 *
 * It only ever reports booleans — never the secret values themselves — so the
 * /health payload is safe to expose.
 */

export type ServiceStatus = {
  key: string;
  label: string;
  ready: boolean;
  /** true = the whole site depends on this; false = only a feature degrades */
  critical: boolean;
  /** shown when not ready, to point at the fix */
  hint: string;
};

const has = (value: string | undefined) => Boolean(value && value.trim());

function driveReady(): boolean {
  return (
    has(process.env.GOOGLE_CLIENT_ID) &&
    has(process.env.GOOGLE_CLIENT_SECRET) &&
    has(process.env.GOOGLE_REFRESH_TOKEN) &&
    (has(process.env.GOOGLE_DRIVE_FOLDER_ID) || has(process.env.GOOGLE_DRIVE_ROOT_FOLDER_ID))
  );
}

export function serviceReadiness(): ServiceStatus[] {
  return [
    {
      key: 'supabase',
      label: 'Supabase (database + storage)',
      ready: has(process.env.SUPABASE_URL) && (has(process.env.SUPABASE_SERVICE_ROLE_KEY) || has(process.env.SUPABASE_KEY)),
      critical: true,
      hint: 'Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.',
    },
    {
      key: 'auth',
      label: 'Supabase Auth (verifies user tokens on every route)',
      ready: has(process.env.SUPABASE_JWT_SECRET),
      critical: true,
      hint: 'Set SUPABASE_JWT_SECRET (Supabase → Settings → API → JWT Secret) — without it every protected route returns 401.',
    },
    {
      key: 'grokMarking',
      label: 'Grok — AI marking (text + handwritten), Groq/Gemini fallback',
      ready:
        has(process.env.XAI_API_KEY) ||
        has(process.env.GROK_API_KEY) ||
        has(process.env.GROQ_GRADING_API_KEY) ||
        has(process.env.GROQ_API_KEY) ||
        has(process.env.GEMINI_API_KEY),
      critical: false,
      hint: 'Set XAI_API_KEY (primary). Grading falls back to GROQ_GRADING_API_KEY / GROQ_API_KEY / GEMINI_API_KEY.',
    },
    {
      key: 'groqPaperParsing',
      label: 'Groq — paper parsing (PDF → JSON)',
      ready: has(process.env.GROQ_API_KEY),
      critical: false,
      hint: 'Set GROQ_API_KEY — the paper parser will error without it.',
    },
    {
      key: 'askAiChatbot',
      label: 'Ask-AI RAG service (Past-Paper Chatbot)',
      ready: has(process.env.CHATBOT_SERVICE_URL),
      critical: false,
      hint: 'Set CHATBOT_SERVICE_URL to the deployed Past-Paper Chatbot (defaults to localhost for dev) — Ask/Find text answers 502 without it.',
    },
    {
      key: 'googleDrive',
      label: 'Google Drive — past-paper library',
      ready: driveReady(),
      critical: false,
      hint: 'Set GOOGLE_CLIENT_ID/SECRET, GOOGLE_REFRESH_TOKEN and a Drive folder id.',
    },
  ];
}

/** Compact booleans for the /health endpoint (no secrets). */
export function serviceReadinessMap(): Record<string, boolean> {
  return Object.fromEntries(serviceReadiness().map((s) => [s.key, s.ready]));
}

/** Print a ✅/⚠️ readiness table at boot. Never throws, never exits. */
export function logConfigReport(): void {
  const services = serviceReadiness();
  console.log('──────── configuration readiness ────────');
  for (const s of services) {
    const mark = s.ready ? '✅' : s.critical ? '❌' : '⚠️ ';
    console.log(`  ${mark} ${s.label}`);
    if (!s.ready) console.log(`       ↳ ${s.hint}`);
  }
  const missingCritical = services.filter((s) => s.critical && !s.ready);
  if (missingCritical.length > 0) {
    console.warn(
      `⚠️  ${missingCritical.length} critical service(s) not configured — the app will boot but core functionality is broken until fixed.`
    );
  }
  console.log('─────────────────────────────────────────');
}
