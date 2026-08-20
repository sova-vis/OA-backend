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

/**
 * Clerk JWT verification is what every protected route now depends on. It works
 * with either a valid SPKI public key (CLERK_JWT_KEY) or an issuer for JWKS
 * (CLERK_ISSUER) — mirror clerkAuth's own requirement so this stays accurate.
 */
function clerkVerifierReady(): boolean {
  const jwtKey = process.env.CLERK_JWT_KEY || '';
  const hasValidKey = jwtKey.includes('BEGIN PUBLIC KEY');
  return hasValidKey || has(process.env.CLERK_ISSUER);
}

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
      key: 'clerkAuth',
      label: 'Clerk auth (protects every route)',
      ready: clerkVerifierReady(),
      critical: true,
      hint: 'Set CLERK_JWT_KEY (PEM public key) or CLERK_ISSUER — without it every protected route returns 401.',
    },
    {
      key: 'grokMarking',
      label: 'Grok — AI marking (text + handwritten), Groq fallback',
      ready:
        has(process.env.XAI_API_KEY) ||
        has(process.env.GROK_API_KEY) ||
        has(process.env.GROQ_GRADING_API_KEY) ||
        has(process.env.GROQ_API_KEY),
      critical: false,
      hint: 'Set XAI_API_KEY (primary Grok). Grading falls back to GROQ_GRADING_API_KEY / GROQ_API_KEY.',
    },
    {
      key: 'groqAskAi',
      label: 'Groq — Ask-AI + paper parsing',
      ready: has(process.env.GROQ_API_KEY),
      critical: false,
      hint: 'Set GROQ_API_KEY — Ask-AI and paper parsing will error without it.',
    },
    {
      key: 'cohereEmbeddings',
      label: 'Cohere — Ask-AI retrieval embeddings',
      ready: has(process.env.COHERE_API_KEY),
      critical: false,
      hint: 'Set COHERE_API_KEY — Ask-AI retrieval quality degrades without it.',
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
