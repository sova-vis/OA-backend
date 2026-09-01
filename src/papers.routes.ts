import { Router } from 'express';
import { clerkAuth } from './lib/clerkAuth';
import {
  listFilesInFolder,
  getFileMetadata,
  getEmbedViewerUrl,
  getDirectDownloadUrl,
  searchFilesByName,
  findFilesByNameGlobal,
  isDescendantOf,
  DriveFile,
  getFileStream,
  getFileBuffer,
  listFoldersAndFiles,
  getAllPDFsRecursive
} from './lib/googleDrive';
import { env } from './lib/env';

const router = Router();

// mupdf ships as ESM with top-level await; keep a real dynamic import in the CJS
// build (same trick uploadCheck.routes uses) and cache the module.
const importEsm = new Function('m', 'return import(m)') as (m: string) => Promise<any>;
let mupdfPromise: Promise<any> | null = null;
const loadMupdf = () => (mupdfPromise ??= importEsm('mupdf'));

// Per-file page text, so opening several questions of the same paper parses the
// PDF once. Small LRU-ish cap to bound memory.
const pageTextCache = new Map<string, string[]>();
const PAGE_CACHE_MAX = 40;

function normalizeForMatch(s: string): string {
  return (s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

async function pageTextsFor(fileId: string): Promise<string[]> {
  const hit = pageTextCache.get(fileId);
  if (hit) return hit;
  const buffer = await getFileBuffer(fileId);
  const mupdf = await loadMupdf();
  const doc = mupdf.Document.openDocument(new Uint8Array(buffer), 'application/pdf');
  const texts: string[] = [];
  const count = doc.countPages();
  for (let i = 0; i < count; i++) {
    const page = doc.loadPage(i);
    let txt = '';
    try {
      txt = page.toStructuredText('preserve-whitespace').asText();
    } catch {
      txt = '';
    }
    texts.push(normalizeForMatch(txt));
  }
  if (pageTextCache.size >= PAGE_CACHE_MAX) {
    const oldest = pageTextCache.keys().next().value;
    if (oldest !== undefined) pageTextCache.delete(oldest);
  }
  pageTextCache.set(fileId, texts);
  return texts;
}

/**
 * 1-based page number whose text contains a distinctive chunk of the question,
 * or null when it cannot be located (caller then opens the paper at page 1).
 */
async function findQuestionPage(fileId: string, questionText: string): Promise<number | null> {
  const needle = normalizeForMatch(questionText);
  if (needle.length < 12) return null;              // too short to be distinctive
  // a chunk from a little into the stem avoids leading "1"/"2" question numbers
  const probe = needle.slice(0, 48);
  try {
    const pages = await pageTextsFor(fileId);
    let idx = pages.findIndex((t) => t.includes(probe));
    if (idx < 0) idx = pages.findIndex((t) => t.includes(needle.slice(0, 24)));
    return idx >= 0 ? idx + 1 : null;
  } catch (e) {
    console.error('page lookup failed:', e);
    return null;
  }
}

type PaperLevel = 'olevel' | 'alevel';

function resolveDriveFolderId(explicitFolderId?: string, level?: PaperLevel): string | undefined {
  const fromParam = explicitFolderId?.trim();
  if (fromParam) return fromParam;

  // A-Level library lives in its own Drive root
  if (level === 'alevel') return env.GOOGLE_DRIVE_ALEVEL_FOLDER_ID?.trim();

  const fromPrimaryEnv = env.GOOGLE_DRIVE_FOLDER_ID?.trim();
  const fromLegacyEnv = env.GOOGLE_DRIVE_ROOT_FOLDER_ID?.trim();

  return fromPrimaryEnv || fromLegacyEnv;
}

function missingFolderConfigError(level?: PaperLevel) {
  const varName =
    level === 'alevel' ? 'GOOGLE_DRIVE_ALEVEL_FOLDER_ID' : 'GOOGLE_DRIVE_FOLDER_ID';
  return {
    error: `Google Drive folder ID is not configured. Set ${varName} in OA-backend/.env${
      level === 'alevel' ? '' : ' (or GOOGLE_DRIVE_ROOT_FOLDER_ID for legacy setups)'
    }.`,
  };
}

function getDriveErrorMessage(error: unknown): { status: number; message: string } {
  const driveError = error as {
    code?: number;
    response?: { status?: number; data?: { error?: { message?: string } } };
    message?: string;
  };

  const status = driveError?.response?.status ?? driveError?.code;
  const providerMessage =
    driveError?.response?.data?.error?.message ?? driveError?.message;

  if (status === 401 || status === 403) {
    return {
      status: 502,
      message:
        'Google Drive authorization failed. Verify GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN, and Drive file permissions.',
    };
  }

  if (status === 404) {
    return {
      status: 404,
      message:
        'Configured Google Drive folder was not found or is not accessible. Verify GOOGLE_DRIVE_FOLDER_ID and sharing permissions.',
    };
  }

  return {
    status: 500,
    message: providerMessage || 'Failed to browse folder',
  };
}

/**
 * GET /papers/drive/list
 * List all papers from Google Drive folder (recursively gets all PDFs)
 */
router.get('/drive/list', clerkAuth, async (req, res) => {
  try {
    const folderId = resolveDriveFolderId();

    if (!folderId) {
      return res.status(500).json(missingFolderConfigError());
    }

    console.log('📂 Getting all PDFs recursively...');
    const allPDFs = await getAllPDFsRecursive(folderId);

    console.log(`📄 Found ${allPDFs.length} PDF files`);

    // Transform to include URLs
    const filesWithUrls = allPDFs.map((file) => ({
      id: file.id,
      name: file.name,
      path: file.path,
      fullPath: file.fullPath,
      size: file.size,
      modifiedTime: file.modifiedTime,
      viewUrl: `/papers/view/${file.id}`,
      downloadUrl: `/papers/download/${file.id}`,
      embedUrl: `/papers/view/${file.id}`,
      mimeType: 'application/pdf',
    }));

    res.json({ files: filesWithUrls });
  } catch (error) {
    console.error('Error listing papers:', error);
    res.status(500).json({ error: 'Failed to fetch papers from Google Drive' });
  }
});

/**
 * GET /papers/browse/:folderId
 * Browse a specific folder (for navigation) - returns immediate children only
 */
router.get('/browse/:folderId?', clerkAuth, async (req, res) => {
  try {
    // ?level=alevel switches the root to the A-Level library (only relevant
    // when no explicit folderId is given — subfolder ids are globally unique)
    const level: PaperLevel = req.query.level === 'alevel' ? 'alevel' : 'olevel';
    const folderId = resolveDriveFolderId(req.params.folderId, level);

    if (!folderId) {
      return res.status(500).json(missingFolderConfigError(level));
    }

    console.log(`📂 Browsing folder: ${folderId}`);
    const items = await listFoldersAndFiles(folderId);

    // Helper function to detect folder type based on name
    const detectFolderType = (name: string): 'subject' | 'category' | 'year' | 'month' | 'unknown' => {
      // Check if it's a year (4 digits)
      if (/^\d{4}$/.test(name)) return 'year';

      // Check if it's a month/session (contains month names or session patterns)
      const monthPatterns = /may|june|march|october|november|feb|jan|apr|jul|aug|sep|dec|m\/j|o\/n|f\/m/i;
      if (monthPatterns.test(name)) return 'month';

      // Check if it's a category folder
      const categoryPatterns = /past\s*papers?|syllabus|reference|topical|notes/i;
      if (categoryPatterns.test(name)) return 'category';

      // Otherwise assume it's a subject
      return 'subject';
    };

    // Transform items to include URLs for files and type detection for folders
    const transformedItems = items.map(item => {
      if (item.isFolder) {
        const folderType = detectFolderType(item.name);
        return {
          id: item.id,
          name: item.name,
          isFolder: true,
          folderType,
          mimeType: item.mimeType,
        };
      } else {
        return {
          id: item.id,
          name: item.name,
          isFolder: false,
          mimeType: item.mimeType,
          size: item.size,
          modifiedTime: item.modifiedTime,
          viewUrl: `/papers/view/${item.id}`,
          downloadUrl: `/papers/download/${item.id}`,
          embedUrl: `/papers/view/${item.id}`,
        };
      }
    });

    // Sort items: folders first (subjects, then categories, then years, then months), then files
    const sortOrder: Record<string, number> = { subject: 0, category: 1, year: 2, month: 3, unknown: 4 };
    transformedItems.sort((a, b) => {
      if (a.isFolder && !b.isFolder) return -1;
      if (!a.isFolder && b.isFolder) return 1;
      if (a.isFolder && b.isFolder) {
        const aType = (a as any).folderType || 'unknown';
        const bType = (b as any).folderType || 'unknown';
        const aOrder = sortOrder[aType] || 5;
        const bOrder = sortOrder[bType] || 5;
        if (aOrder !== bOrder) return aOrder - bOrder;
      }
      return a.name.localeCompare(b.name);
    });

    console.log(`✅ Found ${transformedItems.length} items (${transformedItems.filter(i => i.isFolder).length} folders, ${transformedItems.filter(i => !i.isFolder).length} files)`);

    res.json({
      folderId,
      items: transformedItems
    });
  } catch (error) {
    console.error('Error browsing folder:', error);
    const mappedError = getDriveErrorMessage(error);
    res.status(mappedError.status).json({ error: mappedError.message });
  }
});

/**
 * GET /papers/view/:fileId
 * Stream file for viewing (embedded PDF viewer)
 */
router.get('/view/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;

    // Get file metadata first
    const metadata = await getFileMetadata(fileId);

    // Set appropriate headers
    res.setHeader('Content-Type', metadata.mimeType);
    res.setHeader('Content-Disposition', `inline; filename="${metadata.name}"`);

    // Stream the file
    const stream = await getFileStream(fileId);
    stream.pipe(res);
  } catch (error) {
    console.error('Error viewing file:', error);
    res.status(500).json({ error: 'Failed to view file' });
  }
});

/**
 * GET /papers/download/:fileId
 * Stream file for download
 */
router.get('/download/:fileId', async (req, res) => {
  try {
    const { fileId } = req.params;

    // Get file metadata first
    const metadata = await getFileMetadata(fileId);

    // Set headers for download
    res.setHeader('Content-Type', metadata.mimeType);
    res.setHeader('Content-Disposition', `attachment; filename="${metadata.name}"`);

    // Stream the file
    const stream = await getFileStream(fileId);
    stream.pipe(res);
  } catch (error) {
    console.error('Error downloading file:', error);
    res.status(500).json({ error: 'Failed to download file' });
  }
});

/**
 * GET /papers/drive/search?q=searchTerm
 * Search papers by name
 */
router.get('/drive/search', clerkAuth, async (req, res) => {
  try {
    const { q } = req.query;
    const folderId = resolveDriveFolderId();

    if (!folderId) {
      return res.status(500).json(missingFolderConfigError());
    }

    if (!q || typeof q !== 'string') {
      return res.status(400).json({ error: 'Search term is required' });
    }

    const files = await searchFilesByName(folderId, q);

    const filesWithUrls = files.map((file: DriveFile) => ({
      id: file.id,
      name: file.name,
      size: file.size,
      modifiedTime: file.modifiedTime,
      viewUrl: `/papers/view/${file.id}`,
      downloadUrl: `/papers/download/${file.id}`,
      embedUrl: `/papers/view/${file.id}`,
      mimeType: file.mimeType,
    }));

    res.json({ files: filesWithUrls });
  } catch (error) {
    console.error('Error searching papers:', error);
    res.status(500).json({ error: 'Failed to search papers' });
  }
});

/**
 * GET /papers/find-qp
 * Resolve a question's original past-paper PDF (or mark scheme) to a Drive file,
 * so the practice UI can open it when an extracted image is not good enough.
 *
 * Query: subject, year, session, paper, variant  — OR  name=<QP filename/stem>.
 * The paper's filename encodes all of these ("Biology_2023_May_June_Paper_2_
 * Variant_1_QP.pdf"), so we build the distinctive stem and let Drive substring-
 * match it, then prefer an exact-stem, question-paper match. kind=ms opens the
 * mark scheme instead.
 */
router.get('/find-qp', clerkAuth, async (req, res) => {
  try {
    const q = req.query as Record<string, string | undefined>;
    const kind = (q.kind || 'qp').toLowerCase() === 'ms' ? 'ms' : 'qp';

    // Build the filename stem (without _QP/_MS and extension) to search on.
    let stem = '';
    if (q.name) {
      stem = q.name.split(/[\\/]/).pop()!.replace(/\.pdf$/i, '').replace(/_(QP|MS)$/i, '');
    } else if (q.subject && q.year && q.session && q.paper) {
      // A-Level questions carry a "(A Level)" tag on the subject to keep them
      // separate from O-Level; the PDF filenames in Drive don't, so strip it.
      const cleanSubject = q.subject.trim().replace(/\s*\(A Level\)\s*$/i, '');
      const parts = [
        cleanSubject.replace(/\s+/g, '_'),
        q.year, q.session, q.paper,
        q.variant && q.variant.trim() ? q.variant.trim() : '',
      ].filter(Boolean);
      stem = parts.join('_');
    } else {
      return res.status(400).json({ error: 'provide name, or subject+year+session+paper' });
    }

    let files = await findFilesByNameGlobal(stem, { mimeType: 'application/pdf' });
    // no-variant papers store the variant as part of the folder but not always the
    // filename; if the exact stem found nothing, retry without the trailing variant
    if (files.length === 0 && /_Variant_\d+$/i.test(stem)) {
      files = await findFilesByNameGlobal(stem.replace(/_Variant_\d+$/i, ''), { mimeType: 'application/pdf' });
    }

    // O-Level and A-Level papers share identical filenames for common subjects
    // (Physics, Chemistry, Biology, Economics, Maths, …) and differ only by which
    // library root they live under. When more than one match comes back, keep only
    // the ones under THIS level's Drive root so an O-Level question never opens the
    // A-Level paper (and vice-versa).
    const level: PaperLevel = q.level === 'alevel' ? 'alevel' : 'olevel';
    const levelRootId = resolveDriveFolderId(undefined, level);
    if (levelRootId && files.length > 1) {
      const scoped: DriveFile[] = [];
      for (const f of files) {
        if (await isDescendantOf(f.id, levelRootId)) scoped.push(f);
      }
      if (scoped.length) files = scoped;
    }

    const wantKind = kind === 'ms' ? /_MS\b/i : /_QP\b/i;
    const stemLc = stem.toLowerCase();
    const exact = files.find((f) => f.name.replace(/\.pdf$/i, '').toLowerCase() === `${stemLc}_${kind}`);
    const pick =
      exact ||
      files.find((f) => wantKind.test(f.name) && f.name.toLowerCase().includes(stemLc)) ||
      files.find((f) => wantKind.test(f.name)) ||
      files[0];

    if (!pick) return res.status(404).json({ error: 'paper not found', stem });

    // Optional: jump to the page the question is on. No page number is stored in
    // the data, so it is located by matching the question text against each page's
    // text. Best-effort — a miss just opens the paper at page 1.
    let page: number | null = null;
    if (kind === 'qp' && typeof q.text === 'string' && q.text.trim()) {
      page = await findQuestionPage(pick.id, q.text);
    }

    res.json({
      fileId: pick.id,
      name: pick.name,
      viewUrl: `/papers/view/${pick.id}`,
      page,
    });
  } catch (error) {
    console.error('Error resolving paper:', error);
    res.status(500).json({ error: 'Failed to resolve paper' });
  }
});

/**
 * GET /papers/drive/:fileId
 * Get metadata for a specific file
 */
router.get('/drive/:fileId', clerkAuth, async (req, res) => {
  try {
    const { fileId } = req.params;

    const file = await getFileMetadata(fileId);

    res.json({
      id: file.id,
      name: file.name,
      size: file.size,
      modifiedTime: file.modifiedTime,
      viewUrl: `/papers/view/${file.id}`,
      downloadUrl: `/papers/download/${file.id}`,
      embedUrl: `/papers/view/${file.id}`,
      mimeType: file.mimeType,
    });
  } catch (error) {
    console.error('Error getting file metadata:', error);
    res.status(500).json({ error: 'Failed to get file metadata' });
  }
});

/**
 * GET /papers/organized
 * Get papers organized by subject, year, session
 */
router.get('/organized', clerkAuth, async (req, res) => {
  try {
    const folderId = resolveDriveFolderId();

    if (!folderId) {
      return res.status(500).json(missingFolderConfigError());
    }

    const files = await listFilesInFolder(folderId);

    // Organize files by parsing filenames
    // Expected format: "SubjectCode_Year_Session_Paper_Variant.pdf"
    // Example: "0580_2023_MJ_P1_12.pdf" or "Physics_2022_ON_P4_41.pdf"

    const organized: Record<string, any> = {};

    files.forEach((file: DriveFile) => {
      const name = file.name;

      // Try to parse the filename
      // Pattern: Subject_Year_Session_Paper_Variant
      const parts = name.replace('.pdf', '').split('_');

      if (parts.length >= 5) {
        const [subject, year, session, paper, variant] = parts;

        if (!organized[subject]) {
          organized[subject] = {};
        }
        if (!organized[subject][year]) {
          organized[subject][year] = {};
        }
        if (!organized[subject][year][session]) {
          organized[subject][year][session] = [];
        }

        organized[subject][year][session].push({
          id: file.id,
          name: file.name,
          paper,
          variant,
          viewUrl: `/papers/view/${file.id}`,
          downloadUrl: `/papers/download/${file.id}`,
          embedUrl: `/papers/view/${file.id}`,
          size: file.size,
        });
      }
    });

    res.json({ papers: organized });
  } catch (error) {
    console.error('Error organizing papers:', error);
    res.status(500).json({ error: 'Failed to organize papers' });
  }
});

export default router;
