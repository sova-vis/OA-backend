/**
 * PDF -> page image rasterization, shared by every flow that feeds scanned work
 * to a vision model (upload-check, practice handwritten grading).
 *
 * mupdf ships as ESM with top-level await, so it is imported through a runtime
 * `import()` that survives the CommonJS TypeScript build.
 */

const importEsm = new Function('m', 'return import(m)') as (m: string) => Promise<any>;
let mupdfPromise: Promise<any> | null = null;

export function loadMupdf(): Promise<any> {
  if (!mupdfPromise) mupdfPromise = importEsm('mupdf');
  return mupdfPromise;
}

export interface PageImage {
  /** base64, no data: prefix */
  base64: string;
  mimeType: string;
  /** 1-based position in the combined document */
  page: number;
  /** originating file name, for error messages */
  source?: string;
}

/** Rendering at 2x keeps handwriting legible to the vision model. */
const RASTER_SCALE = 2;

/**
 * Rasterize a PDF buffer to page PNGs.
 *
 * @param startPage 1-based page number to assign to the first rendered page
 * @param budget    maximum pages to render from this document
 * @returns the rendered pages plus how many pages were left unrendered
 */
export async function rasterizePdfPages(
  buffer: Buffer,
  startPage: number,
  budget: number,
  source?: string,
): Promise<{ pages: PageImage[]; totalPages: number; dropped: number }> {
  if (budget <= 0) return { pages: [], totalPages: 0, dropped: 0 };

  const mupdf = await loadMupdf();
  let doc: any;
  try {
    doc = mupdf.Document.openDocument(new Uint8Array(buffer), 'application/pdf');
  } catch (error) {
    throw new PdfReadError(source, error);
  }

  let totalPages: number;
  try {
    totalPages = doc.countPages();
  } catch (error) {
    throw new PdfReadError(source, error);
  }
  if (!Number.isFinite(totalPages) || totalPages <= 0) throw new PdfReadError(source);

  const count = Math.min(totalPages, budget);
  const scale = mupdf.Matrix.scale(RASTER_SCALE, RASTER_SCALE);
  const pages: PageImage[] = [];
  for (let i = 0; i < count; i++) {
    const page = doc.loadPage(i);
    const pixmap = page.toPixmap(scale, mupdf.ColorSpace.DeviceRGB, false, true);
    pages.push({
      base64: Buffer.from(pixmap.asPNG()).toString('base64'),
      mimeType: 'image/png',
      page: startPage + i,
      source,
    });
  }
  return { pages, totalPages, dropped: Math.max(0, totalPages - count) };
}

/** A PDF that mupdf could not open — corrupt, encrypted, or not a PDF at all. */
export class PdfReadError extends Error {
  constructor(source?: string, cause?: unknown) {
    super(
      `Could not read ${source ? `"${source}"` : 'the PDF'}. The file may be corrupted, password-protected, or not a real PDF.`,
    );
    this.name = 'PdfReadError';
    if (cause !== undefined) (this as { cause?: unknown }).cause = cause;
  }
}

/** Image types Grok vision accepts. */
export function isSupportedImageType(mimeType: string): boolean {
  return /^image\/(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(mimeType || '');
}

const PRACTICE_UPLOAD_EXT = new Set(['jpg', 'jpeg', 'png', 'pdf']);

function extensionOf(name: string): string {
  const base = (name || '').split(/[/\\]/).pop() || '';
  const dot = base.lastIndexOf('.');
  return dot >= 0 ? base.slice(dot + 1).toLowerCase() : '';
}

/**
 * Files the Full Paper handwritten flow accepts: JPG, PNG, or PDF.
 * Falls back to the filename when the browser/cloud-drive sends an empty MIME.
 */
export function isPracticeUploadType(mimeType: string, fileName?: string): boolean {
  const mime = (mimeType || '').toLowerCase();
  if (mime === 'application/pdf' || mime === 'image/jpeg' || mime === 'image/jpg' || mime === 'image/png') {
    return true;
  }
  if (mime && mime !== 'application/octet-stream') return false;
  return PRACTICE_UPLOAD_EXT.has(extensionOf(fileName || ''));
}
