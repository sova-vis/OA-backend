from __future__ import annotations

import mimetypes
from pathlib import Path

import fitz

from .config import Settings
from .types import DocumentInput, DocumentPage, InputDocumentError

SUPPORTED_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def load_document(
    input_path: str | Path,
    settings: Settings,
    *,
    page_number: int | None = None,
) -> DocumentInput:
    path = Path(input_path)
    if not path.exists():
        raise InputDocumentError("Input file does not exist.", path=str(path))
    if not path.is_file():
        raise InputDocumentError("Input path must be a file.", path=str(path))

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, settings, page_number=page_number)
    if page_number is not None and page_number != 1:
        raise InputDocumentError(
            "page_number is only supported for PDF inputs; use 1 or omit for images.",
            path=str(path),
        )
    return _load_image(path, settings)


def _load_image(path: Path, settings: Settings) -> DocumentInput:
    mime_type = SUPPORTED_IMAGE_MIME_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(str(path))[0]
    if mime_type not in {"image/png", "image/jpeg"}:
        raise InputDocumentError(
            "Unsupported image type. Supported formats are PNG, JPG, and JPEG.",
            path=str(path),
        )

    content = path.read_bytes()
    if len(content) > settings.max_image_size_bytes:
        raise InputDocumentError(
            "Input image exceeds the 20 MiB xAI image size limit.",
            path=str(path),
        )

    page = DocumentPage(
        page_number=1,
        mime_type=mime_type,
        content_bytes=content,
        source_name=path.name,
    )
    return DocumentInput(input_type="image", source_path=path, pages=(page,))


def _load_pdf(path: Path, settings: Settings, *, page_number: int | None = None) -> DocumentInput:
    document = fitz.open(path)
    try:
        if document.page_count == 0:
            raise InputDocumentError("PDF has no pages.", path=str(path))

        if page_number is not None:
            if page_number < 1 or page_number > document.page_count:
                raise InputDocumentError(
                    f"PDF page_number must be between 1 and {document.page_count} (got {page_number}).",
                    path=str(path),
                )
            page_indices = [page_number - 1]
        else:
            page_indices = list(range(document.page_count))

        pages: list[DocumentPage] = []
        for index in page_indices:
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=300, alpha=False)
            image_bytes = pixmap.tobytes("png")
            mime_type = "image/png"

            if len(image_bytes) > settings.max_image_size_bytes:
                image_bytes = pixmap.tobytes("jpg")
                mime_type = "image/jpeg"

            if len(image_bytes) > settings.max_image_size_bytes:
                raise InputDocumentError(
                    "Rendered PDF page exceeds the 20 MiB xAI image size limit.",
                    path=str(path),
                )

            pages.append(
                DocumentPage(
                    page_number=index + 1,
                    mime_type=mime_type,
                    content_bytes=image_bytes,
                    source_name=f"{path.stem}_page_{index + 1}",
                )
            )

        return DocumentInput(input_type="pdf", source_path=path, pages=tuple(pages))
    finally:
        document.close()

