from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageFilter, ImageOps

from .config import Settings
from .types import DocumentInput, DocumentPage, DocumentVariant, OCRVariant


def build_variants(document: DocumentInput, settings: Settings) -> tuple[DocumentVariant, ...]:
    original = DocumentVariant(name=OCRVariant.ORIGINAL.value, pages=document.pages)
    if not settings.enable_image_variants:
        return (original,)

    grayscale_pages = tuple(
        _transform_page(page, settings, OCRVariant.GRAYSCALE_AUTOCONTRAST.value, _grayscale_autocontrast)
        for page in document.pages
    )
    binary_pages = tuple(
        _transform_page(page, settings, OCRVariant.SHARPENED_BINARY.value, _sharpened_binary)
        for page in document.pages
    )
    return (
        original,
        DocumentVariant(name=OCRVariant.GRAYSCALE_AUTOCONTRAST.value, pages=grayscale_pages),
        DocumentVariant(name=OCRVariant.SHARPENED_BINARY.value, pages=binary_pages),
    )


def _transform_page(
    page: DocumentPage,
    settings: Settings,
    variant_name: str,
    transform,
) -> DocumentPage:
    with Image.open(BytesIO(page.content_bytes)) as image:
        processed = transform(image)
        content_bytes, mime_type = _serialize_image(processed, settings)

    return DocumentPage(
        page_number=page.page_number,
        mime_type=mime_type,
        content_bytes=content_bytes,
        source_name=page.source_name,
        variant_name=variant_name,
    )


def _grayscale_autocontrast(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    return ImageOps.autocontrast(grayscale)


def _sharpened_binary(image: Image.Image) -> Image.Image:
    grayscale = ImageOps.autocontrast(image.convert("L"))
    sharpened = grayscale.filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=2))
    threshold = sharpened.point(lambda pixel: 255 if pixel > 160 else 0, mode="1")
    return threshold.convert("L")


def _serialize_image(image: Image.Image, settings: Settings) -> tuple[bytes, str]:
    png_bytes = _save_image(image, "PNG")
    if len(png_bytes) <= settings.max_image_size_bytes:
        return png_bytes, "image/png"

    jpeg_bytes = _save_image(image.convert("RGB"), "JPEG")
    return jpeg_bytes, "image/jpeg"


def _save_image(image: Image.Image, fmt: str) -> bytes:
    buffer = BytesIO()
    save_kwargs = {"optimize": True}
    if fmt == "JPEG":
        save_kwargs["quality"] = 95
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()
