"""Load a P&ID drawing (image file or first page of a PDF) as PNG bytes for the LLM."""
from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
DEFAULT_MAX_SIDE = int(os.getenv("HAZOP_PID_MAX_SIDE", "2000"))
PDF_DPI = 200


def _pdf_first_page(path: Path) -> Image.Image:
    import pypdfium2 as pdfium   # permissive licence (Apache-2.0 / BSD-3-Clause)

    pdf = pdfium.PdfDocument(str(path))
    try:
        if len(pdf) == 0:
            raise ValueError(f"PDF has no pages: {path}")
        page = pdf[0]
        return page.render(scale=PDF_DPI / 72).to_pil()
    finally:
        pdf.close()


def load_pid(path: str | Path, max_side: int = DEFAULT_MAX_SIDE) -> tuple[bytes, str]:
    """Return (PNG bytes, 'image/png'). Metadata is dropped; the long side is capped at max_side."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"P&ID not found: {path}")
    ext = path.suffix.lower()
    if ext == ".pdf":
        img = _pdf_first_page(path)
    elif ext in IMAGE_EXTENSIONS:
        with Image.open(path) as im:
            im.seek(0)
            img = im.copy()
    else:
        raise ValueError(f"Unsupported P&ID file type '{ext}'. Use an image ({', '.join(sorted(IMAGE_EXTENSIONS))}) or a PDF.")

    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"
