from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pypdfium2 as pdfium


RAW_DIR = Path("data/raw")
PAGES_DIR = Path("pages")
DEFAULT_DPI = 200


def render_pdf(pdf_path: Path, output_root: Path, dpi: int, force: bool) -> None:
    output_dir = output_root / pdf_path.stem

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72

    print(f"PDF -> PNG : {pdf_path} -> {output_dir}")

    for index, page in enumerate(document, start=1):
        output_path = output_dir / f"page-{index:03d}.png"

        if output_path.exists() and not force:
            print(f"Skip existing : {output_path}")
            continue

        image = page.render(scale=scale).to_pil()
        image.save(output_path)
        print(f"OK : {output_path}")

    document.close()


def list_pdf_files(input_dir: Path, only: str | None) -> list[Path]:
    pdf_files = sorted(input_dir.rglob("*.pdf"))

    if only:
        pdf_files = [
            pdf_path
            for pdf_path in pdf_files
            if pdf_path.stem == only or pdf_path.name == only
        ]

    return pdf_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render input PDFs from data/raw into pages/<pdf-name>/page-*.png."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory containing input PDFs. Default: {RAW_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAGES_DIR,
        help=f"Directory where rendered pages are written. Default: {PAGES_DIR}",
    )
    parser.add_argument(
        "--only",
        help="Render only one PDF, matched by filename or stem.",
        default=None,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG resolution. Default: {DEFAULT_DPI}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate existing page directories before rendering.",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {args.input_dir}")

    pdf_files = list_pdf_files(args.input_dir, args.only)

    if not pdf_files:
        print(f"Aucun PDF trouvé dans {args.input_dir}")
        return

    for pdf_path in pdf_files:
        render_pdf(pdf_path, args.output_dir, args.dpi, args.force)


if __name__ == "__main__":
    main()
