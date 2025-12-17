#!/usr/bin/env python3
"""
PDF-CraftQ CLI - PDF to Markdown/EPUB converter using quantized DeepSeek OCR model

Usage similar to pandoc:
    pdf-craftq input.pdf -o output.md
    pdf-craftq input.pdf -o output.epub
"""

import argparse
import sys
from pathlib import Path

# Apply quantized model patch before importing pdf_craft
from quantized_model import apply_quantized_model_patch
apply_quantized_model_patch(quiet=True)


def get_output_format(output_path: Path, explicit_format: str | None) -> str:
    """Determine output format from file extension or explicit format flag."""
    if explicit_format:
        return explicit_format.lower()

    suffix = output_path.suffix.lower()
    if suffix in ('.md', '.markdown'):
        return 'markdown'
    elif suffix == '.epub':
        return 'epub'
    else:
        return 'markdown'  # default


def convert_to_markdown(
    pdf_path: Path,
    output_path: Path,
    assets_path: Path | None,
    ocr_size: str,
    local_only: bool,
    includes_footnotes: bool,
    ignore_pdf_errors: bool,
    verbose: bool,
) -> None:
    """Convert PDF to Markdown."""
    from pdf_craft import transform_markdown

    if verbose:
        print(f"Converting {pdf_path} to Markdown...")

    result = transform_markdown(
        pdf_path=str(pdf_path),
        markdown_path=str(output_path),
        markdown_assets_path=str(assets_path) if assets_path else None,
        ocr_size=ocr_size,
        local_only=local_only,
        includes_footnotes=includes_footnotes,
        ignore_pdf_errors=ignore_pdf_errors,
    )

    if verbose:
        print(f"Conversion complete!")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        print(f"  Output: {output_path}")
        if assets_path and assets_path.exists():
            print(f"  Assets: {assets_path}")


def convert_to_epub(
    pdf_path: Path,
    output_path: Path,
    ocr_size: str,
    local_only: bool,
    includes_cover: bool,
    includes_footnotes: bool,
    ignore_pdf_errors: bool,
    language: str,
    verbose: bool,
) -> None:
    """Convert PDF to EPUB."""
    from pdf_craft import transform_epub

    if verbose:
        print(f"Converting {pdf_path} to EPUB...")

    result = transform_epub(
        pdf_path=str(pdf_path),
        epub_path=str(output_path),
        ocr_size=ocr_size,
        local_only=local_only,
        includes_cover=includes_cover,
        includes_footnotes=includes_footnotes,
        ignore_pdf_errors=ignore_pdf_errors,
        lan=language,
    )

    if verbose:
        print(f"Conversion complete!")
        print(f"  Input tokens: {result.input_tokens}")
        print(f"  Output tokens: {result.output_tokens}")
        print(f"  Output: {output_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='pdf-craftq',
        description='Convert PDF to Markdown or EPUB using quantized DeepSeek OCR model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s input.pdf -o output.md          Convert PDF to Markdown
  %(prog)s input.pdf -o output.epub        Convert PDF to EPUB
  %(prog)s input.pdf -t markdown -o out    Explicit format specification
  %(prog)s input.pdf -o out.md --ocr-size base   Use base OCR model size
''',
    )

    # Positional argument: input file
    parser.add_argument(
        'input',
        type=Path,
        help='Input PDF file',
    )

    # Output file (required, like pandoc -o)
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output file path (.md for Markdown, .epub for EPUB)',
    )

    # Output format (optional, inferred from extension if not specified)
    parser.add_argument(
        '-t', '--to',
        choices=['markdown', 'md', 'epub'],
        help='Output format (default: inferred from output file extension)',
    )

    # OCR options
    parser.add_argument(
        '--ocr-size',
        choices=['tiny', 'small', 'base', 'large', 'gundam'],
        default='base',
        help='OCR model size (default: base)',
    )

    parser.add_argument(
        '--local-only',
        action='store_true',
        help='Use only locally cached models, do not download',
    )

    # Markdown-specific options
    parser.add_argument(
        '--assets-path',
        type=Path,
        help='Path for extracted images (Markdown only, default: <output>_assets)',
    )

    # EPUB-specific options
    parser.add_argument(
        '--no-cover',
        action='store_true',
        help='Do not include cover image (EPUB only)',
    )

    parser.add_argument(
        '--language', '-l',
        choices=['zh', 'en'],
        default='zh',
        help='Document language (EPUB only, default: zh)',
    )

    # Common options
    parser.add_argument(
        '--footnotes',
        action='store_true',
        help='Include footnotes in output',
    )

    parser.add_argument(
        '--ignore-pdf-errors',
        action='store_true',
        help='Continue processing even if PDF errors occur',
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose output',
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.1.0',
    )

    args = parser.parse_args(argv)

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    if not args.input.suffix.lower() == '.pdf':
        print(f"Warning: Input file does not have .pdf extension: {args.input}", file=sys.stderr)

    # Determine output format
    output_format = get_output_format(args.output, args.to)

    try:
        if output_format in ('markdown', 'md'):
            convert_to_markdown(
                pdf_path=args.input,
                output_path=args.output,
                assets_path=args.assets_path,
                ocr_size=args.ocr_size,
                local_only=args.local_only,
                includes_footnotes=args.footnotes,
                ignore_pdf_errors=args.ignore_pdf_errors,
                verbose=args.verbose,
            )
        elif output_format == 'epub':
            convert_to_epub(
                pdf_path=args.input,
                output_path=args.output,
                ocr_size=args.ocr_size,
                local_only=args.local_only,
                includes_cover=not args.no_cover,
                includes_footnotes=args.footnotes,
                ignore_pdf_errors=args.ignore_pdf_errors,
                language=args.language,
                verbose=args.verbose,
            )
        else:
            print(f"Error: Unsupported output format: {output_format}", file=sys.stderr)
            return 1

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
