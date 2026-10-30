"""Run the Markdown loader on every .md and .txt file in a configured folder.

Set MARKDOWN_TEST_DIRECTORY in config/settings.py or in the .env file, then run:
    python scripts/test_markdown_loader.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root so this script works when started from another folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from src.rag.markdown_loader import MarkdownLoader, process_markdown_file  # noqa: E402


def main() -> int:
    configured_path = settings.MARKDOWN_TEST_DIRECTORY.strip()
    if not configured_path:
        print(
            "Set MARKDOWN_TEST_DIRECTORY in config/settings.py or .env.",
            file=sys.stderr,
        )
        return 2

    directory = Path(configured_path).expanduser()
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory = directory.resolve()
    if not directory.is_dir():
        print(f"Directory does not exist: {directory}", file=sys.stderr)
        return 2

    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        print(f"No .md or .txt files found in: {directory}")
        return 0

    async def process_all() -> None:
        for path in files:
            if path.suffix.lower() == ".txt":
                try:
                    await MarkdownLoader(path).load()
                except ValueError as exc:
                    # The loader's existing extension check rejects .txt files.
                    print(f"{path.name}: {exc}")
                continue

            try:
                chunks = await process_markdown_file(path)
            except Exception as exc:
                print(f"Could not load {path}: {exc}", file=sys.stderr)
                continue

            print(f"{path.name}: {len(chunks)} chunks")
            for index, chunk in enumerate(chunks, start=1):
                preview = chunk.text[:100].replace("\n", " ")
                print(f"  {index}: {preview}")

    try:
        asyncio.run(process_all())
    except Exception as exc:
        print(f"Test run failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
