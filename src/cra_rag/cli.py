"""Command-line entry point: `uv run cra-rag <command>`."""

import argparse
import sys
from pathlib import Path

from cra_rag import download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cra-rag",
        description="Cited question-answering over EU product cybersecurity regulation.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download_cmd = commands.add_parser(
        "download", help="download the regulatory texts (EU Publications Office) into corpus/raw/"
    )
    download_cmd.add_argument("--out", type=Path, default=Path("corpus/raw"))

    args = parser.parse_args(argv)

    if args.command == "download":
        try:
            manifest = download.download_all(args.out, log=print)
        except download.DownloadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        for filename, entry in manifest.items():
            print(f"  {filename}: {entry['bytes']:,} bytes, sha256 {entry['sha256'][:12]}...")
        print(f"Done: {len(manifest)} files in {args.out}")
    return 0
