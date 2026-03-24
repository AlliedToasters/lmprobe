"""Command-line interface for lmprobe utilities."""

from __future__ import annotations

import argparse
import logging
import sys


def _cmd_migrate(args: argparse.Namespace) -> None:
    """Run the migrate sub-command."""
    from .sharing import migrate_dataset

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    result = migrate_dataset(
        repo_id=args.repo_id,
        shard_max_bytes=args.shard_max_bytes,
        token=args.token,
        dry_run=args.dry_run,
    )
    print(result)


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``lmprobe`` CLI."""
    parser = argparse.ArgumentParser(
        prog="lmprobe",
        description="lmprobe command-line utilities",
    )
    sub = parser.add_subparsers(dest="command")

    # --- migrate ---
    migrate_p = sub.add_parser(
        "migrate",
        help=(
            "Migrate a full_sequence dataset to use "
            "last-token shard splitting"
        ),
    )
    migrate_p.add_argument(
        "repo_id",
        help="HuggingFace repo ID (e.g. 'username/my-activations')",
    )
    migrate_p.add_argument(
        "--shard-max-bytes",
        type=int,
        default=1_073_741_824,
        help="Max bytes per shard (default: 1 GB)",
    )
    migrate_p.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token",
    )
    migrate_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute new plan without uploading",
    )
    migrate_p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "migrate":
        _cmd_migrate(args)


if __name__ == "__main__":
    main()
