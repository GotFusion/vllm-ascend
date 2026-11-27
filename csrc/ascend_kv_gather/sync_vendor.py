# SPDX-License-Identifier: Apache-2.0
"""Copy a revision-locked subset of ascend-kv-gather without local modifications."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FILES = (
    "LICENSE",
    "SOURCES.md",
    "aiv/main.asc",
    "aiv/protocol.h",
    "aiv/transfer.cpp",
    "aiv/vector_builder.h",
    "tests/vector_builder_test.cc",
)
ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "ascend-kv-gather"
MANIFEST = ROOT / "SOURCE_MANIFEST.json"


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args])


def check():
    manifest = json.loads(MANIFEST.read_text())
    if set(manifest["files"]) != set(FILES):
        raise ValueError("source manifest does not match the explicit import list")
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"vendored file differs: {name}")
    print(f"Verified {len(FILES)} unchanged source files at {manifest['revision']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--revision", help="commit to import; never reads uncommitted source files")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.source is None or args.revision is None:
        parser.error("--source and --revision are required unless --check is supplied")
    revision = git(args.source, "rev-parse", "--verify", f"{args.revision}^{{commit}}").decode().strip()
    # Read everything before replacing files, so a missing source cannot cause a partial import.
    contents = {name: git(args.source, "show", f"{revision}:{name}") for name in FILES}
    for name, data in contents.items():
        target = VENDOR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = {
        "project": "ascend-kv-gather",
        "revision": revision,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    check()


if __name__ == "__main__":
    main()
