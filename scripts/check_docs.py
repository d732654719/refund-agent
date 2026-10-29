"""Check AGENTS.md links and remind maintainers about unchanged module notes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


def git(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, stdout=subprocess.PIPE, encoding="utf-8",
    )
    return [name for name in result.stdout.split("\0") if name]


def report(level: str, message: str) -> None:
    if os.getenv("GITHUB_ACTIONS") == "true":
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::{level}::{escaped}")
    else:
        print(f"{level.upper()}: {message}")


def owner(path: str, documents: set[str]) -> str | None:
    for parent in Path(path).parents:
        candidate = (parent / "AGENTS.md").as_posix()
        if candidate in documents:
            return candidate
    return None


def check(root: Path, base: str | None = None) -> int:
    files = set(git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"))
    documents = {
        name for name in files
        if (Path(name).name == "AGENTS.md" or name == "docs/documentation.md")
        and (root / name).is_file()
    }
    errors = 0
    for required in ("AGENTS.md", "docs/documentation.md"):
        if required not in documents:
            report("error", f"Missing documentation entry: {required}")
            errors += 1
    for name in sorted(documents):
        content = (root / name).read_text(encoding="utf-8")
        if not content.strip():
            report("error", f"Empty documentation: {name}")
            errors += 1
        if Path(name).name == "AGENTS.md" and len(content.splitlines()) > 100:
            report("warning", f"{name}: over 100 lines; review whether details belong closer to code.")
        # Only ordinary inline file links outside fenced code are checked.
        prose = re.sub(r"^```[^\n]*\n.*?^```[^\n]*(?:\n|$)", "", content, flags=re.M | re.S)
        for target in re.findall(r"\]\((<[^>]+>|[^\s)]+)\)", prose):
            link = urlsplit(target.strip("<>"))
            if link.scheme or link.netloc or not link.path:
                continue
            path = (root / name).parent / unquote(link.path)
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                report("warning", f"{name}: external checkout link not checked: {target}")
            elif not path.exists():
                report("error", f"{name}: missing local link: {target}")
                errors += 1

    if base:
        # Resolve to a commit before constructing a revision range.
        commit = git(root, "rev-parse", "--verify", f"{base}^{{commit}}")[0].strip()
        changed = set(git(root, "diff", "--name-only", "--no-renames", "-z", f"{commit}...HEAD", "--"))
    else:
        changed = set(git(root, "diff", "--name-only", "--no-renames", "-z", "HEAD", "--"))
        changed.update(git(root, "ls-files", "-z", "--others", "--exclude-standard"))

    by_document: dict[str, list[str]] = {}
    for name in sorted(changed):
        if Path(name).suffix.lower() == ".md":
            continue
        document = owner(name, documents)
        if document and document not in changed:
            by_document.setdefault(document, []).append(name)
    for document, names in sorted(by_document.items()):
        examples = ", ".join(names[:3])
        report(
            "warning",
            f"{document}: {len(names)} changed file(s), e.g. {examples}. "
            "Review responsibilities, interfaces and commands; update notes or explain why unchanged.",
        )
    print(
        f"Checked {len(documents)} documentation files; "
        f"{errors} link/entry error(s), {len(by_document)} module review reminder(s)."
    )
    return 1 if errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Compare merge-base with HEAD; default checks local tracked and untracked changes.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        return check(root, args.base)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        report("error", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
