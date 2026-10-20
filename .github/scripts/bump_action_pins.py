"""Raise ``uses:`` pins in workflow files to the newest upstream major tag.

Dependabot's github-actions ecosystem stopped producing updates on 2026-05-02 along with
its uv one, and unlike the lockfile there is no other mechanism watching these pins. This
script rewrites ``uses: owner/repo@vN`` to the upstream's current major tag and prints
what it changed.

Pins that are commit SHAs, branch names, or docker:// images are left alone on purpose:
only major-tag pins can be raised without changing what the pin means.

    python3 bump_action_pins.py [--workflows .github/workflows] [--dry-run]
"""

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

USES = re.compile(r"^(\s*(?:-\s+)?)uses:\s+([\w./-]+)/([\w.-]+)@([^\s#]+)(.*)$")
MAJOR = re.compile(r"^v(\d+)(?:\.\d+){0,2}$")


def latest_major(slug, current):
    """Return the newest major tag (e.g. ``v7``) for ``owner/repo``, or None.

    Uses the REST API directly rather than the gh CLI, so the script runs anywhere python
    does. GH_TOKEN or GITHUB_TOKEN raises the rate limit; anonymous is enough for a
    handful of lookups.
    """
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "rcpyci-action-refresher"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"https://api.github.com/repos/{slug}/releases/latest", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            tag = json.load(response).get("tag_name", "")
    except urllib.error.HTTPError as exc:
        print(f"::warning::{slug} lookup failed with HTTP {exc.code}, leaving it at {current}")
        return None
    except OSError as exc:
        print(f"::warning::{slug} lookup failed: {exc}")
        return None
    tag = str(tag).strip()
    match = MAJOR.match(tag)
    if not match:
        print(f"::warning::{slug} latest tag {tag!r} is not a version, skipping")
        return None
    return f"v{match.group(1)}"


def bump(path, dry_run):
    """Rewrite one workflow file, returning the changes made as (slug, old, new) triples."""
    changes = []
    lines = path.read_text().splitlines(keepends=True)
    for index, line in enumerate(lines):
        match = USES.match(line.rstrip("\n"))
        if not match:
            continue
        prefix, owner, repo, pinned, rest = match.groups()
        slug = f"{owner}/{repo}"
        if not MAJOR.match(pinned):
            continue  # sha or branch pin: not ours to move
        newest = latest_major(slug, pinned)
        if newest is None or newest == pinned:
            continue
        major = int(pinned[1:])
        if int(newest[1:]) <= major:
            continue
        lines[index] = f"{prefix}uses: {slug}@{newest}{rest}\n"
        changes.append((slug, pinned, newest))
    if changes and not dry_run:
        path.write_text("".join(lines))
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default=".github/workflows", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.workflows.is_dir():
        sys.exit(f"no workflows directory at {args.workflows}")

    all_changes = []
    for path in sorted(args.workflows.glob("*.yml")) + sorted(args.workflows.glob("*.yaml")):
        for slug, old, new in bump(path, args.dry_run):
            print(f"  {path.name}: {slug} {old} -> {new}")
            all_changes.append((path.name, slug, old, new))

    if not all_changes:
        print("action pins are already current, nothing to propose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
