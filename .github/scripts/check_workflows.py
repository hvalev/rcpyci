"""Validate every workflow file: YAML structure, and shell syntax of each ``run`` block.

Workflow YAML is only parsed by GitHub when the event fires, and a shell mistake inside a
``run: |`` block - an unclosed ``$(``, a stray quote - is invisible until the job runs,
which for a scheduled workflow can be weeks. This catches both at review time.

    python3 check_workflows.py [--workflows .github/workflows]
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

import yaml


def check(path):
    """Return a list of human-readable problems found in one workflow file."""
    problems = []
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or "jobs" not in document:
        return [f"{path}: no jobs section"]
    for job_name, job in (document.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            script = step.get("run")
            if not script:
                continue
            handle = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
            try:
                handle.write(script)
            finally:
                handle.close()
            try:
                result = subprocess.run(["bash", "-n", handle.name], capture_output=True, text=True)
            finally:
                pathlib.Path(handle.name).unlink()
            if result.returncode != 0:
                label = step.get("name") or "unnamed step"
                problems.append(f"{path.name} :: {job_name} / {label}: {result.stderr.strip()}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflows", default=".github/workflows", type=pathlib.Path)
    args = parser.parse_args()

    if not args.workflows.is_dir():
        sys.exit(f"no workflows directory at {args.workflows}")

    problems = []
    files = sorted(args.workflows.glob("*.yml")) + sorted(args.workflows.glob("*.yaml"))
    for path in files:
        problems.extend(check(path))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        sys.exit(1)
    print(f"  {len(files)} workflow files: YAML valid, all run blocks parse")


if __name__ == "__main__":
    main()
