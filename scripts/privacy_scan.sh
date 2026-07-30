#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

blocked_extensions='(parquet|db|sqlite|sqlite3|lance|jsonl|mbox|eml|vcf|heic|mov|mp4|zip)$'
if find . -type f -not -path './.git/*' -not -path './node_modules/*' -not -path './dist/*' | grep -Eiq "\\.${blocked_extensions}"; then
  echo "Blocked personal-data file type found."
  find . -type f -not -path './.git/*' -not -path './node_modules/*' -not -path './dist/*' | grep -Ei "\\.${blocked_extensions}"
  exit 1
fi

patterns='(/Users/[^/]+/|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|github_pat_|ghp_|sk-[A-Za-z0-9]|-----BEGIN .*PRIVATE KEY-----)'
if rg -n --hidden -g '!.git/**' -g '!node_modules/**' -g '!dist/**' -g '!package-lock.json' -g '!scripts/privacy_scan.sh' -e "$patterns" .; then
  echo "Potential private identifier or credential found."
  exit 1
fi

echo "Privacy scan passed."
