#!/usr/bin/env bash
# Remove stale snapshot YAML files and zarr output directories from tests/data/ and
# tests/data-extended/. Defaults to dry-run; pass --confirm to actually delete.

set -euo pipefail

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=true

for arg in "$@"; do
    [[ "$arg" == "--confirm" ]] && DRY_RUN=false
done

$DRY_RUN && echo "[dry-run] Pass --confirm to actually delete."$'\n'

_delete() {
    if $DRY_RUN; then
        echo "  would remove: $1"
    else
        echo "  removing: $1"
        rm -rf "$1"
    fi
}

# --- tests/data/ ---
echo "=== tests/data/ ==="
for instrument_dir in "$TESTS_DIR/data"/*/; do
    [[ -d "$instrument_dir" ]] || continue

    # zarr outputs (gitignored, on disk)
    output_dir="${instrument_dir%/}/output"
    if [[ -d "$output_dir" ]]; then
        while IFS= read -r -d '' zarr; do
            _delete "$zarr"
        done < <(find "$output_dir" -maxdepth 2 -name "*.zarr" -print0)
    fi

    # snapshot YAMLs (git-tracked — will appear as deleted in git status)
    snapshot_dir="${instrument_dir%/}/snapshots"
    if [[ -d "$snapshot_dir" ]]; then
        while IFS= read -r -d '' yaml; do
            _delete "$yaml"
        done < <(find "$snapshot_dir" -maxdepth 1 -name "*.yaml" -print0)
    fi
done

# --- tests/data-extended/ ---
echo ""
echo "=== tests/data-extended/ ==="
for instrument_dir in "$TESTS_DIR/data-extended"/*/; do
    [[ -d "$instrument_dir" ]] || continue

    # zarr outputs (entire output/ dir, gitignored)
    output_dir="${instrument_dir%/}/output"
    [[ -d "$output_dir" ]] && _delete "$output_dir"

    # snapshot YAMLs (gitignored)
    snapshot_dir="${instrument_dir%/}/snapshots"
    if [[ -d "$snapshot_dir" ]]; then
        while IFS= read -r -d '' yaml; do
            _delete "$yaml"
        done < <(find "$snapshot_dir" -maxdepth 1 -name "*.yaml" -print0)
    fi
done

if $DRY_RUN; then
    echo ""
    echo "Re-run with --confirm to delete the above."
else
    echo ""
    echo "Done. Note: deleted snapshot YAMLs in tests/data/ are git-tracked."
    echo "Run 'git add -u tests/data/' to stage their removal."
fi
