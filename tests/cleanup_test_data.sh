#!/usr/bin/env bash
# Remove stale snapshot JSON files and zarr output directories from tests/data/ and
# tests/data-extended/. Defaults to dry-run; pass --confirm to actually delete.
#
# WARNING: tests/data-extended/<Instrument> are symlinks into an external data store,
# so --confirm deletes that store's reference snapshots too, not just local outputs.
# Only run it when you intend to regenerate every snapshot with --update-snapshots.
# Read the dry-run output before confirming.

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

    # snapshot JSONs (git-tracked — will appear as deleted in git status)
    snapshot_dir="${instrument_dir%/}/snapshots"
    if [[ -d "$snapshot_dir" ]]; then
        while IFS= read -r -d '' snapshot; do
            _delete "$snapshot"
        done < <(find "$snapshot_dir" -maxdepth 1 -name "*.json" -print0)
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

    # snapshot JSONs (gitignored)
    snapshot_dir="${instrument_dir%/}/snapshots"
    if [[ -d "$snapshot_dir" ]]; then
        while IFS= read -r -d '' snapshot; do
            _delete "$snapshot"
        done < <(find "$snapshot_dir" -maxdepth 1 -name "*.json" -print0)
    fi
done

if $DRY_RUN; then
    echo ""
    echo "Re-run with --confirm to delete the above."
else
    echo ""
    echo "Done. Note: deleted snapshots in tests/data/ are git-tracked."
    echo "Run 'git add -u tests/data/' to stage their removal."
fi
