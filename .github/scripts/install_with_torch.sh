#!/usr/bin/env bash
# install_with_torch.sh - install a PyTorch ecosystem package matching a given torch version
#
# Usage:
#   ./install_with_torch.sh <package> <torch_version> [--dry-run] [-h|--help]
#
# Examples:
#   ./install_with_torch.sh torchaudio 2.10.0
#   ./install_with_torch.sh torchvision "2.10"
#   ./install_with_torch.sh torchaudio 2100 --dry-run
#
# Accepted torch version formats:
#   2.10.0  -> torch2100
#   2.10    -> torch2100 (preferred), then torch210
#   2.1     -> torch2100 (preferred), then torch210, then torch21
#   2100    -> torch2100
#
# Exit codes:
#   0  success
#   1  generic failure
#   2  invalid arguments
#   3  pip index query failed
#   4  no matching version found
#   5  pip install failed

set -euo pipefail

# ---- helpers ---------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: install_with_torch.sh <package> <torch_version> [--dry-run]

Arguments:
  package        Package name, e.g. torchaudio, torchvision, torchtext
  torch_version  Target torch version, accepts: 2.10.0 / 2.10 / 2100 / etc.
  --dry-run      Print the resolved version without installing

Examples:
  ./install_with_torch.sh torchaudio 2.10.0
  ./install_with_torch.sh torchvision 2.10
  ./install_with_torch.sh torchaudio 2100 --dry-run
EOF
}

color() {
    # color <code> <text...>
    local code=$1; shift
    printf '\033[%dm%s\033[0m\n' "$code" "$*"
}

say()  { color 36 "$@"; }   # cyan
note() { color 90 "$@"; }   # gray
ok()   { color 32 "$@"; }   # green
warn() { color 33 "$@" >&2; }
die()  { color 31 "$@" >&2; exit 1; }

# ---- preflight -------------------------------------------------------------
command -v pip >/dev/null 2>&1 || die "pip not found in PATH."

# ---- argument parsing ------------------------------------------------------
if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi
case "${1:-}" in -h|--help) usage; exit 0;; esac

PKG="$1"
TORCH_VER="${2:-}"
DRY_RUN=0
if [[ $# -eq 3 ]]; then
    case "$3" in
        --dry-run) DRY_RUN=1 ;;
        *) die "Unknown argument: $3";;
    esac
fi

[[ -n "$PKG"      ]] || die "Package name is empty."
[[ -n "$TORCH_VER" ]] || die "Torch version is empty."

# ---- 1. normalize the torch version tag -----------------------------------
# "2.10.0" -> "2100",  "2.10" -> "210",  "2.1" -> "21"
TORCH_TAG="$(printf '%s' "$TORCH_VER" | tr -cd '0-9')"
[[ -n "$TORCH_TAG" ]] || die "Cannot parse torch version from '$TORCH_VER'."

# Build candidate tag list (most specific first, then fall back).
#   "2.10.0" -> ["2100"]
#   "2.10"   -> ["2100", "210"]
#   "2.1"    -> ["2100", "210", "21"]
#   "2100"   -> ["2100"]
declare -a TAG_CANDIDATES
LEN=${#TORCH_TAG}
if (( LEN < 4 )); then
    if (( LEN < 3 )); then TAG_CANDIDATES+=("${TORCH_TAG}00"); fi
    TAG_CANDIDATES+=("${TORCH_TAG}0")
fi
TAG_CANDIDATES+=("$TORCH_TAG")

# Dedupe while preserving order
declare -a UNIQUE_TAGS=()
for t in "${TAG_CANDIDATES[@]}"; do
    if [[ ! " ${UNIQUE_TAGS[*]} " == *" $t "* ]]; then
        UNIQUE_TAGS+=("$t")
    fi
done
TAG_CANDIDATES=("${UNIQUE_TAGS[@]}")

say "Looking for $PKG matching torch tag: ${TAG_CANDIDATES[*]}"

# ---- 2. query pip index ----------------------------------------------------
note "Querying 'pip index versions $PKG' ..."
if ! PIP_OUT="$(pip index versions "$PKG" 2>&1)"; then
    die "pip index failed. Output was:
$PIP_OUT"
fi

# ---- 3. parse the "Available versions:" line -------------------------------
VERSION_LINE="$(printf '%s\n' "$PIP_OUT" | grep -E '^Available versions:' | head -n 1 || true)"
if [[ -z "$VERSION_LINE" ]]; then
    die "Could not find an 'Available versions:' line in pip output:
$PIP_OUT"
fi

# Split comma-separated list, trim each entry
declare -a VERSIONS=()
while IFS= read -r v; do
    v="${v#"${v%%[![:space:]]*}"}"   # ltrim
    v="${v%"${v##*[![:space:]]}"}"   # rtrim
    [[ -n "$v" ]] && VERSIONS+=("$v")
done < <(printf '%s' "${VERSION_LINE#Available versions:}" | tr ',' '\n')

if (( ${#VERSIONS[@]} == 0 )); then
    die "No versions parsed from pip output."
fi

# ---- 4. find first match across tag candidates -----------------------------
MATCHED=""
MATCHED_TAG=""
for tag in "${TAG_CANDIDATES[@]}"; do
    # Anchors (\. or $) prevent torch21 from spuriously matching torch2100.
    pattern="torch${tag}(\.|$)"
    for v in "${VERSIONS[@]}"; do
        if [[ "$v" =~ $pattern ]]; then
            MATCHED="$v"
            MATCHED_TAG="$tag"
            break 2
        fi
    done
done

# ---- 5. report -------------------------------------------------------------
if [[ -z "$MATCHED" ]]; then
    warn "No wheel of $PKG matches torch $TORCH_VER (tried tags: ${TAG_CANDIDATES[*]})"
    echo
    say "Available torch tags in $PKG:"
    printf '%s\n' "$PIP_OUT" | grep -oE 'torch[0-9]+' | sort -u | sed 's/^/  /'
    exit 4
fi

echo
ok  "Resolved version: $MATCHED"
note "(matched torch tag: $MATCHED_TAG)"

if (( DRY_RUN )); then
    echo
    warn "DryRun: skipping install."
    exit 0
fi

# ---- 6. install -----------------------------------------------------------
echo
say "Running: pip install $PKG==$MATCHED"
echo
if pip install "$PKG==$MATCHED"; then
    echo
    ok "Done. Installed $PKG==$MATCHED"
else
    rc=$?
    die "pip install exited with code $rc"
fi