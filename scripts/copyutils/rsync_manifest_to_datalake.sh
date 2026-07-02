#!/usr/bin/env bash
# Copy an exact manifest of files from the current host to datalake via rsync.
#
# Intended NBU workflow:
#   Wrangell -> run this script in tmux -> rsync to auto on Elias
#
# Safety defaults:
#   - dry-run unless --copy is passed
#   - exact --files-from manifest only
#   - no --delete, chmod/chown, recursive source sweep, or wildcard copy
#   - refuses manifest entries with slashes, '..', leading '-', or shell metacharacters
#   - refuses destination files larger than source
#   - treats same-size destination files as complete and resumes shorter partial files

set -euo pipefail

REMOTE="auto"
ALLOWED_SOURCE_ROOT="/scratch/yewen/BCM"
ALLOWED_DEST_ROOT="/mnt/datalake/data/TRBD-53761/TRBD001/NBU"
SOURCE_DIR=""
DEST_DIR=""
FILES_FROM=""
EXPECTED_COUNT=""
DO_COPY=0
VALIDATE_FFPROBE=1
ALLOW_SOURCE_OUTSIDE_ROOT=0
ALLOW_DEST_OUTSIDE_ROOT=0
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20)

usage() {
  cat <<'USAGE'
Usage:
  scripts/copyutils/rsync_manifest_to_datalake.sh \
    --source-dir /scratch/.../panel_out \
    --dest-dir /mnt/datalake/.../video/sleep_synced \
    --files-from approved_files.txt \
    --expected-count 27 \
    [--remote auto] [--copy] [--no-ffprobe] \
    [--allow-source-outside-root] [--allow-dest-outside-root]

Default is dry-run. Pass --copy only after the exact source, destination, and
manifest have been reviewed and approved.

The manifest must contain one basename per line. Subdirectories are intentionally
not accepted by this helper.
USAGE
}

while (($#)); do
  case "$1" in
    --source-dir) SOURCE_DIR="${2:?}"; shift 2 ;;
    --dest-dir) DEST_DIR="${2:?}"; shift 2 ;;
    --files-from) FILES_FROM="${2:?}"; shift 2 ;;
    --expected-count) EXPECTED_COUNT="${2:?}"; shift 2 ;;
    --remote) REMOTE="${2:?}"; shift 2 ;;
    --copy) DO_COPY=1; shift ;;
    --dry-run) DO_COPY=0; shift ;;
    --no-ffprobe) VALIDATE_FFPROBE=0; shift ;;
    --allow-source-outside-root) ALLOW_SOURCE_OUTSIDE_ROOT=1; shift ;;
    --allow-dest-outside-root) ALLOW_DEST_OUTSIDE_ROOT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_arg() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing required argument: $name" >&2
    usage >&2
    exit 2
  fi
}

local_file_size() {
  local path="$1"
  if stat -c '%s' "$path" >/dev/null 2>&1; then
    stat -c '%s' "$path"
  else
    stat -f '%z' "$path"
  fi
}

local_file_sha256() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  else
    shasum -a 256 "$path" | awk '{print $1}'
  fi
}

require_arg "--source-dir" "$SOURCE_DIR"
require_arg "--dest-dir" "$DEST_DIR"
require_arg "--files-from" "$FILES_FROM"
require_arg "--expected-count" "$EXPECTED_COUNT"

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "source-dir does not exist: $SOURCE_DIR" >&2
  exit 1
fi
if [[ ! -f "$FILES_FROM" ]]; then
  echo "files-from manifest does not exist: $FILES_FROM" >&2
  exit 1
fi
if [[ "$ALLOW_SOURCE_OUTSIDE_ROOT" -eq 0 && "$SOURCE_DIR" != "$ALLOWED_SOURCE_ROOT"/* ]]; then
  echo "refusing source outside $ALLOWED_SOURCE_ROOT: $SOURCE_DIR" >&2
  exit 1
fi
if [[ "$ALLOW_DEST_OUTSIDE_ROOT" -eq 0 && "$DEST_DIR" != "$ALLOWED_DEST_ROOT"/* ]]; then
  echo "refusing destination outside $ALLOWED_DEST_ROOT: $DEST_DIR" >&2
  exit 1
fi
if ! [[ "$EXPECTED_COUNT" =~ ^[0-9]+$ ]] || (( EXPECTED_COUNT < 1 )); then
  echo "expected-count must be a positive integer: $EXPECTED_COUNT" >&2
  exit 1
fi

TMP_MANIFEST="$(mktemp /tmp/rsync_manifest.XXXXXX)"
TMP_SIZES="$(mktemp /tmp/rsync_manifest_sizes.XXXXXX)"
cleanup() {
  rm -f "$TMP_MANIFEST" "$TMP_SIZES"
}
trap cleanup EXIT

grep -v '^[[:space:]]*$' "$FILES_FROM" > "$TMP_MANIFEST" || true

count="$(wc -l < "$TMP_MANIFEST" | tr -d ' ')"
if [[ "$count" != "$EXPECTED_COUNT" ]]; then
  echo "manifest count mismatch: expected=$EXPECTED_COUNT actual=$count" >&2
  exit 1
fi
if [[ "$(sort "$TMP_MANIFEST" | uniq -d | wc -l | tr -d ' ')" != "0" ]]; then
  echo "manifest contains duplicate entries" >&2
  sort "$TMP_MANIFEST" | uniq -d >&2
  exit 1
fi

if ! awk '
  /^[A-Za-z0-9._-]+$/ && $0 !~ /^\-/ && $0 !~ /\.\./ { next }
  { print "bad manifest entry: " $0 > "/dev/stderr"; bad=1 }
  END { exit bad }
' "$TMP_MANIFEST"; then
  exit 1
fi

while IFS= read -r name; do
  if [[ ! -f "$SOURCE_DIR/$name" ]]; then
    echo "source file missing: $SOURCE_DIR/$name" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\n' \
    "$name" \
    "$(local_file_size "$SOURCE_DIR/$name")" \
    "$(local_file_sha256 "$SOURCE_DIR/$name")" >> "$TMP_SIZES"
done < "$TMP_MANIFEST"

echo "PRECHECK"
echo "source_dir=$SOURCE_DIR"
echo "dest=$REMOTE:$DEST_DIR"
echo "files=$count"
echo "mode=$([[ "$DO_COPY" -eq 1 ]] && echo copy || echo dry-run)"
echo "ffprobe_validation=$([[ "$VALIDATE_FFPROBE" -eq 1 ]] && echo enabled || echo disabled)"

ssh "${SSH_OPTS[@]}" "$REMOTE" "bash -s" <<REMOTE_CHECK
set -euo pipefail
DST='$DEST_DIR'
bad=0
parent=\$(dirname "\$DST")
if [[ ! -d "\$parent" ]]; then
  echo "DEST_PARENT_MISSING	\$parent"
  exit 1
fi
if [[ ! -d "\$DST" ]]; then
  echo "DEST_DIR_MISSING_WOULD_CREATE	\$DST"
fi
while IFS=\$'\\t' read -r name src_size src_sha256; do
  target="\$DST/\$name"
  if [[ -e "\$target" ]]; then
    dst_size=\$(stat -c '%s' "\$target")
    if (( dst_size > src_size )); then
      echo "DEST_TOO_LARGE	\$name	dst=\$dst_size	src=\$src_size"
      bad=1
    elif (( dst_size == src_size )); then
      dst_sha256=\$(sha256sum "\$target" | awk '{print \$1}')
      if [[ "\$dst_sha256" == "\$src_sha256" ]]; then
        echo "DEST_COMPLETE	\$name	\$dst_size	sha256=\$dst_sha256"
      else
        echo "DEST_SAME_SIZE_DIFFERENT	\$name	size=\$dst_size	src_sha256=\$src_sha256	dst_sha256=\$dst_sha256"
        bad=1
      fi
    else
      echo "DEST_PARTIAL	\$name	dst=\$dst_size	src=\$src_size"
    fi
  else
    echo "DEST_MISSING	\$name	src=\$src_size"
  fi
done <<'SIZES'
$(cat "$TMP_SIZES")
SIZES
exit "\$bad"
REMOTE_CHECK

if [[ "$DO_COPY" -eq 1 ]]; then
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$DEST_DIR'"
fi

RSYNC_ARGS=(
  -t
  -v
  --human-readable
  --progress
  --partial
  --append-verify
  --checksum
  --no-perms
  --no-owner
  --no-group
  --files-from="$TMP_MANIFEST"
  --rsh="ssh ${SSH_OPTS[*]}"
)
if [[ "$DO_COPY" -eq 0 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "RSYNC_START"
rsync "${RSYNC_ARGS[@]}" "$SOURCE_DIR"/ "$REMOTE:$DEST_DIR"/
echo "RSYNC_DONE"

if [[ "$DO_COPY" -eq 1 ]]; then
  echo "VALIDATE_MANIFEST"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "bash -s" <<REMOTE_VALIDATE
set -euo pipefail
DST='$DEST_DIR'
expected='$EXPECTED_COUNT'
validate_ffprobe='$VALIDATE_FFPROBE'
copy_validation_fail=0
ffprobe_fail=0
total=0
while IFS=\$'\\t' read -r name src_size src_sha256; do
  total=\$((total + 1))
  target="\$DST/\$name"
  if [[ ! -f "\$target" ]]; then
    echo "DEST_MISSING_AFTER_COPY	\$name"
    copy_validation_fail=\$((copy_validation_fail + 1))
    continue
  fi
  dst_size=\$(stat -c '%s' "\$target")
  if (( dst_size != src_size )); then
    echo "DEST_SIZE_MISMATCH_AFTER_COPY	\$name	dst=\$dst_size	src=\$src_size"
    copy_validation_fail=\$((copy_validation_fail + 1))
    continue
  fi
  dst_sha256=\$(sha256sum "\$target" | awk '{print \$1}')
  if [[ "\$dst_sha256" != "\$src_sha256" ]]; then
    echo "DEST_SHA256_MISMATCH_AFTER_COPY	\$name	src_sha256=\$src_sha256	dst_sha256=\$dst_sha256"
    copy_validation_fail=\$((copy_validation_fail + 1))
    continue
  fi
  if [[ "\$validate_ffprobe" == "1" ]]; then
    if ! ffprobe -v error -show_entries format=duration -of csv=p=0 "\$target" >/dev/null; then
      echo "FFPROBE_FAIL	\$name"
      ffprobe_fail=\$((ffprobe_fail + 1))
    fi
  fi
done <<'FILES'
$(cat "$TMP_SIZES")
FILES
echo "manifest_total=\$total"
echo "manifest_expected=\$expected"
echo "copy_validation_fail=\$copy_validation_fail"
echo "ffprobe_fail=\$ffprobe_fail"
test "\$total" -eq "\$expected"
test "\$copy_validation_fail" -eq 0
test "\$ffprobe_fail" -eq 0
REMOTE_VALIDATE
fi
