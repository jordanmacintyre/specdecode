#!/usr/bin/env bash
#
# dl_hf.sh — Download a Hugging Face model repo to a local directory.
#
# Default: Saves into ./models/<org>__<model_name> (relative to where script is run).
# You can override the destination with `-d` or `--dir`.
#
# This script uses the `huggingface-cli download` command from the
# huggingface_hub package to fetch model files directly (no need for Python code).
#
# Benefits:
# - Keeps model files inside your project (for reproducibility or portability).
# - Option to use **no symlinks** so all files are real copies (default here).
# - Option to specify an exact commit/branch/tag revision for reproducibility.
#
# Requirements:
# - huggingface_hub >= 0.20 (`pip install "huggingface_hub[cli]"`)
# - Internet access to pull from huggingface.co

set -euo pipefail
# -e → exit script if any command fails
# -u → treat unset variables as errors
# -o pipefail → fail if any command in a pipeline fails


# --- Step 1: Check CLI availability ---
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Error: huggingface-cli not found. Install with: pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi


# --- Step 2: Argument parsing ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_id> [-d OUTPUT_DIR] [-r REVISION] [--no-symlinks|--symlinks]" >&2
  echo "Example: $0 meta-llama/Llama-2-7b -d ./models/llama2 -r main"
  exit 1
fi

MODEL_ID=""        # e.g., "google/gemma-2b"
OUT_DIR="./models" # default base directory in current project
REVISION=""        # optional branch/tag/commit to fetch
USE_SYMLINKS="False" # default to downloading full files instead of symlinks

MODEL_ID="$1"; shift  # first positional argument is the model id, remove it from $@
# Remaining args are parsed in the loop below

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dir)        # set custom output directory
      OUT_DIR="$2"; shift 2;;
    -r|--revision)   # set commit/branch/tag for reproducible snapshot
      REVISION="$2"; shift 2;;
    --no-symlinks)   # ensure local copy is fully self-contained
      USE_SYMLINKS="false"; shift;;
    --symlinks)      # use symlinks to point into HF cache (saves disk space)
      USE_SYMLINKS="true"; shift;;
    -*)              # any other unknown flag
      echo "Unknown option: $1" >&2; exit 1;;
    *)               # unexpected positional argument
      echo "Unexpected argument: $1" >&2; exit 1;;
  esac
done


# --- Step 3: Prepare target directory name ---
# Replace `/` in "org/model" with `__` to make it safe for filesystem storage.
safe_name="$(echo "$MODEL_ID" | sed 's|/|__|g')"

# Full path where model files will be stored
TARGET_DIR="${OUT_DIR%/}/${safe_name}"
mkdir -p "$TARGET_DIR"  # create directory if not exists


# --- Step 4: Logging the plan ---
echo "Downloading '$MODEL_ID' → '$TARGET_DIR'"
echo "Revision: ${REVISION:-(default branch)}"
echo "Symlinks: $USE_SYMLINKS"


# --- Step 5: Build huggingface-cli arguments ---
args=(download "$MODEL_ID" \
      --local-dir "$TARGET_DIR" \
      --local-dir-use-symlinks "$USE_SYMLINKS")

# Optionally, only download model files (exclude datasets/spaces):
args+=(--repo-type model)

if [[ -n "$REVISION" ]]; then
  args+=(--revision "$REVISION")
fi


# --- Step 6: Perform the download ---
# This is idempotent: if you run again, it will only fetch updated files.
huggingface-cli "${args[@]}"


# --- Step 7: Done ---
echo "Done."
echo "Files are in: $TARGET_DIR"
