#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-data/fastapi-docs}"
REPOSITORY="https://github.com/fastapi/fastapi.git"

if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION" >&2
  echo "Choose another destination or remove it before running this script." >&2
  exit 1
fi

mkdir -p "$(dirname "$DESTINATION")"

git clone \
  --depth 1 \
  --filter=blob:none \
  --sparse \
  "$REPOSITORY" \
  "$DESTINATION"

git -C "$DESTINATION" sparse-checkout set docs/en LICENSE

if [[ ! -d "$DESTINATION/docs/en" ]]; then
  echo "Expected documentation folder was not found: $DESTINATION/docs/en" >&2
  exit 1
fi

echo "Documentation corpus fetched to: $DESTINATION/docs/en"
echo "Upstream license file: $DESTINATION/LICENSE"