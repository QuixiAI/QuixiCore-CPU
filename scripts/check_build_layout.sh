#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
found=0

for path in "$root"/build-* "$root"/cmake-build-*; do
  if [ -d "$path" ]; then
    echo "obsolete build directory: $path" >&2
    found=1
  fi
done

if [ "$found" -ne 0 ]; then
  echo "use a checked-in profile under $root/build/ instead" >&2
  exit 1
fi
