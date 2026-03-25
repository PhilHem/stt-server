#!/usr/bin/env bash
# Dependency direction checker — Go equivalent of deptrac.
# Enforces hexagonal architecture: inner packages import nothing from the project,
# server imports only inward, no adapter-to-adapter imports.
#
# Usage: ./scripts/check-deps.sh
# Exit 0 = clean, Exit 1 = violation found.
set -euo pipefail

MODULE="github.com/PhilHem/stt-server"
VIOLATIONS=0

# Define allowed project imports per package layer.
# Format: "package_path|allowed_import_1|allowed_import_2|..."
# Empty after | means no project imports allowed.
RULES=(
  "internal/config|"
  "internal/observe|"
  "internal/audio|"
  "internal/recognizer|"
  "internal/recognizer/sherpa|internal/recognizer"
  "internal/recognizer/remote|internal/recognizer|internal/recognizer/remote/sttpb"
  "internal/recognizer/remote/sttpb|"
  "internal/model|"
  "internal/server|internal/config|internal/audio|internal/recognizer|internal/observe"
  "cmd/stt-server|internal/config|internal/model|internal/observe|internal/recognizer|internal/recognizer/remote|internal/recognizer/sherpa|internal/server"
)

check_package() {
  local pkg_path="$1"
  local allowed_raw="$2"

  # Get all project imports: production + test + xtest
  local actual
  actual=$(go list -f '
{{- range .Imports}}{{.}}{{"\n"}}{{end -}}
{{- range .TestImports}}{{.}}{{"\n"}}{{end -}}
{{- range .XTestImports}}{{.}}{{"\n"}}{{end -}}
' "./${pkg_path}/" 2>/dev/null \
    | grep "^${MODULE}/" \
    | sed "s|^${MODULE}/||" \
    | sort -u \
    || true)

  if [ -z "$actual" ] && [ -z "$allowed_raw" ]; then
    return 0
  fi

  # Parse allowed imports
  IFS='|' read -ra allowed_list <<< "$allowed_raw"

  while IFS= read -r imp; do
    [ -z "$imp" ] && continue
    local found=0
    for allowed in "${allowed_list[@]}"; do
      [ -z "$allowed" ] && continue
      if [ "$imp" = "$allowed" ]; then
        found=1
        break
      fi
    done
    if [ "$found" -eq 0 ]; then
      echo "VIOLATION: ${pkg_path} → ${imp} (not allowed)"
      VIOLATIONS=$((VIOLATIONS + 1))
    fi
  done <<< "$actual"
}

echo "Checking dependency direction..."

for rule in "${RULES[@]}"; do
  pkg="${rule%%|*}"
  allowed="${rule#*|}"
  check_package "$pkg" "$allowed"
done

if [ "$VIOLATIONS" -gt 0 ]; then
  echo ""
  echo "FAILED: ${VIOLATIONS} dependency violation(s)."
  echo ""
  echo "Layer rules:"
  echo "  config, observe, audio, recognizer, model → (nothing)"
  echo "  recognizer/sherpa → recognizer (port only)"
  echo "  recognizer/remote → recognizer (port), recognizer/remote/sttpb (generated)"
  echo "  server → config, audio, recognizer, observe"
  echo "  cmd/stt-server → all internal packages"
  exit 1
fi

echo "OK: all dependency rules pass."
