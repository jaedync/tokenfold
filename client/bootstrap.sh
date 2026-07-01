#!/usr/bin/env bash
# bootstrap.sh — one-command install entry point for the tokenfold usage hook.
#
# Served by the app at GET /install.sh with __TOKENFOLD_URL__ replaced by the
# server's own base URL, so onboarding a new machine is a single line:
#
#   curl -fsSL https://<your-server>/install.sh | bash -s -- --token 'tk_XXXX'
#
# Deliberately thin: fetch the latest client from GitHub, hand off to
# client/install-tokenfold-hook.sh, which owns all real logic (config
# resolution, hook registration, verification). Piped into bash — not sh —
# because the installer is bash-only.
#
# No sha pinning here: the installed self-updater (tokenfold-update.sh) pins
# shas from the first push onward, so main-at-install-time is good enough.
# Test seam: TOKENFOLD_BOOTSTRAP_TARBALL (full tarball URL, file:// ok),
# mirroring the self-updater's TOKENFOLD_UPDATE_TARBALL_BASE.
set -euo pipefail

# Substituted by the server when it serves this file. A raw copy (fetched
# straight from GitHub) still has the placeholder; treat that as "no default"
# rather than handing the installer a junk URL.
TOKENFOLD_URL_DEFAULT="__TOKENFOLD_URL__"
TARBALL_URL="${TOKENFOLD_BOOTSTRAP_TARBALL:-https://codeload.github.com/jaedync/tokenfold/tar.gz/refs/heads/main}"

say() { printf '[bootstrap] %s\n' "$*"; }
die() { printf '[bootstrap] error: %s\n' "$*" >&2; exit 1; }

# ── Args ──────────────────────────────────────────────────────────────────────
# Only --url and --token are recognized here (both take a value, so the parser
# must consume the value too or it could be misread as a flag). Everything
# else is the installer's business and passes through verbatim.
URL=""
INSTALLER_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --url)   [ $# -ge 2 ] || die "--url requires a value"
             URL="$2"; shift 2 ;;
    --token) [ $# -ge 2 ] || die "--token requires a value"
             INSTALLER_ARGS+=("--token" "$2"); shift 2 ;;
    *)       INSTALLER_ARGS+=("$1"); shift ;;
  esac
done

# ── Effective URL: explicit --url beats the baked default ─────────────────────
BAKED_URL="$TOKENFOLD_URL_DEFAULT"
case "$BAKED_URL" in __*) BAKED_URL="" ;; esac
EFFECTIVE_URL="${URL:-$BAKED_URL}"
[ -n "$EFFECTIVE_URL" ] || die "no server URL. Pass --url <SERVER_URL>, or fetch this script from your tokenfold server's /install.sh so the URL comes baked in"

# ── Preflight ─────────────────────────────────────────────────────────────────
# The installer needs all three anyway; checking here fails fast (and names
# the culprit) before any download happens.
for tool in curl tar python3; do
  command -v "$tool" >/dev/null 2>&1 || die "required command not found on PATH: $tool"
done

# ── Fetch + extract latest main ───────────────────────────────────────────────
TMPD="$(mktemp -d)" || die "could not create a temp directory"
trap 'rm -rf "$TMPD"' EXIT

say "fetching tokenfold client ($TARBALL_URL)"
curl -fsSL -m 120 -H "User-Agent: tokenfold-bootstrap" "$TARBALL_URL" -o "$TMPD/tokenfold.tar.gz" \
  || die "download failed: $TARBALL_URL"
tar -xzf "$TMPD/tokenfold.tar.gz" -C "$TMPD" || die "could not extract the downloaded tarball"

# codeload tarballs unpack to a single <repo>-<ref>/ dir whose exact name
# depends on the ref, so locate the installer by glob instead of hardcoding.
installer=""
for candidate in "$TMPD"/*/client/install-tokenfold-hook.sh; do
  [ -f "$candidate" ] && { installer="$candidate"; break; }
done
[ -n "$installer" ] || die "install-tokenfold-hook.sh not found in the downloaded tarball"

# ── Hand off ──────────────────────────────────────────────────────────────────
# --url goes FIRST so that if a caller sneaks another --url into the
# passthrough args, the installer's last-occurrence-wins parser honors theirs.
# ${arr[@]+...} guards the empty-array case under set -u on bash 3.2 (macOS).
say "running installer"
rc=0
bash "$installer" --url "$EFFECTIVE_URL" ${INSTALLER_ARGS[@]+"${INSTALLER_ARGS[@]}"} || rc=$?
exit "$rc"
