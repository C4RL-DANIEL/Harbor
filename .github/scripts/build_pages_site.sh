#!/usr/bin/env bash
# =============================================================================
# Harbor — assembles the complete GitHub Pages site.
# =============================================================================
# Two pipelines publish this site: build_and_release.yml (which owns the update
# manifest) and deploy_update_server.yml (which owns the Flutter Web admin
# dashboard). actions/deploy-pages replaces the entire site with the uploaded
# artifact, so if each workflow published only its own half, whichever finished
# last would delete the other half — a dashboard deploy would 404 the manifest
# and a release would 404 the dashboard.
#
# Both workflows therefore call this script, and both publish the *whole* site:
# the dashboard plus, when release metadata is available, latest_version.json.
# The result is idempotent — whichever pipeline runs, the deployed site is
# complete — and the manifest generation lives in exactly one place.
#
# Manifest inputs come from the release job's outputs in build_and_release.yml
# and from the published GitHub Release in deploy_update_server.yml, so the two
# paths cannot drift.
#
# Environment:
#   SITE_DIR                 (required) directory to assemble the site into
#   REPO                     (required) owner/name, used in manifest URLs
#   API_BASE_URL             (required) origin the dashboard calls
#   ADMIN_BASE_HREF          (default /update-admin/) dashboard base href
#   MIN_SUPPORTED_VERSION    (optional) defaults to VERSION
#   VERSION, TAG, SHA256,    (optional group) when all four are present the
#   SIZE_BYTES, CHANGELOG_FILE, PUBLISHED_AT   manifest is written; otherwise it
#                                              is skipped with a warning
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

site_dir="${SITE_DIR:?SITE_DIR must be set}"
repository="${REPO:?REPO must be set}"
api_base_url="${API_BASE_URL:?API_BASE_URL must be set}"
admin_base_href="${ADMIN_BASE_HREF:-/update-admin/}"

# ---------------------------------------------------------------------------
# Normalise the base href: it must start and end with exactly one slash, and
# the directory it names is the dashboard's subdirectory inside the site.
# ---------------------------------------------------------------------------
case "${admin_base_href}" in
  /*) ;;
  *) admin_base_href="/${admin_base_href}" ;;
esac
case "${admin_base_href}" in
  */) ;;
  *) admin_base_href="${admin_base_href}/" ;;
esac
if [ "${admin_base_href}" = "//" ]; then
  admin_base_href="/update-admin/"
fi
admin_dir_name="${admin_base_href#/}"
admin_dir_name="${admin_dir_name%/}"
if [ -z "${admin_dir_name}" ]; then
  echo "::error::ADMIN_BASE_HREF '${admin_base_href}' does not name a directory." >&2
  exit 1
fi

rm -rf "${site_dir}"
mkdir -p "${site_dir}"

# ---------------------------------------------------------------------------
# 1. Flutter Web admin dashboard
# ---------------------------------------------------------------------------
server_app="${repo_root}/apps/update_server"
for required in pubspec.yaml lib/main.dart web/index.html; do
  if [ ! -f "${server_app}/${required}" ]; then
    echo "::error::apps/update_server/${required} is missing; cannot build the dashboard." >&2
    exit 1
  fi
done

echo "Building the Flutter Web dashboard (base href ${admin_base_href})..."
(
  cd "${server_app}"
  flutter pub get
  flutter build web --release \
    --base-href "${admin_base_href}" \
    --dart-define=HARBOR_API_BASE_URL="${api_base_url}"
)

web_build="${server_app}/build/web"
if [ ! -d "${web_build}" ]; then
  echo "::error::Expected Flutter Web output at apps/update_server/build/web." >&2
  exit 1
fi
for required in index.html flutter_bootstrap.js main.dart.js; do
  if [ ! -s "${web_build}/${required}" ]; then
    echo "::error::Web build member build/web/${required} is missing or empty." >&2
    exit 1
  fi
done

mkdir -p "${site_dir}/${admin_dir_name}"
cp -R "${web_build}/." "${site_dir}/${admin_dir_name}/"
if grep -q "${admin_base_href}" "${site_dir}/${admin_dir_name}/index.html"; then
  echo "Dashboard base href ${admin_base_href} is present in the built index.html."
else
  echo "::warning::Built dashboard index.html does not contain the configured base href ${admin_base_href}."
fi
echo "Dashboard staged at ${site_dir}/${admin_dir_name} ($(du -sh "${site_dir}/${admin_dir_name}" | cut -f1))."

# ---------------------------------------------------------------------------
# 2. latest_version.json — the frozen /api/v1/update-check payload
# ---------------------------------------------------------------------------
version="${VERSION:-}"
tag="${TAG:-}"
sha256="${SHA256:-}"
size_bytes="${SIZE_BYTES:-}"
changelog_file="${CHANGELOG_FILE:-}"
published_at="${PUBLISHED_AT:-}"

manifest_requested=0
for candidate in "${version}" "${tag}" "${sha256}" "${size_bytes}" "${changelog_file}"; do
  if [ -n "${candidate}" ]; then
    manifest_requested=1
  fi
done

manifest_written=0
if [ "${manifest_requested}" -eq 1 ]; then
  missing=""
  for pair in "VERSION:${version}" "TAG:${tag}" "SHA256:${sha256}" "SIZE_BYTES:${size_bytes}"; do
    name="${pair%%:*}"
    value="${pair#*:}"
    if [ -z "${value}" ]; then
      missing="${missing} ${name}"
    fi
  done
  if [ -n "${missing}" ]; then
    echo "::error::Manifest inputs are incomplete; missing:${missing}" >&2
    exit 1
  fi
  if ! printf '%s' "${sha256}" | grep -Eq '^[0-9a-f]{64}$'; then
    echo "::error::Refusing to publish: sha256 '${sha256}' is not 64 lowercase hex characters." >&2
    exit 1
  fi
  if ! printf '%s' "${size_bytes}" | grep -Eq '^[0-9]+$'; then
    echo "::error::Refusing to publish: size_bytes '${size_bytes}' is not an integer." >&2
    exit 1
  fi
  if [ -z "${changelog_file}" ] || [ ! -s "${changelog_file}" ]; then
    echo "::error::Changelog file '${changelog_file:-<unset>}' is missing or empty." >&2
    exit 1
  fi

  min_supported="${MIN_SUPPORTED_VERSION:-}"
  if [ -z "${min_supported}" ]; then
    min_supported="${version}"
  fi

  echo "Writing latest_version.json for ${version} (min supported ${min_supported})."
  SITE_DIR="${site_dir}" \
  REPOSITORY="${repository}" \
  VERSION="${version}" \
  TAG="${tag}" \
  SHA256="${sha256}" \
  SIZE_BYTES="${size_bytes}" \
  MIN_SUPPORTED="${min_supported}" \
  CHANGELOG_FILE="${changelog_file}" \
  PUBLISHED_AT="${published_at}" \
  python3 - <<'PY'
import datetime
import json
import os

with open(os.environ["CHANGELOG_FILE"], encoding="utf-8") as handle:
    changelog = handle.read().strip()

published_at = os.environ.get("PUBLISHED_AT", "").strip()
try:
    parsed = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    parsed = parsed.astimezone(datetime.timezone.utc)
except ValueError:
    parsed = datetime.datetime.now(datetime.timezone.utc)
published_at = parsed.replace(microsecond=0).isoformat().replace("+00:00", ".000Z")

version = os.environ["VERSION"]
tag = os.environ["TAG"]
repository = os.environ["REPOSITORY"]

payload = {
    "latest_version": version,
    "min_supported_version": os.environ["MIN_SUPPORTED"],
    "download_url": f"https://github.com/{repository}/releases/download/{tag}/app-release.apk",
    "sha256": os.environ["SHA256"],
    "size_bytes": int(os.environ["SIZE_BYTES"]),
    "changelog": changelog,
    "force_update": False,
    "published_at": published_at,
    "platform": "android",
    "update_available": True,
    "update_required": False,
    "release_notes_url": f"https://github.com/{repository}/releases/tag/{tag}",
}

destination = os.path.join(os.environ["SITE_DIR"], "latest_version.json")
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print(f"Wrote {destination}")
PY

  python3 -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8")); print("JSON OK", sys.argv[1])' \
    "${site_dir}/latest_version.json"

  # SHA256SUMS accompanies the manifest so a consumer can verify the APK it
  # downloads from the URL named in latest_version.json.
  printf '%s  app-release.apk\n' "${sha256}" > "${site_dir}/SHA256SUMS"
  manifest_written=1
else
  echo "::warning::No release metadata supplied; publishing the dashboard without latest_version.json."
fi

# ---------------------------------------------------------------------------
# 3. Landing page
# ---------------------------------------------------------------------------
dashboard_link=""
if [ -d "${site_dir}/${admin_dir_name}" ]; then
  dashboard_link="<li><a href=\"./${admin_dir_name}/\">Admin dashboard</a> - live release, force-update and feature-flag editor</li>"
fi
manifest_link=""
if [ "${manifest_written}" -eq 1 ]; then
  manifest_link="<li><a href=\"./latest_version.json\">latest_version.json</a> - the update-check contract</li>
                <li><a href=\"./SHA256SUMS\">SHA256SUMS</a> - checksums for the published release assets</li>"
fi

cat > "${site_dir}/index.html" <<HTML
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Harbor OTA ecosystem</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 3rem auto; max-width: 42rem; line-height: 1.6; padding: 0 1rem; }
      code { background: #f2f2f2; padding: 0.1rem 0.3rem; border-radius: 4px; }
    </style>
  </head>
  <body>
    <h1>Harbor OTA ecosystem</h1>
    <p>
      This GitHub Pages site publishes the artifacts produced by
      <code>build_and_release.yml</code> and <code>deploy_update_server.yml</code>.
      Both workflows deploy the complete site, so neither can remove the other's
      files.
    </p>
    <ul>
      ${dashboard_link}
      ${manifest_link}
    </ul>
  </body>
</html>
HTML

# ---------------------------------------------------------------------------
# 4. Final validation
# ---------------------------------------------------------------------------
if [ ! -s "${site_dir}/index.html" ]; then
  echo "::error::Pages site is incomplete: index.html is missing or empty." >&2
  exit 1
fi
if [ ! -s "${site_dir}/${admin_dir_name}/index.html" ]; then
  echo "::error::Pages site is incomplete: the dashboard is missing." >&2
  exit 1
fi

echo "Pages site assembled at ${site_dir}:"
( cd "${site_dir}" && find . -maxdepth 1 -mindepth 1 -printf '  %f\n' | sort )