#!/usr/bin/env bash
# Build the image under podman and verify the running container serves /search.
# Follows the build doctrine: remove any stale image, check the build exit code,
# run the container, confirm the endpoints respond, then clean up.
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE=hilbert-rag:verify
NAME=hilbert-rag-verify
PORT="${PORT:-8137}"   # host port; override with PORT=... if it collides

echo "==> remove stale container and image tag (cached layers are kept for a fast rebuild)"
podman rm -f "$NAME" >/dev/null 2>&1 || true
podman untag "$IMAGE" >/dev/null 2>&1 || true

echo "==> build (layer cache on, so unchanged steps are reused and need no network)"
podman build --layers -t "$IMAGE" .
echo "BUILD EXIT: $?"

echo "==> run (demo mode, the image default)"
podman run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null

cleanup() { podman rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Probe the endpoints from INSIDE the container, so the check is authoritative even when
# host port forwarding is blocked (rootless podman / pasta in restricted sandboxes). The
# slim image ships python but not curl, so use urllib.
GET='import urllib.request,sys; print(urllib.request.urlopen("http://localhost:8000"+sys.argv[1]).read().decode())'
POST='import urllib.request,json,sys; r=urllib.request.Request("http://localhost:8000"+sys.argv[2],data=sys.argv[1].encode(),headers={"content-type":"application/json"}); print(urllib.request.urlopen(r).read().decode())'

echo "==> wait for startup"
for _ in $(seq 1 30); do
  if podman exec "$NAME" python -c 'import urllib.request; urllib.request.urlopen("http://localhost:8000/healthz")' >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "==> /healthz";                              podman exec "$NAME" python -c "$GET" /healthz
echo "==> /search (default backend)";             podman exec "$NAME" python -c "$POST" '{"query":"space filling curves for retrieval","k":3}' /search
echo "==> /search (sfc backend, with a filter)";  podman exec "$NAME" python -c "$POST" '{"query":"indexing","k":3,"backend":"sfc","filter":{"categories":["cs.LG"]}}' /search
echo "==> /ask";                                  podman exec "$NAME" python -c "$POST" '{"query":"what is this corpus about?","k":3}' /ask

echo "==> host port check (best effort; may be blocked in a sandbox)"
if curl -fsS "http://localhost:$PORT/healthz" >/dev/null 2>&1; then
  echo "    host port $PORT reachable"
else
  echo "    host port $PORT not reachable here; the in-container checks above are authoritative"
fi

echo "CONTAINER VERIFY: OK"
