#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
printf 'golden bytes\n' > "${WORK}/object"
sha="$(shasum -a 256 "${WORK}/object" | awk '{print $1}')"
bytes="$(wc -c < "${WORK}/object" | tr -d ' ')"
jq -n --arg s "${sha}" --argjson b "${bytes}" '{
  timed_prompt_pool:[{r2_path:"hidden/pool-01.golden.json",sha256:$s,bytes:$b}],
  warmup_golden:{r2_path:"hidden/warmup-01.golden.json",sha256:$s,bytes:$b},
  live_golden_speculative:{dflash1:{"pool-01":{r2_path:"hidden/pool-01.dflash1.golden.json",sha256:$s,bytes:$b}}}
}' > "${WORK}/contract.json"
cat > "${WORK}/download" <<'MOCK'
#!/usr/bin/env bash
cp "${MOCK_OBJECT}" "$2"
MOCK
chmod +x "${WORK}/download"
base=(env TF_CONTRACT="${WORK}/contract.json" TF_R2_DOWNLOADER="${WORK}/download" MOCK_OBJECT="${WORK}/object" R2_BUCKET_ENDPOINT=https://example.invalid)
! "${base[@]}" "${ROOT}/tools/fetch-goldens.sh" --r2-path hidden/pool-01.golden.json --sha256 "${sha}" --bytes "${bytes}" --out "${WORK}/refused" >/dev/null 2>&1
"${base[@]}" "${ROOT}/tools/fetch-goldens.sh" --allow-hidden --r2-path hidden/pool-01.golden.json --sha256 "${sha}" --bytes "${bytes}" --out "${WORK}/one" >/dev/null
cmp "${WORK}/object" "${WORK}/one"
"${base[@]}" "${ROOT}/tools/fetch-goldens.sh" --all --out "${WORK}/all" >/dev/null
[[ -f "${WORK}/all/pool-01.golden.json" && -f "${WORK}/all/pool-01.dflash1.golden.json" ]]
! "${base[@]}" "${ROOT}/tools/fetch-goldens.sh" --allow-hidden --r2-path public/x --sha256 "${sha}" --bytes 1 --out "${WORK}/bad" >/dev/null 2>&1
printf 'test-fetch-goldens.sh: PASS\n'
