# CI coverage

## Result

CI uses offline tests and JSON validation. The `macos-security` job in
`.github/workflows/ci.yml` runs every test below on `macos-15`.

1. `tools/test-author-prompts.sh` checks prompt authoring with synthetic tokenizer data.
2. `tools/test-bench-worker-staging.sh` checks sandboxed worker staging and self-test.
3. `tools/test-benchmark-facade-env.sh` checks serial and DFlash benchmark arguments.
4. `tools/test-build-cache.sh` checks build-cache keys, save, and restore.
5. `tools/test-calibrate-box.sh` checks calibration commands and guards with mocks.
6. `tools/test-editable-surface-size.sh` checks the compressed editable-surface limit.
7. `tools/test-fetch-benchd.sh` checks local benchd channel resolution and verification.
8. `tools/test-fetch-goldens.sh` checks hidden golden download selection and verification.
9. `tools/test-golden-arming-patch.sh` checks synthetic golden arming patches.
10. `tools/test-new-track.sh` checks track creation.
11. `tools/test-participant-guard-docs.sh` checks documented guard values against the fixture.
12. `tools/test-private-golden-dir.sh` checks private golden path containment.
13. `tools/test-ranked-box-preflight-env.sh` checks ranked preflight variable names and override refusal.
14. `tools/test-ranked-box-preflight.sh` checks that credential refusal precedes box checks.
15. `tools/test-record-goldens.sh` checks mocked serial, warm-up, oracle, and apply phases.
16. `tools/test-reference-stamp.sh` checks the baseline commit stamp.
17. `tools/test-resident-up.sh` checks resident lifecycle and containment with a mock worker.
18. `tools/test-sandboxed-build.sh` checks the build sandbox profile.
19. `tools/test-sandboxed-cli.sh` checks CLI sandbox containment.
20. `tools/test-spec-declaration.sh` checks serial and DFlash declarations.
21. `tools/test-stage-baseline-workspace.sh` checks baseline workspace staging and target verification.
22. `tools/test-stage-drafter.sh` checks drafter staging, pins, and declaration guards.
23. `tools/test-stop-leaked-residents.sh` checks resident process cleanup.
24. `tools/test-submission-security.sh` checks editable-surface and workflow security gates.
25. `tools/test-trusted-cli-containment.sh` checks that trusted workers stay inside the baseline workspace.
26. `tools/test-vendored-subset.sh` checks the deletions-only engine subset guard.
27. `tools/test-yukon-run.sh` checks ranked and local Yukon routing.
28. `tools/test-zig-bin.sh` checks Zig `0.17.0` selection and refusal.

The `lint` job on Linux (`ubuntu-latest`, or a Darkbloom Linux Sandbox) runs
shell syntax, JSON, manifest, editable archive size, workflow egress, and
actionlint checks. It does not run the `tools/test-*.sh` loop.

The tests do not use the network, a GPU, or model weights.

## Runner routing

`.github/workflows/darkbloom-route.yml` chooses where each CI job runs: a
Darkbloom Sandbox when the Darkbloom plane grants one, otherwise the job's
GitHub-hosted fallback. The route job runs on GitHub-hosted `ubuntu-24.04`
with no token permissions and no checkout. Each routed job runs on
`fromJSON(needs.route.outputs.routes)['<lane>']`. The lane is the job name.

| Lane | Darkbloom profile | Fallback |
|---|---|---|
| `lint` | `linux_arm64` | `ubuntu-latest` |

`macos-security` is not routed. It always runs on GitHub-hosted `macos-15`.
On a self-hosted runner, the Seatbelt scripts take the ranked Mac's official
path (`RUNNER_ENVIRONMENT=self-hosted`), so the job cannot run as plain CI
there.

Routing is off unless the repository variable `DARKBLOOM_ROUTING` is exactly
`on`. While it is off, and for pull requests from forks, every job keeps its
fallback and the route job sends no request. When it is on, the route job
sends one `POST /v1/ci/routes` to `DARKBLOOM_SANDBOX_URL` (default
`https://api-sandbox.darkbloom.dev`). The request names only the repository,
the run ID, the run attempt and the lanes. It holds no credential. Errors,
timeouts and malformed replies keep the fallback. The plane never fails the
route job.

Darkbloom Linux Sandboxes are ARM64; the `ubuntu-latest` fallback is x64. The
`lint` job picks the actionlint and ShellCheck assets that match the runner.
The decision is fixed for the run: **Re-run failed jobs** reuses it. To move a
run back to GitHub-hosted runners, turn routing off and use **Re-run all
jobs**.

The ranked pipeline, `.github/workflows/benchmark.yml`, is not routed.
