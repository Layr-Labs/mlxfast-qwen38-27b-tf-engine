# Participant contract

## Editable paths

1. A submission can edit `Vendor/TensorFold/`.
2. A submission can also edit `mtp-head.manifest.json`.
3. The `mtp-head.manifest.json` declaration is optional.
4. Its default declares serial mode.
5. Edit it to declare DFlash depth `1`.
6. The base `Vendor/TensorFold/` tree is the reference tree at the fixture's
   `baseline_reference_commit`. Submissions may change files in it.
7. The subset keeps TensorFold revision
   `eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab` (v1.1.6).
8. The compressed editable-surface archive limit is 20,971,520 bytes.

## Run contract

1. The target checkpoint is fixed by the fixture.
2. The drafter checkpoint is fixed by the fixture.
3. The worker speaks Engine Protocol v1 on standard input and output.
4. One resident owns each loaded model window.
5. Serial mode has no drafter path.
6. DFlash mode has depth `1`.
7. The target decides every emitted token.
8. The benchmarker owns time and score calculations.
9. A missing pin causes a refusal.
10. Do not specialize the engine for benchmark prompts or golden content.
11. Do not use memorized outputs or hidden expected tokens.
12. Do not read or transmit files outside the permitted runtime inputs.
13. Do not change, replace, or modify the target checkpoint.

## Draft mode

When a submission declares DFlash depth 1 in `mtp-head.manifest.json`, the engine must really run DFlash depth 1 with the pinned drafter in the candidate leg. It must not use another depth and must not bypass the drafter. The token stream cannot show the depth, so promoted entries are reviewed. An entry that uses another depth or bypasses the drafter is removed.

## How the score is measured

- Higher scores are better.
- The score is a unitless speedup against the live control leg.
- For each pair, prefill gain is control prefill s/token divided by candidate prefill s/token.
- For each pair, decode gain is control decode s/token divided by candidate decode s/token.
- The pair score is `prefill_gain^0.25 * decode_gain^0.75`.
- An official run has 2 pairs.
- The run score is the arithmetic mean of the 2 pair scores.
- Each leg first runs 1 discarded warm-up window.
- This warm-up window has a prefill pass and 128 decode steps.
- The measured window then has 1 unmeasured prefill warm-up pass.
- Each leg then has 1 timed prefill pass.
- Prefill s/token is the median timed prefill time divided by the prompt token count.
- With 1 timed pass, the median is that pass.
- Each leg has 1 timed decode run of 128 steps.
- Decode s/token is the seed window time plus the decode window time, divided by 128.
- The box calibration sample CV limit is 2.0% on prefill and decode.
- A live control leg can be at most 5% slower than the calibrated prefill mean.
- A live control leg can be at most 3% slower than the calibrated decode mean.
- No lower control-leg band is enabled.
- A run above a variance or control-leg limit writes no score.
- A correctness failure writes no score.
- A failed run reports `passed=false` and `score=null`.
- The score is not rounded.
- Benchd reports the score at full stored float precision.
- The promotion margin is 200 bips, or 2.00% (`minScoreImprovementBips` in
  `benchmark.json`).
- A new submission must beat the current promoted score by this margin.
- An equal score does not promote.

<!--
Sources for this section:
- `fixtures/qwen3_8_27b_tf_mlx_v1_track.json` fields `official_pairs`,
  `score_prefill_weight`, `score_decode_weight`, `seed_tokens`,
  `benchmark_decode_steps`, `official_prefill_warmup_runs`,
  `prefill_band_up_tolerance`, `decode_band_up_tolerance`,
  `prefill_band_down_enabled`, `decode_band_down_enabled`, and
  `decode_window_includes_seed`.
- The benchd source revision in the fixture.
- `benchmark.json` for the score formula and limits.
-->

## Guards that can refuse a run

- Prefill speedup ceiling: `1.5`.
- Decode speedup ceiling: `5.0`.
- A pair above a ceiling refuses the run.
- Prefill speedup floor: `0.95`.
- Decode speedup floor: `0.95`.
- A pair below a floor refuses the run.
- Decode time includes the seed step.
- Seed window maximum ratio: `1.4`.
- The ratio is seed seconds per token divided by the same leg's prefill seconds per token.
- A ratio above `1.4` refuses the run.
- Speed up the seed step together with prefill.
- A new PPID-1 process in the job coalition (`orphan-process`) refuses the run.
- A new PPID-1 process outside the job coalition also refuses the run unless it is a trusted system agent.
- A trusted system agent must use a root-owned system binary. An interpreter or a defence-in-depth code host on the guard's list does not pass. The list is not complete. A passing agent is logged.
- A candidate step that commits more tokens than its declared depth allows (`dflash-depth`) refuses the run.

## Build isolation limits

- The build and engine write only to non-indexed locations, and a sandboxed process cannot ask Spotlight to index.
- xcrun lookup cache: The build reads one shared xcrun lookup cache file of the job account. The trusted wrapper refreshes it and checks the Metal tool paths before and after the build. A process of the same account outside the sandbox could still change it during the build and restore it before the check.
- Xcode owner: The selected Xcode can be owned by an admin account instead of root. The Metal compiler can come from the selected Xcode app; it then follows the same owner and write checks, while a Metal compiler from the system Metal Toolchain must be owned by root. Before the build, the wrapper checks the macOS build tool directories, the SDK in use, and python3. It refuses with `toolchain-writable` if a path is writable by group or other. It also refuses if a path owner is neither root nor the Xcode owner. The check also covers ACLs and the Xcode app directory, Contents, Platforms, and Toolchains directories, and fails closed when a path is missing.
- Shared account orphans: Runner jobs share one account. The guard uses the Runner.Worker resource coalition as the job coalition. A runner or official benchmark environment without a Runner.Worker ancestor fails closed. A new PPID-1 process in the job coalition makes the job red. A new PPID-1 process in another coalition also makes the job red unless it runs a trusted system binary. A child that still has a living parent is not checked. The binary and all parent directories must be root-owned, must not be writable by group or other, must not be symlinks, and must have no ACL. The only exception is a world-writable Cryptexes or Cryptexes/Incoming parent whose next child is sealed with `SF_NOUNLINK` and meets the other trust rules. The path must start with `/System/`, `/usr/libexec/`, or `/usr/sbin/`. Interpreters do not pass. The guard also blocks a defence-in-depth list of code hosts. That list is not complete. A launchd-submitted system binary that passes these rules can pass and is logged. Processes older than the job start are ignored. The operator must clear an old orphan.
- Raw launch APIs: The build, engine and command-line profiles of this repository deny `job-creation`. The tests prove that `launchctl submit` fails inside each of them. The runtime-worker profile of the pinned benchmarker does not deny `job-creation`. A launchd helper that exits before the post-timing check is not caught. The fix needs a change to the benchmarker (organizer). launchd checks `job-creation` on its side for the spawn API (Apple launchd source, `job_mig_spawn_internal`). Coverage of every current raw launch API is not verified. Such a job is covered by the code-host list and by the coalition rule for any binary that is not a trusted system binary.
- Page cache channel: Jobs on the box share the page cache. Where enabled, they also share file access times. A submission could leave a timing mark in which host files are cached. A later job could read it. Process cleanup and write denial do not reset it. Promoted entries are reviewed.
- DFlash depth report: The engine reports the depth it runs; the trusted check only bounds the committed tokens per step. The complete fix needs a trusted check in benchd. The trusted depth check covers the sealed pair only. This benchd does not report the counts of the other pair, so that pair is not checked. Full coverage needs per-pair counts from benchd. Every promoted entry gets a contract review within 12 hours of promotion. An entry that drafted deeper than it declared is reported to the organizers for removal. The review also checks the entry's diff for launchd and XPC job APIs (`launch_*`, `SMJob*`, `xpc_*`, `bootstrap_*`, a spawn of `launchctl`, or `NSTask`/`Process` running `launchctl`) and for any read of benchmarker private paths. A hit is reported to the organizers for removal.
- The account cache directory (`DARWIN_USER_CACHE_DIR`) stays shared between jobs of the account.
