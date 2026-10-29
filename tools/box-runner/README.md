# Box runner

These scripts provide an optional launchd deployment for the self-hosted
GitHub Actions runner used by this track. They mint one just-in-time runner
registration, run one job, reap its process tree, and repeat.

The ranked workflow can also use another runner deployment. This directory
supports only the JIT launchd deployment described here.

## Files

| File | Purpose |
| --- | --- |
| `supervisor.sh` | Mint, spawn, reap, and report status. |
| `gh-app-mint.sh` | Mint short-lived GitHub App credentials and JIT config. |
| `lib/pidtree.sh` | Track and stop a live process tree. |
| `install.sh` | Install the control plane and render the launchd plist. |
| `box-runner.env.example` | Configuration template. |
| `com.qwen38tf.box-runner.plist.template` | launchd template. |
| `test/test-box-runner.sh` | Offline tests with mock processes. |

## Process-tree rule

`su -` starts a new login session. A process-group signal sent to the
supervisor does not necessarily reach the runner session or its descendants.

The supervisor writes the job root PID, process start time, and command to
`state/job.pid` before it waits. The `stop` command reads that state, walks the
live parent-PID tree, checks process identity again, and signals each process.
The `run` command performs the same cleanup before it mints another runner.

This method works while the tracked root process is alive. It cannot find a
fully detached child after its root has already exited and its parent link is
gone. The implementation does not use a command-line substring sweep because
that could stop an unrelated process.

## JIT config limitation

The supervisor passes JIT config to the listener through
`ACTIONS_RUNNER_INPUT_JITCONFIG`. This keeps it out of the command line, but job
steps run under the same user as the listener. On macOS, same-user code can
inspect another process's initial environment.

Use `test/test-jit-config-peer-read.sh` to test this boundary. The default mode
reports exposure. `--strict` fails when any access mode reads the canary.

Possible deployment changes are:

1. Run job steps under a different user from the listener.
2. Reduce the credential lifetime and permissions further.
3. Move the listener and job execution to separate trust boundaries.

No production validation of those changes is included here.

## Configuration

`supervisor.sh` sources `box-runner.env` after reading its process environment.
Values in the config file therefore override exported values. If
`BOX_RUNNER_CONFIG` is set, it selects another config file.

Required values:

| Variable | Meaning |
| --- | --- |
| `RUNNER_DIR` | Path to the installed actions-runner. |
| `RUNNER_NAME` | Stable GitHub runner name. It must equal the ranked calibration box name. |
| `RUNNER_LABELS` | Labels that match `.github/workflows/benchmark.yml`. Use `macOS,qwen3.8-27b-tf-mlx-v1`; `self-hosted` is added by `gh-app-mint.sh`. |
| `GH_APP_ID` | GitHub App ID. |
| `GH_APP_KEY` | Path to the App private key. |
| `GH_RUNNER_REPO` | Repository in `owner/repo` form. |

Optional values are listed in `box-runner.env.example`.

`gh-app-mint.sh` can also source `BOX_RUNNER_APP_ENV`. Values in that file
override exported App values. Missing required values cause a refusal.

## Install boundary

Run `install.sh` as root for a real deployment. It installs root-owned copies
of the control plane, the config, and the launchd plist. It does not load the
plist.

The installer checks each existing path component. It refuses symlinks,
non-root ownership, group or other write access, and macOS ACLs. It prints the
SHA-256 digest of each installed file.

The runner directory must not exist before installation. The unprivileged
runner user creates it and extracts the runner there. Root does not recursively
change its ownership.

These variables are for tests only:

- `BOX_RUNNER_INSTALL_ROOT`
- `BOX_RUNNER_INSTALL_OWNER_UID`
- `BOX_RUNNER_RUN_AS_USER`
- `RUNNER_SU_ENABLED=0`
- `BOX_RUNNER_MAX_ITERATIONS`

## Security properties

- Required values fail closed.
- The private key path is configured, but key data is not stored here.
- Tokens do not appear in command arguments.
- The installation token is scoped to one repository and is revoked on exit.
- The JIT config file has mode `0600` and is removed on exit.
- Each JIT registration accepts one job.
- Any existing same-name runner stops minting. The helper never deletes a runner.
- The supervisor runs one job process at a time.
- The supervisor reaps the current tree before the next registration.
- These scripts do not calculate or seal a benchmark score.

## Tests

```bash
tools/box-runner/test/test-box-runner.sh [-v]
```

The suite requires at least 80 assertions. It uses mock processes and does not
use the network, credentials, or a real actions-runner. Root-only checks are
optional through `BOX_RUNNER_TEST_USER`.

The suite checks:

1. Direct process-tree cleanup, including a detached child.
2. A negative control that leaves the child alive when only the root PID stops.
3. End-to-end `run`, `stop`, and `status` behavior.
4. Fail-closed configuration checks.
5. Installer checksum helpers.
6. Trusted paths, installed modes, extraction ownership, and refusal cases.

## Operate

```bash
sudo tools/box-runner/install.sh
sudo vi /usr/local/etc/qwen38tf-box-runner/box-runner.env
sudo launchctl bootstrap system /Library/LaunchDaemons/com.qwen38tf.box-runner.plist
sudo /usr/local/libexec/qwen38tf-box-runner/supervisor.sh status
sudo /usr/local/libexec/qwen38tf-box-runner/supervisor.sh stop
```

To stop the service, stop its tracked job tree as well:

```bash
sudo launchctl bootout system/com.qwen38tf.box-runner
sudo /usr/local/libexec/qwen38tf-box-runner/supervisor.sh stop
sudo /usr/local/libexec/qwen38tf-box-runner/supervisor.sh status
```

## Validation limits

The offline suite validates the scripts with mock processes. It does not prove:

1. A live runner download and extraction.
2. The root-to-runner-user privilege drop on a provisioned host.
3. GitHub App token minting, scoping, and revocation against GitHub.
4. launchd recovery after a real crash.
5. A ranked dispatch with the configured labels.

Those deployment checks are not verified by this repository test suite.
