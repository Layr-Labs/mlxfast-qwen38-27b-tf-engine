#!/usr/bin/env bash
set -u
export LC_ALL=C

exec /usr/bin/python3 -I - "$@" <<'PY'
import ctypes
import errno
import os
import re
import signal
import stat
import subprocess
import sys
import time

NAME = "orphan-guard.sh"
PS = "/bin/ps"
LS = "/bin/ls"
PROC_PIDTBSDINFO = 3
PROC_PIDTBSDINFO_SIZE = 136
PROC_PIDCOALITIONINFO = 20
PROC_PIDCOALITIONINFO_SIZE = 40
PROC_PIDPATHINFO_MAXSIZE = 4096
SYSTEM_PREFIXES = ("/System/", "/usr/libexec/", "/usr/sbin/")
CRYPTEX_WRITABLE_PARENTS = {
    "/System/Volumes/Preboot/Cryptexes",
    "/System/Volumes/Preboot/Cryptexes/Incoming",
}
SF_NOUNLINK = 0x00100000
INTERPRETERS = {
    "sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish",
    "osascript", "jsc", "expect", "env", "xargs", "nohup", "script",
    "launchctl",
}
INTERPRETER_PREFIXES = (
    "python", "perl", "ruby", "tclsh", "wish", "node", "php", "lua")
# Defence in depth only. This list is not complete.
CODE_HOSTS = {
    "httpd", "rpcsvchost", "sshd", "sshd-session", "cron", "atrun",
    "xpcproxy", "smbd",
}


class GuardError(Exception):
    pass


class RunnerRootError(GuardError):
    pass


class CoalitionError(GuardError):
    pass


class BsdInfo(ctypes.Structure):
    # sys/proc_info.h: struct proc_bsdinfo. Keep the field order and sizes exact.
    _fields_ = (
        [(name, ctypes.c_uint32) for name in (
            "flags", "status", "xstatus", "pid", "ppid", "uid", "gid",
            "ruid", "rgid", "svuid", "svgid", "rfu_1")]
        + [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
        + [(name, ctypes.c_uint32) for name in (
            "nfiles", "pgid", "pjobc", "e_tdev", "e_tpgid")]
        + [("nice", ctypes.c_int32),
           ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]
    )


class CoalitionInfo(ctypes.Structure):
    _fields_ = (
        ("coalition_id", ctypes.c_uint64 * 2),
        ("reserved", ctypes.c_uint64 * 3),
    )


try:
    LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    LIBPROC.proc_pidinfo.argtypes = (
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int)
    LIBPROC.proc_pidinfo.restype = ctypes.c_int
    LIBPROC.proc_pidpath.argtypes = (
        ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
    LIBPROC.proc_pidpath.restype = ctypes.c_int
except OSError:
    LIBPROC = None


def fail_usage():
    print(f"{NAME}: ERROR: usage", file=sys.stderr)
    raise SystemExit(2)


def epoch(text):
    try:
        value = float(text)
    except ValueError as err:
        raise GuardError("bad epoch") from err
    if value < 0:
        raise GuardError("bad epoch")
    return value


def run_ps(argv):
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            [PS, *argv], check=False, capture_output=True, text=True, env=env
        )
    except OSError as err:
        raise GuardError("process table unavailable") from err
    if result.returncode != 0:
        raise GuardError("process table unavailable")
    return result.stdout


def bsd_info(pid):
    if LIBPROC is None or ctypes.sizeof(BsdInfo) != PROC_PIDTBSDINFO_SIZE:
        raise GuardError("libproc unavailable")
    info = BsdInfo()
    ctypes.set_errno(0)
    size = LIBPROC.proc_pidinfo(
        pid, PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info) or info.pid != pid:
        error = ctypes.get_errno()
        if error == errno.ESRCH:
            raise ProcessLookupError(error, os.strerror(error))
        if error:
            raise OSError(error, os.strerror(error))
        raise GuardError("process identity unavailable")
    return info


def resource_coalition(pid):
    if LIBPROC is None or ctypes.sizeof(CoalitionInfo) != PROC_PIDCOALITIONINFO_SIZE:
        return None
    info = CoalitionInfo()
    size = LIBPROC.proc_pidinfo(
        pid, PROC_PIDCOALITIONINFO, 0, ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info):
        return None
    return int(info.coalition_id[0])


def kernel_identity(pid):
    """Read pid, start, uid, ppid and executable path from the kernel."""
    try:
        first = bsd_info(pid)
    except ProcessLookupError:
        raise
    except OSError as err:
        raise GuardError("process identity unavailable") from err
    if first is None:
        return None
    if first.uid != os.getuid():
        return None
    path_buffer = ctypes.create_string_buffer(PROC_PIDPATHINFO_MAXSIZE)
    size = LIBPROC.proc_pidpath(pid, path_buffer, len(path_buffer))
    if size <= 0:
        try:
            second = bsd_info(pid)
        except ProcessLookupError:
            raise
        except OSError as err:
            raise GuardError("process identity unavailable") from err
        if second is None:
            return None
        path = None
    else:
        try:
            second = bsd_info(pid)
        except ProcessLookupError:
            raise
        except OSError as err:
            raise GuardError("process identity unavailable") from err
        if second is None:
            return None
        try:
            path = os.fsdecode(path_buffer.value)
        except UnicodeError as err:
            raise GuardError("bad process path") from err
        if not os.path.isabs(path):
            raise GuardError("bad process path")
    first_key = (first.pid, first.start_sec, first.start_usec, first.uid)
    second_key = (second.pid, second.start_sec, second.start_usec, second.uid)
    if first_key != second_key:
        raise GuardError("process identity changed")
    return {
        "pid": pid,
        "ppid": int(second.ppid),
        "uid": int(second.uid),
        "start_sec": int(second.start_sec),
        "start_usec": int(second.start_usec),
        "start": second.start_sec + second.start_usec / 1000000.0,
        "path": path,
    }


def identity_key(proc):
    return (proc["start_sec"], proc["start_usec"], proc["uid"], proc["path"])


def process_gone(pid, existence_checker=os.kill):
    try:
        existence_checker(pid, 0)
    except ProcessLookupError:
        return True
    except OSError as err:
        return err.errno == errno.ESRCH
    return False


def snapshot(ps_reader=run_ps, identity_reader=kernel_identity,
             existence_checker=os.kill):
    rows = {}
    own_uid = os.getuid()
    output = ps_reader(["-A", "-o", "pid=,uid=,stat="])
    for line in output.splitlines():
        parts = line.split(None, 2)
        if len(parts) != 3:
            raise GuardError("bad process table row")
        try:
            pid = int(parts[0])
            uid = int(parts[1])
        except ValueError as err:
            raise GuardError("bad process table row") from err
        # A zombie holds no CPU, GPU or memory and cannot be signalled.
        if parts[2].startswith("Z"):
            continue
        try:
            proc = identity_reader(pid)
        except ProcessLookupError:
            continue
        except (GuardError, OSError) as err:
            if uid != own_uid:
                continue
            if process_gone(pid, existence_checker):
                continue
            raise GuardError("process identity unavailable") from err
        if proc is None and uid == own_uid and not process_gone(
                pid, existence_checker):
            raise GuardError("process identity unavailable")
        if proc is not None:
            rows[pid] = proc
    return rows


def has_acl(path):
    try:
        result = subprocess.run(
            [LS, "-led", path], check=False, capture_output=True, text=True
        )
    except OSError:
        return True
    if result.returncode != 0 or result.stderr:
        return True
    return len(result.stdout.splitlines()) > 1


def interpreter(path):
    name = os.path.basename(path)
    return (name in INTERPRETERS or name in CODE_HOSTS
            or name.startswith(INTERPRETER_PREFIXES))


def trusted_system_path(path, stat_reader=os.lstat, acl_reader=has_acl):
    if (not any(path.startswith(prefix) for prefix in SYSTEM_PREFIXES)
            or os.path.normpath(path) != path or interpreter(path)):
        return False
    current = path
    is_file = True
    child_info = None
    child_has_acl = None
    while True:
        try:
            info = stat_reader(current)
        except OSError:
            return False
        wanted_type = stat.S_ISREG if is_file else stat.S_ISDIR
        current_has_acl = acl_reader(current)
        if (not wanted_type(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != 0 or current_has_acl):
            return False
        if stat.S_IMODE(info.st_mode) & 0o022:
            writable_bits = stat.S_IMODE(info.st_mode) & 0o022
            child_is_sealed = (
                current in CRYPTEX_WRITABLE_PARENTS
                and writable_bits & 0o002
                and child_info is not None
                and child_info.st_uid == 0
                and not (stat.S_IMODE(child_info.st_mode) & 0o022)
                and not child_has_acl
                and getattr(child_info, "st_flags", 0) & SF_NOUNLINK
            )
            if not child_is_sealed:
                return False
        if current == "/":
            return True
        child_info = info
        child_has_acl = current_has_acl
        current = os.path.dirname(current)
        is_file = False


def system_agent_passes(proc, job_coalition_value,
                        coalition_reader=resource_coalition,
                        stat_reader=os.lstat, acl_reader=has_acl,
                        logged=None, log_pass=True, trust_cache=None):
    if proc["ppid"] != 1:
        return False
    coalition = coalition_reader(proc["pid"])
    if coalition is None or coalition == job_coalition_value:
        return False
    path = proc["path"]
    if path is None:
        return False
    if trust_cache is None:
        trusted = trusted_system_path(path, stat_reader, acl_reader)
    else:
        if path not in trust_cache:
            trust_cache[path] = trusted_system_path(path, stat_reader, acl_reader)
        trusted = trust_cache[path]
    if not trusted:
        return False
    key = (proc["pid"], proc["start_sec"], proc["start_usec"], path, coalition)
    if log_pass and (logged is None or key not in logged):
        print(
            f"{NAME}: system agent passed: pid={proc['pid']} path={path} "
            f"coalition={coalition} start={proc['start_sec']}.{proc['start_usec']:06d}",
            file=sys.stderr,
        )
        if logged is not None:
            logged.add(key)
    return True


def orphaned(proc, job_coalition_value, **kwargs):
    return not system_agent_passes(proc, job_coalition_value, **kwargs)


def allowed(proc, allow):
    wanted = allow.get(proc["pid"])
    return proc["path"] is not None and wanted is not None and proc["path"] == wanted


def parse_check(argv):
    job_start = None
    allow = {}
    index = 0
    while index < len(argv):
        if argv[index] == "--job-start" and index + 1 < len(argv):
            job_start = epoch(argv[index + 1])
            index += 2
        elif argv[index] == "--allow-pid" and index + 1 < len(argv):
            raw = argv[index + 1]
            pid_text, separator, path = raw.partition(":")
            if not separator or not pid_text.isdigit() or not os.path.isabs(path):
                fail_usage()
            allow[int(pid_text)] = os.path.realpath(path)
            index += 2
        else:
            fail_usage()
    if job_start is None:
        fail_usage()
    return job_start, allow


def check(argv):
    job_start, allow = parse_check(argv)
    job_coalition_value = job_coalition()
    uid = os.getuid()
    found = []
    logged = set()
    trust_cache = {}
    for proc in snapshot().values():
        if proc["uid"] != uid or proc["ppid"] != 1 or proc["start"] < job_start:
            continue
        if allowed(proc, allow) or not orphaned(
                proc, job_coalition_value, logged=logged,
                trust_cache=trust_cache):
            continue
        found.append(proc["pid"])
    if found:
        print(
            f"{NAME}: FAIL: orphan-process ({len(found)} process(es))",
            file=sys.stderr,
        )
        return 1
    return 0


def runner_root_from_path(path):
    if path is None:
        return None
    if os.path.basename(path) not in ("Runner.Listener", "Runner.Worker"):
        return None
    bin_name = os.path.basename(os.path.dirname(path))
    if bin_name != "bin" and not re.fullmatch(r"bin\.[0-9]+(?:\.[0-9]+)*", bin_name):
        return None
    return os.path.realpath(os.path.dirname(os.path.dirname(path)))


def runner_ancestor(required=True):
    pid = os.getpid()
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            proc = kernel_identity(pid)
        except ProcessLookupError:
            proc = None
        if proc is None:
            raise RunnerRootError("runner root refused: ancestor unavailable")
        if proc["path"] is None:
            raise RunnerRootError("runner root refused: ancestor path unavailable")
        if os.path.basename(proc["path"]) == "Runner.Worker":
            root = runner_root_from_path(proc["path"])
            if root is None:
                raise RunnerRootError(
                    "runner root refused: Runner.Worker path has unsupported shape")
            return proc, root
        pid = proc["ppid"]
    if required:
        raise RunnerRootError("runner root refused: Runner.Worker ancestor not found")
    return None, None


def verified_job_coalition(worker=None, coalition_reader=resource_coalition,
                           runner_reader=runner_ancestor, env=None):
    own = coalition_reader(os.getpid())
    if own is None:
        raise CoalitionError("coalition read failed for orphan guard")
    if worker is None:
        worker, _runner_root = runner_reader(required=False)
    if worker is None:
        env = os.environ if env is None else env
        if ("RUNNER_ENVIRONMENT" in env or "GITHUB_ACTIONS" in env
                or env.get("MLXFAST_OFFICIAL_BENCHMARK_RUN") == "1"):
            raise CoalitionError(
                "Runner.Worker ancestor not found in runner or official "
                "benchmark environment")
        return own
    worker_coalition = coalition_reader(worker["pid"])
    if worker_coalition is None:
        raise CoalitionError(
            f"coalition read failed for Runner.Worker pid={worker['pid']}")
    if own != worker_coalition:
        raise CoalitionError(
            f"orphan guard coalition {own} does not match "
            f"Runner.Worker coalition {worker_coalition}")
    return worker_coalition


def job_coalition():
    return verified_job_coalition()


def job_start():
    proc, _runner_root = runner_ancestor()
    verified_job_coalition(proc)
    print(f"{proc['start']:.6f}")
    return 0


def parse_kill(argv):
    job_start_value = None
    runner_root = None
    index = 0
    while index < len(argv):
        if argv[index] == "--job-start" and index + 1 < len(argv):
            job_start_value = epoch(argv[index + 1])
            index += 2
        elif argv[index] == "--runner-root" and index + 1 < len(argv):
            runner_root = argv[index + 1]
            index += 2
        else:
            fail_usage()
    if job_start_value is None:
        fail_usage()
    _worker, derived_root = runner_ancestor()
    if runner_root is not None:
        if not os.path.isabs(runner_root):
            fail_usage()
        runner_root = os.path.realpath(runner_root)
        if runner_root != derived_root:
            raise RunnerRootError(
                f"runner root mismatch: derived {derived_root}, got {runner_root}")
    return job_start_value, derived_root


def ancestry(rows, pid):
    result = []
    seen = set()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        proc = rows.get(pid)
        if proc is None:
            break
        result.append(proc)
        pid = proc["ppid"]
    return result


def under(path, root):
    if path is None:
        return False
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def runner_installation(path):
    if path is None:
        return None
    parts = path.split(os.sep)
    for index, part in enumerate(parts):
        if part.startswith("actions-runner"):
            return os.sep.join(parts[:index + 1]) or os.sep
    return None


def other_runner(rows, proc, runner_root):
    for parent in ancestry(rows, proc["pid"]):
        path = parent["path"]
        if path is None:
            continue
        if os.path.basename(path) in ("Runner.Listener", "Runner.Worker"):
            installation = runner_root_from_path(path)
            return installation is None or installation != runner_root
        installation = runner_installation(path)
        if installation and os.path.realpath(installation) != runner_root:
            return True
    return False


def this_runner(rows, proc, runner_root):
    for parent in ancestry(rows, proc["pid"])[1:]:
        if runner_root_from_path(parent["path"]) == runner_root:
            return True
    return False


def staging_roots():
    roots = [os.path.realpath(os.getcwd())]
    for name in ("MLXFAST_BASELINE_WORKSPACE", "MLXFAST_REFERENCE_DIR"):
        value = os.environ.get(name, "")
        if value:
            roots.append(os.path.realpath(value))
    return roots


def other_staging(proc, roots):
    return (
        proc["path"] is not None
        and os.path.basename(proc["path"]) == "bench-worker"
        and not any(under(proc["path"], root) for root in roots)
    )


def kill_excluded(rows, proc, runner_root, roots, job_coalition_value,
                  logged=None, classifier_kwargs=None):
    classifier_kwargs = {} if classifier_kwargs is None else classifier_kwargs
    coalition = proc.get("coalition")
    if "coalition" not in proc:
        coalition = classifier_kwargs.get(
            "coalition_reader", resource_coalition)(proc["pid"])
    if coalition == job_coalition_value:
        return False
    fixed_classifier_kwargs = dict(classifier_kwargs)
    fixed_classifier_kwargs["coalition_reader"] = lambda _pid: coalition
    return (
        system_agent_passes(
            proc, job_coalition_value, logged=logged,
            **fixed_classifier_kwargs)
        or other_runner(rows, proc, runner_root)
        or other_staging(proc, roots)
    )


def kill_candidates(rows, job_start_value, runner_root, roots,
                    job_coalition_value, logged=None, classifier_kwargs=None):
    classifier_kwargs = {} if classifier_kwargs is None else classifier_kwargs
    uid = os.getuid()
    own = {proc["pid"] for proc in ancestry(rows, os.getpid())}
    result = {}
    for proc in rows.values():
        if proc["uid"] != uid or proc["start"] < job_start_value:
            continue
        if proc["pid"] in own:
            continue
        proc = dict(proc)
        proc["coalition"] = classifier_kwargs.get(
            "coalition_reader", resource_coalition)(proc["pid"])
        if kill_excluded(
                rows, proc, runner_root, roots, job_coalition_value, logged,
                classifier_kwargs):
            continue
        # Kill only escaped orphans and processes inside this runner's own tree,
        # never another process of the account (on a shared or desktop account).
        if proc["ppid"] != 1 and not this_runner(rows, proc, runner_root):
            continue
        result[proc["pid"]] = proc
    return result


def send_one(selected, sig, runner_root, roots, job_coalition_value, rows=None,
             identity_reader=kernel_identity, killer=os.kill, logged=None,
             classifier_kwargs=None):
    current_rows = snapshot() if rows is None else rows
    # This is the last kernel read before the signal.
    try:
        current = identity_reader(selected["pid"])
    except ProcessLookupError:
        return False
    if current is None:
        return False
    current = dict(current)
    coalition_reader = (classifier_kwargs or {}).get(
        "coalition_reader", resource_coalition)
    current["coalition"] = coalition_reader(current["pid"])
    current_rows[current["pid"]] = current
    if kill_excluded(current_rows, current, runner_root, roots,
                     job_coalition_value, logged, classifier_kwargs):
        return False
    if selected["path"] is None or current["path"] is None:
        if selected["path"] is not None or current["path"] is not None:
            return False
        if (selected.get("coalition") != job_coalition_value
                or current["coalition"] != selected["coalition"]):
            return False
        selected_key = (
            selected["pid"], selected["start_sec"], selected["start_usec"],
            selected["uid"])
        current_key = (
            current["pid"], current["start_sec"], current["start_usec"],
            current["uid"])
        if current_key != selected_key:
            return False
    elif identity_key(current) != identity_key(selected):
        return False
    try:
        killer(current["pid"], sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def send(targets, sig, runner_root, roots, job_coalition_value, logged,
         classifier_kwargs=None):
    sent = 0
    for selected in targets.values():
        # Fresh path, start time and uid read immediately before every signal.
        if send_one(selected, sig, runner_root, roots,
                    job_coalition_value, logged=logged,
                    classifier_kwargs=classifier_kwargs):
            sent += 1
    return sent


def kill_mode(argv):
    job_start_value, runner_root = parse_kill(argv)
    job_coalition_value = job_coalition()
    roots = staging_roots()
    logged = set()
    classifier_kwargs = {"trust_cache": {}}
    rows = snapshot()
    first = kill_candidates(
        rows, job_start_value, runner_root, roots, job_coalition_value, logged,
        classifier_kwargs)
    term_count = send(
        first, signal.SIGTERM, runner_root, roots, job_coalition_value, logged,
        classifier_kwargs)
    time.sleep(0.4)

    rows = snapshot()
    second = kill_candidates(
        rows, job_start_value, runner_root, roots, job_coalition_value, logged,
        classifier_kwargs)
    new = {pid: proc for pid, proc in second.items() if pid not in first}
    term_count += send(
        new, signal.SIGTERM, runner_root, roots, job_coalition_value, logged,
        classifier_kwargs)
    time.sleep(0.4)

    rows = snapshot()
    survivors = kill_candidates(
        rows, job_start_value, runner_root, roots, job_coalition_value, logged,
        classifier_kwargs)
    kill_count = send(
        survivors, signal.SIGKILL, runner_root, roots,
        job_coalition_value, logged, classifier_kwargs)
    deadline = time.monotonic() + 3.0
    while True:
        rows = snapshot()
        remaining = kill_candidates(
            rows, job_start_value, runner_root, roots,
            job_coalition_value, logged, classifier_kwargs)
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    print(
        f"{NAME}: kill: term={term_count} kill={kill_count} remain={len(remaining)}"
    )
    return 1 if remaining else 0


def elapsed_seconds(text):
    parts = text.strip().split("-")
    days = int(parts[0]) if len(parts) == 2 else 0
    clock = parts[-1].split(":")
    if len(clock) == 2:
        hours, minutes, seconds = 0, int(clock[0]), int(clock[1])
    elif len(clock) == 3:
        hours, minutes, seconds = map(int, clock)
    else:
        raise GuardError("bad elapsed time")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def recheck_self_test():
    own = kernel_identity(os.getpid())
    if own is None:
        raise GuardError("self identity unavailable")
    selected = dict(own)
    selected.update({"pid": 4242, "ppid": 1, "path": "/tmp/orphan"})
    called = []
    job_coalition_value = 11
    outside_coalition = lambda _pid: 22
    same_coalition = lambda _pid: job_coalition_value
    no_coalition = lambda _pid: None
    system_path = "/System/Library/TestSupport/safe-agent"
    interpreter_path = "/System/Library/python3"
    usr_bin_path = "/usr/bin/safe-agent"
    code_host_path = "/usr/sbin/httpd"
    regular_files = {
        system_path, interpreter_path, usr_bin_path, code_host_path,
    }

    class FakeStat:
        def __init__(self, mode, uid=0, flags=0):
            self.st_mode = mode
            self.st_uid = uid
            self.st_flags = flags

    def fake_root_stat(path):
        file_type = stat.S_IFREG if path in regular_files else stat.S_IFDIR
        return FakeStat(file_type | 0o755)

    unit_cases = (
        ("coalition read failure", dict(selected, path=system_path), no_coalition),
        ("pathless outside coalition", dict(selected, path=None), outside_coalition),
        ("system interpreter", dict(selected, path=interpreter_path),
         outside_coalition),
        ("usr bin", dict(selected, path=usr_bin_path), outside_coalition),
        ("code host", dict(selected, path=code_host_path), outside_coalition),
        ("in-coalition system path", dict(selected, path=system_path), same_coalition),
    )
    for label, proc, coalition_reader in unit_cases:
        if not orphaned(
                proc, job_coalition_value, coalition_reader=coalition_reader,
                stat_reader=fake_root_stat, acl_reader=lambda _path: False,
                log_pass=False):
            raise GuardError(f"{label} was not flagged")
    trusted = dict(selected, path=system_path)
    if orphaned(
            trusted, job_coalition_value, coalition_reader=outside_coalition,
            stat_reader=fake_root_stat, acl_reader=lambda _path: False,
            log_pass=False):
        raise GuardError("trusted system path was flagged")
    trust_cache = {}
    stat_calls = []

    def counting_stat(path):
        stat_calls.append(path)
        return fake_root_stat(path)

    for _index in range(2):
        if not system_agent_passes(
                trusted, job_coalition_value,
                coalition_reader=outside_coalition, stat_reader=counting_stat,
                acl_reader=lambda _path: False, log_pass=False,
                trust_cache=trust_cache):
            raise GuardError("cached trusted system path was flagged")
        if _index == 0:
            first_call_count = len(stat_calls)
    if len(stat_calls) != first_call_count:
        raise GuardError("trusted system path cache was not reused")

    cryptex_root = "/System/Volumes/Preboot/Cryptexes"
    cryptex_child = f"{cryptex_root}/OS"
    cryptex_path = f"{cryptex_child}/System/Library/TestSupport/safe-agent"

    def cryptex_stat(child_uid=0, child_flags=0):
        def reader(path):
            if path == cryptex_path:
                return FakeStat(stat.S_IFREG | 0o755)
            if path == cryptex_root:
                return FakeStat(stat.S_IFDIR | 0o777)
            if path == cryptex_child:
                return FakeStat(
                    stat.S_IFDIR | 0o755, child_uid, child_flags)
            return FakeStat(stat.S_IFDIR | 0o755)
        return reader

    cryptex_proc = dict(selected, path=cryptex_path)
    cryptex_cases = (
        ("user-owned cryptex child", cryptex_stat(child_uid=501, child_flags=SF_NOUNLINK), True),
        ("cryptex child without SF_NOUNLINK", cryptex_stat(), True),
        ("sealed cryptex child", cryptex_stat(child_flags=SF_NOUNLINK), False),
    )
    for label, stat_reader, expected_orphan in cryptex_cases:
        actual = orphaned(
            cryptex_proc, job_coalition_value,
            coalition_reader=outside_coalition, stat_reader=stat_reader,
            acl_reader=lambda _path: False, log_pass=False)
        if actual != expected_orphan:
            raise GuardError(f"{label} classification was wrong")

    owned_path = "/System/Library/TestSupport/owned-agent"

    def owner_stat(file_uid=0, dir_uid=0):
        def reader(path):
            if path == owned_path:
                return FakeStat(stat.S_IFREG | 0o755, file_uid)
            if path == os.path.dirname(owned_path):
                return FakeStat(stat.S_IFDIR | 0o755, dir_uid)
            return FakeStat(stat.S_IFDIR | 0o755)
        return reader

    owned_proc = dict(selected, path=owned_path)
    for label, stat_reader, expected_orphan in (
            ("user-owned system-path file", owner_stat(file_uid=501), True),
            ("user-owned system-path parent", owner_stat(dir_uid=501), True),
            ("root-owned system-path file", owner_stat(), False)):
        actual = orphaned(
            owned_proc, job_coalition_value,
            coalition_reader=outside_coalition, stat_reader=stat_reader,
            acl_reader=lambda _path: False, log_pass=False)
        if actual != expected_orphan:
            raise GuardError(f"{label} classification was wrong")

    for official_env in (
            {"RUNNER_ENVIRONMENT": ""}, {"GITHUB_ACTIONS": ""},
            {"MLXFAST_OFFICIAL_BENCHMARK_RUN": "1"}):
        try:
            verified_job_coalition(
                coalition_reader=lambda _pid: job_coalition_value,
                runner_reader=lambda required=False: (None, None),
                env=official_env)
        except CoalitionError as err:
            if "Runner.Worker ancestor not found" not in str(err):
                raise GuardError("official no-runner error was unclear") from err
        else:
            raise GuardError("official environment accepted no Runner.Worker")
    if verified_job_coalition(
            coalition_reader=lambda _pid: job_coalition_value,
            runner_reader=lambda required=False: (None, None), env={}
    ) != job_coalition_value:
        raise GuardError("local no-runner coalition fallback failed")

    def one_ps_row(uid):
        return lambda _argv: f"4242 {uid} S\n"

    def unreadable(_pid):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES))

    try:
        snapshot(
            ps_reader=one_ps_row(os.getuid()), identity_reader=unreadable,
            existence_checker=lambda _pid, _sig: None)
    except GuardError:
        pass
    else:
        raise GuardError("same-uid EACCES was skipped")

    def gone(_pid):
        raise ProcessLookupError(errno.ESRCH, os.strerror(errno.ESRCH))

    if snapshot(ps_reader=one_ps_row(os.getuid()), identity_reader=gone):
        raise GuardError("same-uid ESRCH was retained")
    if snapshot(
            ps_reader=one_ps_row(os.getuid() + 1),
            identity_reader=unreadable,
            existence_checker=lambda _pid, _sig: (_ for _ in ()).throw(
                GuardError("other uid was probed"))):
        raise GuardError("other-uid EACCES was retained")

    runner_named = dict(
        selected, path="/tmp/other/.zig-cache/Runner.Worker")
    same_kwargs = {"coalition_reader": same_coalition}
    outside_kwargs = {"coalition_reader": outside_coalition}
    runner_rows = {4242: runner_named}
    if kill_excluded(
            runner_rows, runner_named, "/tmp/current", ["/tmp/current"],
            job_coalition_value, classifier_kwargs=same_kwargs):
        raise GuardError("in-coalition runner-named helper was excluded")
    if 4242 not in kill_candidates(
            runner_rows, 0, "/tmp/current", ["/tmp/current"],
            job_coalition_value, classifier_kwargs=same_kwargs):
        raise GuardError("in-coalition runner-named helper was not selected")
    if not kill_excluded(
            runner_rows, runner_named, "/tmp/current", ["/tmp/current"],
            job_coalition_value, classifier_kwargs=outside_kwargs):
        raise GuardError("outside runner-named helper was not excluded")
    if kill_candidates(
            runner_rows, 0, "/tmp/current", ["/tmp/current"],
            job_coalition_value, classifier_kwargs=outside_kwargs):
        raise GuardError("outside runner-named helper was selected")

    allowed_now = trusted
    classifier_kwargs = {
        "coalition_reader": outside_coalition,
        "stat_reader": fake_root_stat,
        "acl_reader": lambda _path: False,
        "log_pass": False,
    }
    send_one(allowed_now, signal.SIGTERM, "/tmp/current", ["/tmp/current"],
             job_coalition_value,
             rows={4242: allowed_now}, identity_reader=lambda _pid: allowed_now,
             killer=lambda pid, sig: called.append((pid, sig)),
             classifier_kwargs=classifier_kwargs)
    if called:
        raise GuardError("passing system agent was signalled")
    changed = dict(selected, start_sec=selected["start_sec"] + 1)
    send_one(selected, signal.SIGTERM, "/tmp/current", ["/tmp/current"],
             job_coalition_value,
             rows={4242: changed}, identity_reader=lambda _pid: changed,
             killer=lambda pid, sig: called.append((pid, sig)))
    other_runner_now = dict(
        selected,
        path="/opt/actions-runner-other/_work/repo/helper")
    send_one(other_runner_now, signal.SIGTERM,
             "/opt/actions-runner-current", ["/tmp/current"],
             job_coalition_value,
             rows={4242: other_runner_now},
             identity_reader=lambda _pid: other_runner_now,
             killer=lambda pid, sig: called.append((pid, sig)))
    other_staging_now = dict(
        selected, path="/opt/other-track/build.noindex/bench-worker")
    send_one(other_staging_now, signal.SIGTERM, "/tmp/current", ["/tmp/current"],
             job_coalition_value,
             rows={4242: other_staging_now},
             identity_reader=lambda _pid: other_staging_now,
             killer=lambda pid, sig: called.append((pid, sig)))
    pathless = dict(selected, path=None)
    pathless = kill_candidates(
        {4242: pathless}, 0, "/tmp/current", ["/tmp/current"],
        job_coalition_value, classifier_kwargs=same_kwargs).get(4242)
    if pathless is None:
        raise GuardError("in-coalition pathless orphan was not retained")
    send_one(
        pathless, signal.SIGTERM, "/tmp/current", ["/tmp/current"],
        job_coalition_value, rows={4242: pathless},
        identity_reader=lambda _pid: pathless,
        killer=lambda pid, sig: called.append((pid, sig)),
        classifier_kwargs=same_kwargs)
    if called != [(4242, signal.SIGTERM)]:
        raise GuardError("in-coalition pathless orphan was not signalled")
    called.clear()
    changed_pathless = dict(
        pathless, start_sec=pathless["start_sec"] + 1)
    send_one(
        pathless, signal.SIGTERM, "/tmp/current", ["/tmp/current"],
        job_coalition_value, rows={4242: changed_pathless},
        identity_reader=lambda _pid: changed_pathless,
        killer=lambda pid, sig: called.append((pid, sig)),
        classifier_kwargs=same_kwargs)
    outside_pathless = kill_candidates(
        {4242: dict(selected, path=None)}, 0, "/tmp/current",
        ["/tmp/current"], job_coalition_value,
        classifier_kwargs=outside_kwargs).get(4242)
    if outside_pathless is None:
        raise GuardError("outside pathless orphan was not retained")
    send_one(
        outside_pathless, signal.SIGTERM, "/tmp/current", ["/tmp/current"],
        job_coalition_value, rows={4242: outside_pathless},
        identity_reader=lambda _pid: outside_pathless,
        killer=lambda pid, sig: called.append((pid, sig)),
        classifier_kwargs=outside_kwargs)
    if called:
        raise GuardError("unsafe pathless identity was signalled")
    return 0


def self_test():
    own = kernel_identity(os.getpid())
    if own is None:
        raise GuardError("self identity unavailable")
    etime = run_ps(["-o", "etime=", "-p", str(os.getpid())])
    if abs((time.time() - own["start"]) - elapsed_seconds(etime)) > 2:
        raise GuardError("kernel start mismatch")
    recheck_self_test()
    print(f"{NAME}: self-test: kernel-start identity-recheck ok")
    return 0


def main():
    if len(sys.argv) < 2:
        fail_usage()
    mode = sys.argv[1]
    argv = sys.argv[2:]
    if mode == "check":
        return check(argv)
    if mode == "kill":
        return kill_mode(argv)
    if mode == "job-start" and not argv:
        return job_start()
    if mode == "self-test" and not argv:
        return self_test()
    if mode == "recheck-self-test" and not argv:
        result = recheck_self_test()
        print(f"{NAME}: self-test: identity-recheck ok")
        return result
    fail_usage()


try:
    raise SystemExit(main())
except RunnerRootError as err:
    print(f"{NAME}: ERROR: {err}", file=sys.stderr)
    raise SystemExit(2)
except CoalitionError as err:
    print(f"{NAME}: ERROR: {err}", file=sys.stderr)
    raise SystemExit(2)
except GuardError:
    print(f"{NAME}: ERROR: process-table", file=sys.stderr)
    raise SystemExit(2)
PY
