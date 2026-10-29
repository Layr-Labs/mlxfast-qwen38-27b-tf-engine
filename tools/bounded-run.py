#!/usr/bin/env python3
"""Run a command with bounded private output and optional build cleanup."""

import argparse
import ctypes
import errno
import os
import signal
import subprocess
import sys
import threading
import time


def process_table():
    """Return same-UID process data from the trusted system ps."""
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-A", "-o", "pid=,ppid=,uid=,stat="],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    table = {}
    for line in output.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid, ppid, uid = (int(value) for value in fields[:3])
        except ValueError:
            continue
        if uid == os.getuid():
            table[pid] = (ppid, fields[3])
    return table


def group_has_live_processes(pgid):
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-A", "-o", "pgid=,stat="],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        try:
            os.killpg(pgid, 0)
        except OSError as exc:
            return exc.errno not in (errno.ESRCH,)
        return None
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == str(pgid) and not fields[1].startswith("Z"):
            return True
    return False


def stop_group(pgid):
    first_state = group_has_live_processes(pgid)
    if first_state is False:
        return True
    for wanted_signal, wait_seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, None)):
        try:
            os.killpg(pgid, wanted_signal)
        except ProcessLookupError:
            return True
        if wait_seconds is None and first_state is None:
            wait_seconds = 2.0
        deadline = None if wait_seconds is None else time.monotonic() + wait_seconds
        while deadline is None or time.monotonic() < deadline:
            state = group_has_live_processes(pgid)
            if state is False:
                return True
            if state is None and deadline is None:
                deadline = time.monotonic() + 2.0
            if wanted_signal == signal.SIGKILL:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    return True
            time.sleep(0.05)
    return False


class _BsdInfo(ctypes.Structure):
    """struct proc_bsdinfo (sys/proc_info.h), for PROC_PIDTBSDINFO."""
    _fields_ = (
        [(name, ctypes.c_uint32) for name in (
            "flags", "status", "xstatus", "pid", "ppid", "uid", "gid",
            "ruid", "rgid", "svuid", "svgid", "rfu_1")]
        + [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
        + [(name, ctypes.c_uint32) for name in ("nfiles", "pgid", "pjobc", "e_tdev", "e_tpgid")]
        + [("nice", ctypes.c_int32), ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]
    )


try:
    _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
except OSError:
    _LIBPROC = None


def kernel_info(pid):
    """Return ((pid, start sec, start usec), ppid, uid) from ONE kernel read, or None if the PID is gone."""
    if _LIBPROC is None:
        return None
    info = _BsdInfo()
    size = _LIBPROC.proc_pidinfo(pid, 3, ctypes.c_uint64(0), ctypes.byref(info), ctypes.sizeof(info))
    if size != ctypes.sizeof(info) or info.pid != pid:
        return None
    return ((pid, info.start_sec, info.start_usec), info.ppid, info.uid)


def start_identity(pid):
    """Return (pid, start seconds, start microseconds) from the kernel, or None if the PID is gone."""
    info = kernel_info(pid)
    return None if info is None else info[0]


class DescendantTracker:
    """Best-effort tracker for same-UID descendants that change sessions."""

    def __init__(self, root_pid):
        self.root_pid = root_pid
        self.seen = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def capture(self):
        snapshot_time = time.time()
        table = process_table()
        if table is None:
            return None
        descendants = set()
        changed = True
        while changed:
            changed = False
            parents = descendants | {self.root_pid}
            for pid, (ppid, stat) in table.items():
                if pid not in descendants and ppid in parents and not stat.startswith("Z"):
                    descendants.add(pid)
                    changed = True
        with self.lock:
            # Drop exited or reused PIDs first, then record each current descendant by its kernel
            # start time, so a reused PID is never signalled and a new descendant is never lost.
            for pid, identity in list(self.seen.items()):
                if start_identity(pid) != identity:
                    del self.seen[pid]
            # Adopt a ps descendant only if the kernel shows our UID and a start time before the ps
            # snapshot: that is the process ps saw. A PID reused after the snapshot starts later and is
            # never adopted; a real descendant whose parent exits in between keeps its start time.
            for pid in descendants:
                info = kernel_info(pid)
                if info is None or info[2] != os.getuid():
                    continue
                _pid, start_sec, start_usec = info[0]
                if start_sec + start_usec / 1e6 < snapshot_time:
                    self.seen[pid] = info[0]
        return table

    def live(self):
        table = self.capture()
        if table is None:
            return None
        with self.lock:
            seen = dict(self.seen)
        return {
            pid: identity for pid, identity in seen.items()
            if pid in table and not table[pid][1].startswith("Z") and start_identity(pid) == identity
        }

    def _run(self):
        while not self.stop_event.is_set():
            self.capture()
            self.stop_event.wait(0.02)

    def start(self):
        self.capture()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1.0)


def stop_descendants(tracker):
    """Stop tracked escapees, but do not claim full process containment."""
    for wanted_signal, wait_seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            live = tracker.live()
            if live is None:
                return True
            if not live:
                return True
            for pid, identity in live.items():
                # Re-check the identity right before the signal; skip a PID that changed.
                if start_identity(pid) != identity:
                    continue
                try:
                    os.kill(pid, wanted_signal)
                except ProcessLookupError:
                    pass
            time.sleep(0.05)
    live = tracker.live()
    return live is None or not live


def stop_process(process):
    """Stop a non-session command after supervisor cancellation."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
        return
    except subprocess.TimeoutExpired:
        process.kill()
    process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    # --stderr PATH: keep the command's stderr in its own private file, so --output holds stdout only.
    parser.add_argument("--stderr")
    parser.add_argument("--limit", type=int, default=1024 * 1024)
    parser.add_argument("--session", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.limit < 1:
        return 126

    # Install the cancel handlers before the command starts. A signal that
    # arrives before or during Popen is recorded and forwarded after Popen
    # returns, so the supervisor never dies with a live build group.
    state = {"process": None, "cancel": None}
    old_handlers = {}

    def forward(signum):
        process = state["process"]
        if process is None:
            return
        try:
            if args.session:
                os.killpg(process.pid, signum)
            elif process.poll() is None:
                process.send_signal(signum)
        except (ProcessLookupError, PermissionError):
            pass

    def cancel(signum, _frame):
        if state["cancel"] is None:
            state["cancel"] = signum
        forward(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, cancel)

    def restore_handlers():
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)

    def private_file(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "wb")

    output = private_file(args.output)
    error_output = private_file(args.stderr) if args.stderr else None

    def close_outputs():
        output.close()
        if error_output is not None:
            error_output.close()

    if state["cancel"] is not None:
        close_outputs()
        restore_handlers()
        return 128 + state["cancel"]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if error_output is not None else subprocess.STDOUT,
            start_new_session=args.session,
        )
    except OSError:
        close_outputs()
        restore_handlers()
        return 126
    state["process"] = process
    if state["cancel"] is not None:
        forward(state["cancel"])

    # Build mode: stdout and stderr share one pipe and one byte limit. --stderr mode: each stream
    # has its own limit, and a dropped stdout byte fails the run, so a cut result never passes.
    stdout_budget = {"left": args.limit, "dropped": 0}
    stderr_budget = {"left": args.limit, "dropped": 0}
    remaining_lock = threading.Lock()

    def drain(stream, sink, budget):
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                with remaining_lock:
                    kept = chunk[:budget["left"]]
                    budget["left"] -= len(kept)
                    budget["dropped"] += len(chunk) - len(kept)
                if kept:
                    sink.write(kept)
            sink.flush()
        except (OSError, ValueError):
            pass

    readers = [threading.Thread(target=drain, args=(process.stdout, output, stdout_budget), daemon=True)]
    if error_output is not None:
        readers.append(threading.Thread(target=drain, args=(process.stderr, error_output, stderr_budget), daemon=True))
    tracker = DescendantTracker(process.pid)
    return_code = None
    clean = True
    try:
        for reader in readers:
            reader.start()
        tracker.start()
        while state["cancel"] is None:
            try:
                return_code = process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if args.session:
            clean = stop_group(process.pid)
        elif process.poll() is None:
            stop_process(process)
        if process.poll() is None:
            process.wait()
        clean = stop_descendants(tracker) and clean
        tracker.stop()
        final_live = tracker.live()
        clean = (final_live is None or not final_live) and clean

    for reader in readers:
        reader.join(timeout=5.0)
        if reader.is_alive():
            clean = False
    for stream in (process.stdout, process.stderr):
        if stream:
            stream.close()
    for reader in readers:
        reader.join(timeout=1.0)
    close_outputs()
    restore_handlers()
    if not clean:
        return 125
    if error_output is not None and stdout_budget["dropped"]:
        return 125
    if state["cancel"] is not None:
        return 128 + state["cancel"]
    if return_code is None:
        return 125
    return return_code if return_code >= 0 else 128 - return_code


if __name__ == "__main__":
    sys.exit(main())
