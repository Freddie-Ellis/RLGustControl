"""Mirrors the tunnel PC's shared folder with the learner's copy on the HPC, using Windows' built-in OpenSSH `sftp`.

    uv run python -m src.ml.distributed.sync [session]      (runs on the tunnel PC; session defaults to SESSION)

A session's two folders are local_shared(session) and hpc_shared(session) in files.py: the same name under a
base folder on each machine, so the learner job and this sync can only meet in the same place.

Start order: submit the learner job on the HPC (it writes config.json), start this, then start the actor once
config.json has arrived locally. The actor only ever touches LOCAL, so a slow or broken link never stalls the control
loop: episodes queue in the local outbox and go up when the link is back. The local outbox doubles as a backup.

    tunnel PC  LOCAL/outbox/ep_*              --put-->   HPC  REMOTE/outbox/
               LOCAL/weights/policy_v*.pt     <--get--        REMOTE/weights/   (the newest only)
               LOCAL/config.json              <--get--        REMOTE/config.json
               LOCAL/learner_run.txt          <--get--        REMOTE/learner_run.txt
               runs_dir/<run>/                <--get--        the learner's run directory (metrics, status, policy)
               LOCAL/STOP                     <-both->        REMOTE/STOP
    (LOCAL = local_shared(session), REMOTE = hpc_shared(session))

STOP goes both ways, and so does clearing it: a STOP that was on both sides and is then deleted here (the Session
tab's "Clear STOP") is deleted on the HPC too, so the session can carry on. A STOP that is only on the HPC when the
sync starts is one left from the last time the session was stopped (whatever it stopped has exited by now), so it is
deleted as well: starting the sync means the session is meant to run.

Everything except learner_state.pt (only the learner needs it) ends up identical on both sides, so the Session tab
and the run list work on the tunnel PC as if the learner were local. learner_run.txt still holds the HPC path;
`files.learner_run_dir` finds the copy in runs_dir by name.

Every pass is one `sftp` connection to list the remote folders and, if anything is due, one more to transfer it.
Uploads go to `<name>.tmp` on the HPC and are renamed there, metadata before transitions, so the learner never reads
a half-uploaded episode. Downloads are renamed locally the same way. Restart-safe: what is missing is worked out from
the listing each pass, so a restart (of this, the actor or the learner job) just carries on.

It finishes by itself once STOP is on both sides, nothing is left to send, and the learner's status says it is done
(or STOP_LINGER_S has passed, in case its job died). Before it does, it fetches the learner's whole run directory
once more: a file is otherwise only fetched again when its size or its modification minute changes, which a final
policy.pt written in the same minute as the last fetch does not. Launched from the Session tab's "Launch sync" too.

After every pass it writes LOCAL/sync_status.json (when the HPC was last reached, or why not), which the Session
tab shows.

Needs key-based SSH login to HOST with no passphrase prompt (`sftp -b` cannot answer prompts), and the tunnel PC
must be able to reach the login node (on campus, or through the university VPN).
"""

import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

from src.config import CONFIG
from src.ml.atomic import replace, tmp_path
from src.ml.distributed.files import (
    SESSION,
    SyncStatus,
    config_path,
    hpc_shared,
    learner_run_dir,
    learner_run_path,
    local_shared,
    outbox_dir,
    prune_weights,
    read_config,
    stop_path,
    weights_dir,
    weights_version,
    write_sync_status,
)
from src.ml.run_config import DistributedConfig
from src.ml.runs import RunDir

# ----- settings for running this file directly -----
HOST = "iridis"  # an alias in ~/.ssh/config (HostName = a login node you may use, User, IdentityFile)

SFTP = "sftp"  # C:\Windows\System32\OpenSSH\sftp.exe on the tunnel PC
CONNECT_TIMEOUT_S = 15
LIST_TIMEOUT_S = 60.0
TRANSFER_TIMEOUT_S = 600.0  # a whole pass, generous for a backlog of episodes after the link was down
STOP_LINGER_S = 120.0  # after STOP, wait this long at most for the learner's final status before finishing

type Entries = dict[str, str]  # file name -> "size month day time" from `ls -l`: changes whenever the file does


class SyncError(OSError):
    """sftp failed or timed out, or a download could not be put in place. The pass is dropped and retried; nothing
    is lost."""


@dataclass(frozen=True)
class RemoteListing:
    root: Entries
    outbox: Entries
    weights: Entries
    run: Entries  # the learner's run directory; empty until learner_run.txt has come down


@dataclass(frozen=True)
class PassResult:
    summary: str | None  # one line for the log, None if there was nothing to report
    idle: bool  # nothing was transferred


@dataclass(frozen=True)
class Download:
    remote: str
    local: Path
    signature: str  # remembered once it is down, so an unchanged file is not fetched again


@dataclass
class Plan:
    """What one pass will do: sftp commands to run, then the downloads to rename into place once all succeeded."""

    commands: list[str] = field(default_factory=list)
    downloads: list[Download] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


def _q(path: str) -> str:
    """Quote a path for an sftp batch file. sftp treats backslashes as escapes, so local paths use forward slashes."""
    return f'"{path}"'


def parse_listing(out: str, n_sections: int) -> list[Entries]:
    """Split batch-mode stdout into one Entries per `ls -l`. sftp echoes each command as `sftp> ...`.

    An `ls -l` line is `perms links user group size month day time|year name`. A folder that does not exist yet (the
    learner has not created it) just gives an empty section.
    """
    sections: list[Entries] = []
    for line in out.splitlines():
        if line.startswith("sftp>"):
            sections.append({})
        elif sections and len(parts := line.split()) >= 9 and parts[0].startswith("-"):  # regular files only
            name = " ".join(parts[8:]).rsplit("/", 1)[-1]  # some sftp versions print the full path
            sections[-1][name] = " ".join(parts[4:8])
    return (sections + [{} for _ in range(n_sections)])[:n_sections]


class Sync:
    def __init__(self, local: Path, host: str, remote: PurePosixPath, runs_dir: Path = CONFIG.runs_dir) -> None:
        self.local = local
        self.host = host
        self.remote = remote
        self.runs_dir = runs_dir
        self.fetched: dict[str, str] = {}  # remote path -> signature of the copy we have
        self._stop_on_both = False  # STOP has been seen here and on the HPC: if ours then goes, it was cleared here
        self._first_listing = True
        self._last_ok: datetime | None = None  # the last pass that reached the HPC
        outbox_dir(local).mkdir(parents=True, exist_ok=True)  # the actor writes here before any weights exist
        weights_dir(local).mkdir(parents=True, exist_ok=True)

    def remote_path(self, local_path: Path) -> str:
        """The HPC copy of a file in LOCAL. The layout lives in files.py only; this just re-roots it."""
        return (self.remote / local_path.relative_to(self.local).as_posix()).as_posix()

    def remote_run(self) -> PurePosixPath | None:
        """The learner's run directory on the HPC, from the learner_run.txt already brought down."""
        path = learner_run_path(self.local)
        return PurePosixPath(path.read_text().strip().replace("\\", "/")) if path.exists() else None

    # ----- sftp -----
    def sftp(self, commands: list[str], timeout: float) -> str:
        """Run commands in one batch-mode connection and return stdout. A command prefixed `-` may fail harmlessly."""
        args = [SFTP, "-b", "-", "-o", f"ConnectTimeout={CONNECT_TIMEOUT_S}", self.host]
        try:
            done = subprocess.run(
                args, input="\n".join(commands) + "\n", capture_output=True, text=True, timeout=timeout, check=False
            )  # the return code is checked below, to raise SyncError rather than CalledProcessError
        except subprocess.TimeoutExpired as e:
            raise SyncError(f"sftp timed out after {timeout:.0f} s") from e
        except FileNotFoundError as e:
            raise SystemExit(f"{SFTP!r} not found; install Windows' OpenSSH client") from e
        if done.returncode != 0:
            raise SyncError(f"sftp exit {done.returncode}: {done.stderr.strip()}")
        return done.stdout

    def list_remote(self) -> RemoteListing:
        dirs = [self.remote_path(d) for d in (self.local, outbox_dir(self.local), weights_dir(self.local))]
        if (run := self.remote_run()) is not None:
            dirs.append(run.as_posix())
        out = self.sftp([f"-ls -l {_q(d)}" for d in dirs], LIST_TIMEOUT_S)
        root, outbox, weights, run_entries = parse_listing(out, 4)
        return RemoteListing(root, outbox, weights, run_entries)

    # ----- one pass -----
    def plan(self, listing: RemoteListing) -> Plan:
        """Work out what is missing on each side. Pure apart from creating a local STOP the HPC asked for."""
        plan = Plan()
        for local_path in (config_path(self.local), learner_run_path(self.local)):
            signature = listing.root.get(local_path.name)
            remote = self.remote_path(local_path)
            if signature is not None and (self.fetched.get(remote) != signature or not local_path.exists()):
                self._get(plan, remote, local_path, signature, report=True)

        # Only the newest weights: the actor never loads an older version, and after the link was down there may
        # be hundreds it has missed.
        versions = [name for name in listing.weights if weights_version(name) is not None]
        if versions:
            newest = max(versions, key=lambda name: weights_version(name) or 0)
            local_path = weights_dir(self.local) / newest
            if not local_path.exists():
                self._get(plan, self.remote_path(local_path), local_path, listing.weights[newest], report=True)

        if (run := self.remote_run()) is not None:
            local_run = self.runs_dir / run.name
            local_run.mkdir(parents=True, exist_ok=True)
            for name, signature in sorted(listing.run.items()):
                if self.fetched.get(f"{run}/{name}") != signature:  # changes every log row: not worth reporting
                    self._get(plan, f"{run}/{name}", local_run / name, signature, report=False)

        stop = stop_path(self.local)
        here, there = stop.exists(), stop.name in listing.root
        if self._first_listing and there and not here:
            self._stop_on_both = True  # left over from before this sync started: treat as cleared here
        self._first_listing = False
        if here and there:
            self._stop_on_both = True
        elif there and self._stop_on_both:
            # It was on both sides and ours has gone: cleared here, so clear it there. The flag stays set until a
            # listing shows the HPC's is gone too, so a failed pass tries again instead of bringing STOP back.
            plan.commands.append(f"-rm {_q(self.remote_path(stop))}")
            plan.summary.append("STOP cleared on HPC")
        elif there:
            stop.touch()  # the learner was stopped on the HPC: stop the actor after its current episode
            self._stop_on_both = True
            plan.summary.append("STOP from HPC")
        elif here:
            plan.commands.append(f"put {_q(stop.as_posix())} {_q(self.remote_path(stop))}")
            plan.summary.append("STOP to HPC")
        else:
            self._stop_on_both = False

        unsent = [p for p in sorted(outbox_dir(self.local).glob("ep_*.parquet")) if p.name not in listing.outbox]
        for parquet in unsent:
            for f in (parquet.with_suffix(".json"), parquet):  # metadata first: the parquet marks the episode complete
                remote, remote_tmp = self.remote_path(f), self.remote_path(tmp_path(f))
                plan.commands += [
                    f"put {_q(f.as_posix())} {_q(remote_tmp)}",
                    f"-rm {_q(remote)}",  # left by an interrupted pass; plain sftp rename won't overwrite
                    f"rename {_q(remote_tmp)} {_q(remote)}",
                ]
        if unsent:
            plan.summary.append(f"{len(unsent)} episode(s) up ({unsent[0].stem}..{unsent[-1].stem})")
        return plan

    @staticmethod
    def _get(plan: Plan, remote: str, local_path: Path, signature: str, report: bool) -> None:
        plan.commands.append(f"get {_q(remote)} {_q(tmp_path(local_path).as_posix())}")
        plan.downloads.append(Download(remote, local_path, signature))
        if report:
            plan.summary.append(f"{local_path.name} down")

    def run_pass(self) -> PassResult:
        """List, then transfer whatever is due."""
        listing = self.list_remote()
        if config_path(self.local).name not in listing.root:
            return PassResult("waiting for the learner to write config.json on the HPC", idle=True)
        plan = self.plan(listing)
        if plan.commands:
            self.sftp(plan.commands, TRANSFER_TIMEOUT_S)
            busy = [d.local.name for d in plan.downloads if not self._put_in_place(d)]
            if busy:  # not remembered as fetched, so the next pass downloads them again
                raise SyncError(f"could not replace {', '.join(busy)} (open in another program); next pass retries")
            prune_weights(self.local)  # the learner prunes its own copy; this keeps ours from growing all session
        return PassResult(", ".join(plan.summary) or None, idle=not plan.commands)

    def _put_in_place(self, d: Download) -> bool:
        """Rename a finished download over the local file. On Windows that fails while anything has the file open
        (the Session or Compare tab reading metrics.csv, say), so `replace` tries a few times. False if it stayed
        busy."""
        try:
            replace(tmp_path(d.local), d.local)
        except PermissionError:
            return False
        self.fetched[d.remote] = d.signature
        return True

    def forget_run(self) -> None:
        """Forget what has been fetched of the learner's run directory, so the next pass fetches all of it again."""
        if (run := self.remote_run()) is not None:
            self.fetched = {remote: sig for remote, sig in self.fetched.items() if not remote.startswith(f"{run}/")}

    def _report(self, error: str | None) -> None:
        """Write sync_status.json for the Session tab. Never worth failing a pass for."""
        now = datetime.now().astimezone()
        if error is None:
            self._last_ok = now
        try:
            write_sync_status(self.local, SyncStatus(last_attempt=now, last_ok=self._last_ok, error=error))
        except OSError:
            pass

    def learner_finished(self) -> bool:
        """The learner's status (from our copy of its run directory) says it is no longer running."""
        path = learner_run_dir(self.local, self.runs_dir)
        try:
            return path is not None and RunDir(path).read_status().state != "running"
        except (OSError, ValueError):  # not down yet, or caught mid-write
            return False

    def session_over(self, stop_seen: float | None) -> bool:
        """STOP is on both sides (after a successful pass a local STOP means the HPC has it too), nothing is left to
        transfer, and the learner has finished, or has had STOP_LINGER_S to do so (its job may have died)."""
        if stop_seen is None:
            return False
        return self.learner_finished() or time.monotonic() - stop_seen > STOP_LINGER_S

    def poll_s(self) -> float:
        """Seconds between passes: the session config's `distributed.sync_poll_s`, or its default until the learner
        has written config.json (the sync starts before it does)."""
        try:
            return read_config(self.local).distributed.sync_poll_s
        except (OSError, ValueError):
            return DistributedConfig().sync_poll_s

    def run(self, poll_s: float | None = None) -> None:
        """Pass after pass until the session is over (see `session_over`) or Ctrl+C. A failed pass is reported and
        retried; the actor keeps running regardless. `poll_s` overrides the config's interval (for tests)."""
        print(f"syncing {self.local} <-> {self.host}:{self.remote}", flush=True)
        last = ""
        stop_seen: float | None = None
        final_fetch_done = False
        while True:
            try:
                result = self.run_pass()
            except OSError as e:  # SyncError, or any file trouble here: a pass is never worth stopping the session for
                message = f"pass failed, retrying: {e}"
                print(message, file=sys.stderr, flush=True)
                self._report(str(e))
            else:
                self._report(None)
                message = result.summary or ""
                if message and message != last:  # don't repeat "waiting ..." every pass
                    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)
                if not stop_path(self.local).exists():
                    stop_seen = None  # STOP cleared: the session carries on
                    final_fetch_done = False
                elif stop_seen is None:
                    stop_seen = time.monotonic()
                if result.idle and self.session_over(stop_seen):
                    if not final_fetch_done:
                        # The learner has written its last files. Fetch its run directory once more, whatever the
                        # listing says: the final policy.pt can look unchanged (same size, same minute).
                        final_fetch_done = True
                        self.forget_run()
                        continue
                    print(f"{time.strftime('%H:%M:%S')} STOP on both sides and nothing left to send: sync done", flush=True)
                    return
            last = message
            time.sleep(poll_s if poll_s is not None else self.poll_s())


def run_sync(session: str, host: str = HOST) -> None:
    """Mirror the session's local folder with its HPC folder until the session is over, checking as often as the
    session's config says. Top level, so the UI can run it in a process."""
    try:
        Sync(local_shared(session), host, hpc_shared(session)).run()
    except KeyboardInterrupt:
        print("sync stopped", flush=True)


if __name__ == "__main__":
    run_sync(sys.argv[1] if len(sys.argv) > 1 else SESSION)
