#!/usr/bin/env python3
"""Run one command on a Taiji pod through its SSH port.

This is a small operational fallback for cases where ``jizhi_client exec`` can
list a worker pod but hangs after selecting it. The script intentionally reads
the password from an environment variable so launch scripts and docs do not
need to store credentials.
"""

from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "usage: taiji_ssh_exec.py <host> <port> <user> <command...>",
            file=sys.stderr,
        )
        return 2

    host, port, user = sys.argv[1:4]
    remote_command = " ".join(sys.argv[4:])
    password = os.environ.get("TAIJI_SSH_PASSWORD")
    if not password:
        print("TAIJI_SSH_PASSWORD is required", file=sys.stderr)
        return 2

    master_fd, slave_fd = pty.openpty()
    argv = [
        "ssh",
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        f"{user}@{host}",
        remote_command,
    ]
    proc = subprocess.Popen(
        argv,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    sent_password = False
    output = bytearray()
    deadline = time.monotonic() + float(os.environ.get("TAIJI_SSH_TIMEOUT", "60"))

    try:
        while proc.poll() is None:
            if time.monotonic() > deadline:
                proc.kill()
                print("taiji_ssh_exec.py: timeout", file=sys.stderr)
                return 124
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            lowered = output[-4096:].lower()
            if not sent_password and b"assword:" in lowered:
                os.write(master_fd, (password + "\n").encode())
                sent_password = True

        # Drain a little trailing output after process exit.
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            ready, _, _ = select.select([master_fd], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
