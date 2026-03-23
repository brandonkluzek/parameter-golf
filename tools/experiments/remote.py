from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SSHConfig:
    host: str
    user: str | None = None
    port: int | None = None
    identity: str | None = None
    options: list[str] = field(default_factory=list)

    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def _base_ssh_args(config: SSHConfig) -> list[str]:
    args = ["ssh"]
    if config.port is not None:
        args += ["-p", str(config.port)]
    if config.identity:
        args += ["-i", config.identity]
    for option in config.options:
        args += ["-o", option]
    args.append(config.target())
    return args


def run_ssh(config: SSHConfig, remote_command: str, *, capture_output: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _base_ssh_args(config) + [remote_command],
        text=True,
        capture_output=capture_output,
        check=False,
    )


def scp_from(config: SSHConfig, remote_path: str, local_path: Path) -> subprocess.CompletedProcess[str]:
    args = ["scp"]
    if config.port is not None:
        args += ["-P", str(config.port)]
    if config.identity:
        args += ["-i", config.identity]
    for option in config.options:
        args += ["-o", option]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    args += [f"{config.target()}:{remote_path}", str(local_path)]
    return subprocess.run(args, text=True, capture_output=True, check=False)
