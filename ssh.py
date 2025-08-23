import shlex
import subprocess

from pydantic import BaseModel


class SSHConfig(BaseModel):
    user: str
    host: str
    port: int = 22


class SSHSession:
    def __init__(self, config: SSHConfig):
        self.config = config

    def run(self, command: str):
        ssh_command = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"{self.config.user}@{self.config.host}",
            "-p",
            str(self.config.port),
            "bash",
            "-lc",
            shlex.quote(command),
        ]

        return subprocess.run(ssh_command, check=True)
