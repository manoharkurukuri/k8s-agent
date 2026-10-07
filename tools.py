import json
import shlex
import subprocess
import threading
from typing import Literal

from langchain.tools import tool


_COMMAND_LOCK = threading.Lock()
_TIMEOUT_SECONDS = 60
_OUTPUT_LIMIT = 20_000


def _command_result(
    command: str,
    status: Literal["success", "error", "denied", "timeout"],
    *,
    exit_code: int | None = None,
    stdout: str | bytes | None = "",
    stderr: str | bytes | None = "",
) -> dict[str, str | int | bool | None]:
    stdout = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout or ""
    stderr = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr or ""
    return {
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "stdout": stdout[:_OUTPUT_LIMIT],
        "stderr": stderr[:_OUTPUT_LIMIT],
        "stdout_truncated": len(stdout) > _OUTPUT_LIMIT,
        "stderr_truncated": len(stderr) > _OUTPUT_LIMIT,
    }


@tool
def run_kubectl(command: str) -> dict[str, str | int | bool | None]:
    """Run one kubectl command after explicit terminal confirmation.

    Provide a complete command, for example `kubectl get pods -n blog-app`.
    The user must approve each execution with 'yes' or 'y'. Commands run
    without a shell or interactive stdin and time out after 60 seconds.
    Returns status, exit code, stdout, stderr, and output truncation flags.
    Respect denied requests; do not retry without a new user request.
    """
    print(f"comma")
    with _COMMAND_LOCK:
        if any(ord(character) < 32 or ord(character) == 127 for character in command):
            error = "Command must be a single line without control characters."
            print(f"\nCommand rejected: {error}")
            return _command_result(command, "error", stderr=error)
        try:
            arguments = shlex.split(command)
        except ValueError as exc:
            print(f"\nCommand rejected: {exc}")
            return _command_result(command, "error", stderr=str(exc))
        if len(arguments) < 2 or arguments[0] != "kubectl":
            error = "Only commands beginning with 'kubectl' and a subcommand are allowed."
            print(f"\nCommand rejected: {error}")
            return _command_result(command, "error", stderr=error)

        print(f"\nProposed command: {json.dumps(shlex.join(arguments), ensure_ascii=True)}")
        print("Runs locally with your Kubernetes credentials; it may change cluster resources.")
        print("Output will be sent to the configured model. Do not approve commands exposing secrets.")
        print(f"Non-interactive execution; timeout: {_TIMEOUT_SECONDS} seconds.")
        try:
            approval = input("Run this command? Type 'yes' to approve [default: no]: ")
        except (EOFError, KeyboardInterrupt):
            approval = ""
        if approval.strip().lower() != "yes" and approval.strip().lower() != "y":
            print("\nCommand denied; nothing was executed.")
            return _command_result(command, "denied", stderr="User denied command execution.")

        try:
            completed = subprocess.run(
                arguments,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                check=False,
                timeout=_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            error = (
                f"kubectl timed out after {_TIMEOUT_SECONDS} seconds. "
                "Cluster-side changes or remote commands may already have occurred."
            )
            print(f"\nCommand timed out: {error}")
            return _command_result(
                command, "timeout", stdout=exc.stdout,
                stderr=f"{error}\n{exc.stderr.decode('utf-8', errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr or ''}",
            )
        except OSError as exc:
            print(f"\nCould not start kubectl: {exc}")
            return _command_result(command, "error", stderr=str(exc))

        print(f"\nCommand finished with exit code {completed.returncode}.")
        return _command_result(
            command,
            "success" if completed.returncode == 0 else "error",
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
