import os
import signal
import subprocess
from typing import Optional


class ExecutionResult:
    def __init__(self, command: str, stdout: str, stderr: str, return_code: int, pid: int = None) -> None:
        self.command = str(command)
        self.stdout = str(stdout).strip()
        self.stderr = str(stderr).strip()
        self.return_code = return_code
        self.pid = pid

    def success(self) -> bool:
        return self.return_code == 0

    def __repr__(self):
        return '%s(%r)' % (self.__class__, self.__dict__)


class ExecutionError(Exception):
    def __init__(self,
                 msg: str,
                 command: str = None,
                 pid: int = None,
                 stdout: str = None,
                 stderr: str = None,
                 return_code: int = None) -> None:
        self.msg = str(msg)
        self.command_string = str(command)
        self.pid = pid
        self.stdout = str(stdout).strip()
        self.stderr = str(stderr).strip()
        self.return_code = return_code

    def __repr__(self):
        return '%s(%r)' % (self.__class__, self.__dict__)

    def __str__(self):
        return 'command string:{}; stdout:{}; stderr:{}; return code:{};'.format(self.command_string, self.stdout,
                                                                                 self.stderr, self.return_code)


def execute_command(command: str, directory: Optional[str] = None, strict: bool = True) -> ExecutionResult:
    cwd = os.getcwd()

    try:
        if directory is not None:
            os.chdir(directory)

        result = _execute_command(command, ignore_signals=True)
        if strict and result.return_code != 0:
            raise ExecutionError(
                msg=result.stderr.strip(),
                command=result.command,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.return_code,
            )

        return result

    except OSError as e:
        raise ExecutionError(msg=str(e))

    finally:
        os.chdir(cwd)


def _execute_command(command_string: str, ignore_signals=False) -> ExecutionResult:
    """ Execute command in shell """
    stderr = stdout = ''

    def preexec_function():
        if ignore_signals:
            # Ignore the SIGINT and SIGUSR2 signals by setting the handler to the standard signal handler SIG_IGN.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)

    try:
        # universal_newlines converts output from byte to string
        with subprocess.Popen(command_string,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              preexec_fn=preexec_function) as process:
            stdout_bytes, stderr_bytes = process.communicate()
            # Attempt to decode to UTF-8, ignore malformed data
            stdout = stdout_bytes.decode(errors='ignore')
            stderr = stderr_bytes.decode(errors='ignore')

            return ExecutionResult(
                command=command_string,
                pid=process.pid,
                return_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )

    except subprocess.SubprocessError as e:
        raise ExecutionError(msg=str(e), command=command_string, stdout=stdout, stderr=stderr)
