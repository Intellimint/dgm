import asyncio
import os

def tool_info():
    return {
        "name": "bash",
        "description": """Run commands in a bash shell\n
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.\n
* You don't have access to the internet via this tool.\n
* You do have access to a mirror of common linux and python packages via apt and pip.\n
* State is persistent across command calls and discussions with the user.\n
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n
* Please avoid commands that may produce a very large amount of output.\n
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }

class BashSession:
    """A session of a bash shell."""
    def __init__(self):
        self._started = False
        self._process = None
        self._timed_out = False
        self._timeout = 120.0  # seconds
        self._sentinel = "<<exit>>"
        self._output_delay = 0.2  # seconds

    async def start(self):
        if self._started:
            return
        self._process = await asyncio.create_subprocess_shell(
            "/bin/bash --noprofile --norc -i",  # Added --noprofile --norc to suppress startup messages
            preexec_fn=os.setsid,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy()  # Ensures inheritance of the current environment
        )
        self._started = True

    def stop(self):
        if not self._started:
            return
        if self._process.returncode is None:
            self._process.terminate()
        self._process = None
        self._started = False

    async def run(self, command):
        if not self._started:
            raise ValueError("Session has not started.")
        if self._process.returncode is not None:
            raise ValueError(f"Bash has exited with returncode {self._process.returncode}")
        if self._timed_out:
            raise ValueError(
                f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
            )

        # Clear any leftover output before sending the command
        self._process.stdout._buffer.clear()
        self._process.stderr._buffer.clear()
        
        # Send command
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # Read output until sentinel
        try:
            output = ''
            start_time = asyncio.get_event_loop().time()
            found_sentinel = False
            while True:
                if asyncio.get_event_loop().time() - start_time > self._timeout:
                    self._timed_out = True
                    raise ValueError(
                        f"Timed out: bash has not returned in {self._timeout} seconds and must be restarted."
                    )
                await asyncio.sleep(self._output_delay)
                stdout_data = self._process.stdout._buffer.decode(errors='ignore')
                stderr_data = self._process.stderr._buffer.decode(errors='ignore')
                if self._sentinel in stdout_data:
                    found_sentinel = True
                    break
            # After breaking, sleep briefly and read any remaining output
            await asyncio.sleep(0.1)
            stdout_data = self._process.stdout._buffer.decode(errors='ignore')
            output_lines = stdout_data.split('\n')
            output = '\n'.join(line for line in output_lines 
                                 if not line.startswith('bash-') and 
                                 not line.startswith('\x1b') and
                                 self._sentinel not in line)
            # Clear buffers
            self._process.stdout._buffer.clear()
            self._process.stderr._buffer.clear()
            output = output.strip()
            error = filter_error(stderr_data)
            return output, error
        except Exception as e:
            self._timed_out = True
            raise ValueError(str(e))

def filter_error(error):
    # Filter out errors that we do not want to see
    filtered_lines = []
    i = 0
    error_lines = error.splitlines()
    while i < len(error_lines):
        line = error_lines[i]

        # Skip shell startup messages and ioctl errors
        if any(msg in line for msg in [
            "zsh is now the default shell",
            "Inappropriate ioctl for device",
            "Welcome to",
            "Last login:",
            "bash-",
            "\x1b[?1034h",
            "<<exit>>",
            "The default interactive shell is now zsh",
            "To update your account to use zsh",
            "For more details, please visit",
            "Error:"
        ]):
            i += 1
            continue

        # Skip the next lines if ioctl error, add relevant lines
        if "Inappropriate ioctl for device" in line:
            i += 3
            if '<<exit>>' in error_lines[i]:
                i += 1
            while i < len(error_lines) - 1:
                filtered_lines.append(error_lines[i])
                i += 1
            i += 1
            continue

        filtered_lines.append(line)
        i += 1
    return '\n'.join(filtered_lines).strip()

async def tool_function_call(command):
    """Execute a command in the bash shell."""
    try:
        bash_session = BashSession()

        if not bash_session._started:
            await bash_session.start()

        output, error = await bash_session.run(command)
        error = filter_error(error)
        result = output.strip()
        # Only append error if it's non-empty, not a substring of output, and does not start with '>'
        if error and error not in result and not error.lstrip().startswith('>'):
            result += "\nError:\n" + error
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def tool_function(command):
    return asyncio.run(tool_function_call(command))

if __name__ == "__main__":
    # Example usage
    import sys

    # Check if the script is called with arguments
    if len(sys.argv) < 2:
        print("Usage: python bash.py '<command>'")
    else:
        # Extract the command from the command-line arguments
        input_command = ' '.join(sys.argv[1:])
        # Run the tool_function asynchronously
        result = tool_function(input_command)
        print(result)