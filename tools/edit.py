from pathlib import Path
import subprocess
from typing import Optional, Tuple, List, Dict
import re

def tool_info():
    return {
        "name": "editor",
        "description": """Custom editing tool for viewing, creating, and editing files\n
* State is persistent across command calls and discussions with the user.\n
* If `path` is a file, `view` displays the entire file with line numbers. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.\n
* The `create` command cannot be used if the specified `path` already exists as a file.\n
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`.\n
* The `edit` command can overwrite the entire file or specific line ranges with the provided `file_text`.\n
* Partial/line-range edits and viewing are supported using `start_line` and `end_line` parameters.\n
* The `test_regex` command validates regex patterns against test strings and shows detailed matching results.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "edit", "test_regex"],
                    "description": "The command to run: `view`, `create`, `edit`, or `test_regex`."
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`. Not required for test_regex command.",
                    "type": "string"
                },
                "file_text": {
                    "description": "Required parameter of `create` or `edit` command, containing the content for the entire file or line range.",
                    "type": "string"
                },
                "start_line": {
                    "description": "Optional 1-based line number to start viewing/editing from.",
                    "type": "integer",
                    "minimum": 1
                },
                "end_line": {
                    "description": "Optional 1-based line number to end viewing/editing at (inclusive).",
                    "type": "integer",
                    "minimum": 1
                },
                "pattern": {
                    "description": "Required parameter for test_regex command. The regex pattern to test.",
                    "type": "string"
                },
                "test_strings": {
                    "description": "Required parameter for test_regex command. List of strings to test against the pattern.",
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "flags": {
                    "description": "Optional parameter for test_regex command. Regex flags to use (e.g. 'i' for case-insensitive).",
                    "type": "string"
                }
            },
            "required": ["command"]
        }
    }

def maybe_truncate(content: str, max_length: int = 10000) -> str:
    """Truncate long content and add marker."""
    if len(content) > max_length:
        return content[:max_length] + "\n<response clipped>"
    return content

def validate_path(path: str, command: str) -> Path:
    """
    Validate the file path for each command:
      - 'view': path may be a file or directory; must exist.
      - 'create': path must not exist (for new file creation).
      - 'edit': path must exist (for overwriting).
    """
    path_obj = Path(path)

    # Check if it's an absolute path
    if not path_obj.is_absolute():
        raise ValueError(
            f"The path {path} is not an absolute path (must start with '/')."
        )

    if command == "view":
        # Path must exist
        if not path_obj.exists():
            raise ValueError(f"The path {path} does not exist.")
    elif command == "create":
        # Path must not exist
        if path_obj.exists():
            raise ValueError(f"Cannot create new file; {path} already exists.")
    elif command == "edit":
        # Path must exist and must be a file
        if not path_obj.exists():
            raise ValueError(f"The file {path} does not exist.")
        if path_obj.is_dir():
            raise ValueError(f"{path} is a directory and cannot be edited as a file.")
    else:
        raise ValueError(f"Unknown or unsupported command: {command}")

    return path_obj

def validate_line_range(start_line: Optional[int], end_line: Optional[int], total_lines: int) -> Tuple[int, int]:
    """Validate and normalize line range parameters."""
    if start_line is None and end_line is None:
        return 1, total_lines
    
    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = total_lines
        
    if start_line < 1:
        raise ValueError("start_line must be >= 1")
    if end_line > total_lines:
        raise ValueError(f"end_line must be <= {total_lines}")
    if start_line > end_line:
        raise ValueError("start_line must be <= end_line")
        
    return start_line, end_line

def format_output(content: str, path: str, start_line: int = 1, end_line: Optional[int] = None) -> str:
    """Format output with line numbers (for file content)."""
    content = maybe_truncate(content)
    content = content.expandtabs()
    lines = content.split("\n")
    
    if end_line is None:
        end_line = len(lines)
        
    numbered_lines = [
        f"{i + start_line:6}\t{line}"
        for i, line in enumerate(lines[start_line-1:end_line])
    ]
    return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered_lines) + "\n"

def read_file(path: Path, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read and return file contents, optionally for a specific line range."""
    try:
        content = path.read_text()
        if start_line is None and end_line is None:
            return content
            
        lines = content.split("\n")
        start_line, end_line = validate_line_range(start_line, end_line, len(lines))
        return "\n".join(lines[start_line-1:end_line])
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

def write_file(path: Path, content: str, start_line: Optional[int] = None, end_line: Optional[int] = None):
    """Write file contents, optionally for a specific line range."""
    try:
        if start_line is None and end_line is None:
            path.write_text(content)
            return
            
        # Read existing content
        existing_lines = path.read_text().split("\n")
        total_lines = len(existing_lines)
        start_line, end_line = validate_line_range(start_line, end_line, total_lines)
        
        # Split new content into lines
        new_lines = content.split("\n")
        
        # Replace the specified range
        existing_lines[start_line-1:end_line] = new_lines
        
        # Write back the entire file
        path.write_text("\n".join(existing_lines))
    except Exception as e:
        raise ValueError(f"Failed to write file: {e}")

def view_path(path_obj: Path, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """View file contents or directory listing."""
    if path_obj.is_dir():
        # For directories: list non-hidden files up to 2 levels deep
        try:
            result = subprocess.run(
                ['find', str(path_obj), '-maxdepth', '2', '-not', '-path', '*/\\.*'],
                capture_output=True,
                text=True
            )
            if result.stderr:
                return f"Error listing directory: {result.stderr}"
            return (
                f"Here's the files and directories up to 2 levels deep in {path_obj}, excluding hidden items:\n"
                + result.stdout
            )
        except Exception as e:
            raise ValueError(f"Failed to list directory: {e}")

    # If it's a file, show the file with line numbers
    content = read_file(path_obj, start_line, end_line)
    return format_output(content, str(path_obj), start_line or 1, end_line)

def test_regex(pattern: str, test_strings: List[str], flags: str = "") -> str:
    """Test a regex pattern against a list of test strings and return detailed results."""
    try:
        # Parse flags
        flag_value = 0
        for flag in flags:
            if flag == 'i':
                flag_value |= re.IGNORECASE
            elif flag == 'm':
                flag_value |= re.MULTILINE
            elif flag == 's':
                flag_value |= re.DOTALL
            elif flag == 'x':
                flag_value |= re.VERBOSE
            elif flag == 'a':
                flag_value |= re.ASCII
            elif flag == 'u':
                flag_value |= re.UNICODE
            elif flag == 'l':
                flag_value |= re.LOCALE
            else:
                return f"Error: Unknown flag '{flag}'"

        # Compile pattern
        try:
            compiled_pattern = re.compile(pattern, flag_value)
        except re.error as e:
            return f"Error: Invalid regex pattern - {str(e)}"

        # Test each string
        results = []
        for test_str in test_strings:
            match = compiled_pattern.match(test_str)
            if match:
                groups = match.groups()
                groupdict = match.groupdict()
                result = {
                    "string": test_str,
                    "matches": True,
                    "start": match.start(),
                    "end": match.end(),
                    "groups": groups,
                    "named_groups": groupdict
                }
            else:
                result = {
                    "string": test_str,
                    "matches": False
                }
            results.append(result)

        # Format results
        output = []
        output.append(f"Testing pattern: {pattern}")
        if flags:
            output.append(f"With flags: {flags}")
        output.append("\nResults:")
        
        for result in results:
            output.append(f"\nTest string: {result['string']}")
            if result['matches']:
                output.append(f"✓ Matches at position {result['start']}-{result['end']}")
                if result['groups']:
                    output.append("Captured groups:")
                    for i, group in enumerate(result['groups'], 1):
                        output.append(f"  Group {i}: {group}")
                if result['named_groups']:
                    output.append("Named groups:")
                    for name, value in result['named_groups'].items():
                        output.append(f"  {name}: {value}")
            else:
                output.append("✗ No match")

        return "\n".join(output)

    except Exception as e:
        return f"Error testing regex: {str(e)}"

def tool_function(command: str, path: str = None, file_text: str = None, start_line: Optional[int] = None, end_line: Optional[int] = None, pattern: str = None, test_strings: List[str] = None, flags: str = "") -> str:
    """
    Main tool function that handles:
      - 'view'      : View the entire file or directory listing
      - 'create'    : Create a new file with the given file_text
      - 'edit'      : Overwrite an existing file or line range with file_text
      - 'test_regex': Test a regex pattern against test strings
    """
    try:
        if command == "test_regex":
            if pattern is None:
                raise ValueError("Missing required `pattern` for 'test_regex' command.")
            if test_strings is None:
                raise ValueError("Missing required `test_strings` for 'test_regex' command.")
            return test_regex(pattern, test_strings, flags)

        if path is None:
            raise ValueError("Missing required `path` parameter.")

        path_obj = validate_path(path, command)

        if command == "view":
            return view_path(path_obj, start_line, end_line)

        elif command == "create":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'create' command.")
            if start_line is not None or end_line is not None:
                raise ValueError("Line range parameters are not supported for 'create' command.")
            write_file(path_obj, file_text)
            return f"File created successfully at: {path}"

        elif command == "edit":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'edit' command.")
            write_file(path_obj, file_text, start_line, end_line)
            if start_line is not None and end_line is not None:
                return f"Lines {start_line}-{end_line} of file at {path} have been updated."
            return f"File at {path} has been overwritten with new content."

        else:
            raise ValueError(f"Unknown command: {command}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    result = tool_function("view", "./coding_agent.py", start_line=1, end_line=10)
    print(result)
