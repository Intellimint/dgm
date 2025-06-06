import pytest
from tools.bash import tool_function, BashSession

@pytest.fixture
def bash_session():
    """Create a BashSession instance for testing."""
    session = BashSession()
    return session

class TestBashTool:
    def test_simple_command(self):
        """Test running a simple command."""
        result = tool_function("echo 'hello world'").strip()
        assert result == "hello world"

    def test_multiple_commands(self):
        """Test running multiple commands in sequence."""
        result = tool_function("echo 'first' && echo 'second'").strip()
        assert result == "first\nsecond"

    def test_command_with_error(self):
        """Test running a command that produces an error."""
        result = tool_function("ls /nonexistent/directory").strip()
        assert "Error" in result
        assert "No such file or directory" in result

    def test_environment_variables(self):
        """Test command with environment variables."""
        result = tool_function("TEST_VAR='hello' && echo $TEST_VAR").strip()
        assert result == "hello"

    def test_command_output_processing(self):
        """Test processing of command output."""
        commands = [
            "echo 'line1'",
            "echo 'line2'",
            "echo 'line3'"
        ]
        result = tool_function(" && ".join(commands)).strip()
        assert result == "line1\nline2\nline3"

    def test_long_running_command(self):
        """Test behavior with a long-running command."""
        result = tool_function("sleep 1 && echo 'done'").strip()
        assert result == "done"

    @pytest.mark.parametrize("invalid_command", [
        "invalid_command_name",
        "cd /nonexistent/path",
        "/bin/nonexistent"
    ])
    def test_invalid_commands(self, invalid_command):
        """Test various invalid commands."""
        result = tool_function(invalid_command).strip()
        assert "Error" in result or "command not found" in result

    def test_command_with_special_chars(self):
        """Test command with special characters."""
        result = tool_function("echo 'test with spaces and !@#$%^&*()'").strip()
        assert result == "test with spaces and !@#$%^&*()"

    def test_multiple_line_output(self):
        """Test handling of multiple line output."""
        command = """printf 'line1\nline2\nline3'"""
        result = tool_function(command).strip()
        assert result == "line1\nline2\nline3"

    def test_large_output_handling(self):
        """Test handling of large command output."""
        # Generate a large output
        command = "for i in {1..100}; do echo \"Line $i\"; done"
        result = tool_function(command).strip()
        lines = result.split('\n')
        assert len(lines) == 100
        assert lines[0] == "Line 1"
        assert lines[-1] == "Line 100"
