def test_longest_common_subsequence():
    """Test the implementation of longest common subsequence."""
    from lcs import longest_common_subsequence
    
    # Test cases
    test_cases = [
        ("ABCDGH", "AEDFHR", "ADH"),  # Basic case
        ("AGGTAB", "GXTXAYB", "GTAB"),  # Another basic case
        ("", "", ""),  # Empty strings
        ("ABC", "", ""),  # One empty string
        ("", "ABC", ""),  # One empty string
        ("ABC", "ABC", "ABC"),  # Identical strings
        ("ABC", "DEF", ""),  # No common subsequence
    ]
    
    for str1, str2, expected in test_cases:
        result = longest_common_subsequence(str1, str2)
        assert result == expected, f"Failed for str1='{str1}', str2='{str2}'. Expected '{expected}', got '{result}'" 