def longest_common_subsequence(str1: str, str2: str) -> str:
    """
    Find the longest common subsequence of two strings.
    
    Args:
        str1: First input string
        str2: Second input string
        
    Returns:
        The longest common subsequence as a string
    """
    # Handle empty string cases
    if not str1 or not str2:
        return ""
        
    # Initialize DP table
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the subsequence
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
            
    return "".join(reversed(lcs)) 