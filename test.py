def count_subsets_with_target_sum(arr, target):
    n = len(arr)
    dp = [[0] * (target + 1) for _ in range(n + 1)]
    
    # Base case: one subset (empty subset) for sum 0
    for i in range(n + 1):
        dp[i][0] = 1
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(target + 1):
            if arr[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] + dp[i - 1][j - arr[i - 1]]
    
    return dp[n][target]

# Example usage
arr = [1, 2, 3, 4, 5]
target = 5
print("Number of subsets:", count_subsets_with_target_sum(arr, target))
