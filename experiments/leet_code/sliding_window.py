from typing import TypedDict, Annotated, List, Literal, Union, Optional, Dict, Any

def max_subarray_sum(nums: List[int], k: int) -> int:
    left = 0
    right = k - 1
    current_sum = 0
    max_sum = 0
    
    if k > len(nums):
        return 0

    for i in range(k):
        current_sum += nums[i]
    
    max_sum = current_sum
        
    while right < len(nums) - 1:
        current_sum -= nums[left]
        left += 1
        right += 1
        current_sum += nums[right]

        if current_sum > max_sum:
            max_sum = current_sum
            
    
    return max_sum

result = max_subarray_sum([2, 1, 5, 1, 3, 2], 3)
print(result)