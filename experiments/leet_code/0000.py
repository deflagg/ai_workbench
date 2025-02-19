from typing import TypeVar, TypedDict, Annotated, List, Literal, Union, Optional, Dict, Any
import heapq


# Input: nums = [2,7,11,15], target = 9

def pair_sum(nums, target):
    hashmap = {}
    
    for i in range(len(nums)):
        comp = target - nums[i]
        if comp in hashmap:
            return [hashmap[comp], i]
        else:
            hashmap[nums[i]] = i



result = pair_sum([2,7,11,15], 9)
print(result)