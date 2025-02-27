def arr_is_subset(arr1, arr2):
    """
    Compare if arr1 is part of arr2, return bool
    """
    return set(arr1) <= set(arr2)