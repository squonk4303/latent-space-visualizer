def arr_is_subset(arr1, arr2):
    """
    Compare if arr1 is part of arr2, return bool
    """
    a = arr1
    b = arr2
    return set(a) <= set(b)