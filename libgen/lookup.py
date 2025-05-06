def find_missing(lst1, lst2):
    """
    Find the elements in lst1 that are not in lst2.
    """
    if len(lst2)>len(lst1):
        lst_temp = lst2
        lst2 = lst1
        lst1 = lst_temp
    return [item for item in lst1 if item not in lst2]