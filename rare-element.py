from collections import Counter

def nth_most_rare(elements, n):
    """
    :param elements: (list) List of integers.
    :param n: (int) The n-th element function should return.
    :returns: (int) The n-th most rare element in the elements list.
    """
    occurence_count = Counter(elements) 
    return occurence_count.most_common(None)[-n][0] 

print(nth_most_rare([5, 4, 3, 2, 1, 5, 4, 3, 2, 5, 4, 3, 5, 4, 5], 2))