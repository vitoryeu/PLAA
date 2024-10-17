import copy

def array_modification(array, method):
    if len(array) > 0:
        shalow_array_copy = copy.copy(array)
        print(f'Original array:\n{array}')
        if method == 1:
            print(f'Modified array (sorted):\n{sorted(shalow_array_copy)}')
        elif method == 2:
            shalow_array_copy.reverse()
            print(f'Modified array (reverse):\n{shalow_array_copy}')
        elif method == 3:
            shalow_array_copy.sort()
            print(f'Modified array (sort):\n{shalow_array_copy}')
        print()