def concat_string_array(array):
    if len(array) > 1:
        result = ', '.join(array[:-1]) + ' and ' + array[-1]
    else:
        result = array[0]
    print(f"Concat result:\n{result}")

array = ['Budapest', 'Rome', 'Istanbul', 'Sydney', 'Kyiv', 'Hong Kong']
concat_string_array(array)