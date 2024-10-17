def combined_array_number(str):
    if len(str) > 0:
        digits_list = str.split()
        reversed_digit_list = digits_list[::-1]
        combined_list = digits_list + reversed_digit_list
        print(f"Combined list:\n{combined_list}\nCombined number: {''.join(combined_list)}")

combined_array_number(input("Enter digits: "))