def sum_numbers(str):
    if len(str) > 0:
        numbers = list(map(int, str.split()))
        print("Sum of the numbers:", sum(numbers))

sum_numbers(input("Enter numbers:"))