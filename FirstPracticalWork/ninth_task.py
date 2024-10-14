# Added comments as a requirement of 12 task
# Vitalii Oryshchyn 14.10.2024

# Method for calculating sum of number
def sum_number(number):
    digits = [int(digit) for digit in str(number)]
    print (f"{' + '.join(str(digit) for digit in digits)} = {sum(digits)}\n")

# Static number
sum_number(123)

# Added input method as a requirement of 11 task
# Input number
sum_number(int(input("Please enter number: ")))