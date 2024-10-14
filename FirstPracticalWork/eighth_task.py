# Added comments as a requirement of 12 task
# Vitalii Oryshchyn 14.10.2024

# Method for converting temperature to F and K
def convert_temperature(celsius):
    fahrenheit = 32 + 9/5 * celsius
    kelvin = celsius + 273.15
    print(f"{'Celsius (C)':^15} {'Fahrenheit (F)':^15} {'Kelvin (K)':^15}")
    print(f"{celsius:^15.2f} {fahrenheit:^15.2f} {kelvin:^15.2f}")

# With static temperature in C
celsius = 25
convert_temperature(25)

# Added input method as a requirement of 11 task
# Using input temperature in C
celsius = int(input("Please enter the temperature in Celsius: "))
convert_temperature(celsius)