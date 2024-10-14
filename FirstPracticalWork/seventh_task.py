# Added comments as a requirement of 12 task
# Vitalii Oryshchyn 14.10.2024

# Method for converting days to hours, minutes and seconds
def convert_to_time(days):
    hours = days * 24
    minutes = hours * 60
    seconds = minutes * 60
    return f"{"Hours:":<10} {hours:<10}\n{"Minutes:":<10} {minutes:<10}\n{"Seconds:":<10} {seconds:<10}"

# With static days
days = 7
print(f"{convert_to_time(days)}\n")

# Added input method as a requirement of 11 task
# Using input days
days = int(input("Please enter the number of days: "))
print(convert_to_time(days))
