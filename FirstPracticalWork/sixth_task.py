# Added comments as a requirement of 12 task
# Vitalii Oryshchyn 14.10.2024

# Constants
inches = 39.3701
feet = 3.28084
miles = 0.000621371
yards = 1.09361

def conversion_distance_measurement_units(meters):
    distance_meters = "Distance in meters: {:.2f} m".format(meters)
    distance_inches = "Distance in inches: {:.2f} in".format(meters * inches)
    distance_feet = "Distance in feet: {:.2f} ft".format(meters * feet)
    distance_miles = "Distance in miles: {:.2f} mi".format(meters * miles)
    distance_yards = "Distance in yards: {:.2f} yd".format(meters * yards)
    
    return f"{distance_meters}\n{distance_inches}\n{distance_feet}\n{distance_miles}\n{distance_yards}"

# Without input
# Static meters
meters = 1000
print(f"{conversion_distance_measurement_units(meters)}\n")

# Added input method as a requirement of 11 task
# With input
meters = float(input("Please enter meters: "))
print(conversion_distance_measurement_units(meters))