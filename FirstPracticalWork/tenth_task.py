# Added comments as a requirement of 12 task
# Vitalii Oryshchyn 14.10.2024
import math

# Constants
radius_earth = 6371.032

def calculate_distance_by_cordinates(first_coordinates, second_coordinates):
    x1 = math.radians(first_coordinates[0])
    y1 = math.radians(first_coordinates[1])
    x2 = math.radians(second_coordinates[0])
    y2 = math.radians(second_coordinates[1])
    distance = radius_earth * math.acos(math.sin(x1) * math.sin(x2) + 
                                     math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
    print(f"{"First coordinates:":<20} ({first_coordinates[0]}, {first_coordinates[1]})\n{"Second coordinates:":<20} ({second_coordinates[0]}, {second_coordinates[1]})\nDistance:{distance:>10.3f}")

calculate_distance_by_cordinates((39.9075000, 116.3972300), (50.4546600, 30.5238000))