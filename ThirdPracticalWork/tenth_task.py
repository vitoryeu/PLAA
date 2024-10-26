things = {
    'key': 3,
    'mace': 1,
    'gold coin': 24,
    'lantern': 1,
    'stone': 10
}
total_items = 0

print("Equipment")
for item, quantity in things.items():
    print(f"{quantity} {item}")
    total_items += quantity
print(f"Total number of things: {total_items}")