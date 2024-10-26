buddy = {
    'type': 'dog',
    'owner': 'Alex'
}
whiskers = {
    'type': 'cat',
    'owner': 'Sarah'
}
coco = {
    'type': 'parrot',
    'owner': 'Liam'
}
bubbles = {
    'type': 'fish',
    'owner': 'Emily'
}
pets = [buddy, whiskers, coco, bubbles]

for pet in pets:
    print(f"{pet['owner']} is the owner of a pet - a {pet['type']}.")