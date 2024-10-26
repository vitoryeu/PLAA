e2g = {
    'stork': 'storch',
    'hawk': 'falke',
    'woodpecker': 'specht',
    'owl': 'eule'
}

print(f"English to German dictionary:\n{e2g}")
print(f"\nThe German word for 'owl' is: {e2g['owl']}")

e2g['sparrow'] = 'spatz'
e2g['eagle'] = 'adler'

print(f"\nUpdated English to German dictionary:\n{e2g}")
print(f"\nKeys in the dictionary:\n{list(e2g.keys())}\n")
print(f"Values in the dictionary:\n{list(e2g.values())}")