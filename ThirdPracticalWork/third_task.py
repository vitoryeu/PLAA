programingLanguages = {
    'Python': 'Guido van Rossum',
    'JavaScript': 'Brendan Eich',
    'Ruby': 'Yukihiro Matsumoto',
    'C++': 'Bjarne Stroustrup'
}

for language, creator in programingLanguages.items():
    print(f"My favorite programming language is {language}. It was created by {creator}.")

del programingLanguages['Ruby']

print(f"\nUpdated languages dictionary:\n{programingLanguages}")