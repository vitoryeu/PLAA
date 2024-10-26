cities = {
    'Kyiv': {
        'country': 'Ukraine',
        'population': '2.8 million',
        'fact': 'It is one of the oldest cities in Eastern Europe, with a history dating back to 482 AD.'
    },
    'Tokyo': {
        'country': 'Japan',
        'population': '14 million',
        'fact': 'Tokyo is the world’s most populous metropolitan area, with over 37 million people in the metro area.'
    },
    'Cairo': {
        'country': 'Egypt',
        'population': '10 million',
        'fact': 'Cairo is home to the Giza Pyramids and the Great Sphinx, which are among the most iconic ancient landmarks in the world.'
    }
}

for city, info in cities.items():
    print(f"City: {city}")
    print(f" Country: {info['country']}")
    print(f" Population: {info['population']}")
    print(f" Fact - {info['fact']}\n")