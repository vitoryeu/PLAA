teams = {
    'Los Angeles Lakers': [25, 15, 0, 10, 70],
    'Boston Celtics': [22, 12, 1, 9, 50],
    'Chicago Bulls': [20, 10, 0, 10, 45],
    'New York Knicks': [22, 7, 6, 9, 45],
    'Miami Heat': [23, 14, 0, 9, 65]
}

for team, stats in teams.items():
    print(f"{team.upper()} {' '.join(map(str, stats))}")