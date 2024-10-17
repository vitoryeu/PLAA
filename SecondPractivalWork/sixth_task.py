keywords = ('for', 'if', 'else', 'in', ':')
code_lines = [
    f"{keywords[0]} each token {keywords[3]} the postfix expression{keywords[4]}",
    f"    {keywords[1]} the token is a number{keywords[4]}",
    "        print('Convert it to an integer and add it to the end of values')",
    f"    {keywords[2]}{keywords[4]}",
    "        print('Append the result to the end of values')"
]
for line in code_lines:
    print(line)