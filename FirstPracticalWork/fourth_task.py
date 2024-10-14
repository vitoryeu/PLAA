# Added comments as a requirement of 12 task
# Vitalii Oryshchyn 14.10.2024

#Added input method as a requirement of 11 task
name = f"\t{input("Please enter user name: ")}\t\n"
#Print name with tans and new line
print(f"Name with tabs and new line:\n{name}")

#Print name without left tab
print(f"Name without left tab:\n{name.lstrip()}")

#Print name without right tab and new lin
print(f"Name without right tab and new line:\n{name.rstrip()}")

#Print name without left, right tab and without line feed
print(f"Name without tabs and new line:\n{name.strip()}")