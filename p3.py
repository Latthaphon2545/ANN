"""
P3. EEG.
Student name: Latthaphon Phoemmanirat
"""

name_file = input("File: ")
number_series = input("Series: ").split()

with open(name_file) as f:
    lines = f.readlines()

find_position = []
for i in range(len(lines)):
    new_line = lines[i].split()
    if new_line[0] == number_series[0] and new_line[1] == number_series[1]:
        find_position.append(new_line[3])
print(list(map(float, find_position)))        
    