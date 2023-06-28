"""
P6. Base-3 stone code.
Student name: Latthaphon Phoemmanirat
"""
stones = input("Stones: ")
lapses = 0
for i in range(1,len(stones)+1):
    cal = (int(stones[-i]) * (3**(i-1)) )
    lapses += cal
print(f"Lapses= {lapses}")