"""
P1. Orientation.
Student name: Latthaphon Phoemmanirat
"""
observe = input("observe: ")
if observe == "U":
    print("facing east") 
elif observe == "D":
    print("facing west")
elif observe == "L":
    print("facing north")
elif observe == "R":
    print("facing south") 
elif observe == "UL":
    print("facing north-east") 
elif observe == "UR":
    print("facing south-east")
elif observe == "DL":
    print("facing north-west") 
elif observe == "DR":
    print("facing south-west")