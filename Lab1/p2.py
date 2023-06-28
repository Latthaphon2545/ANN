"""
P2. Online averaging.
Student name: Latthaphon Phoemmanirat
"""
if __name__ == "__main__":
    input_time = 0
    newx = 1
    while newx >= 0:
        input_time += 1
        newx = float(input(f"x{input_time}: "))
        if input_time == 1:
            xbar = newx
        else:
            xbar = ( xbar * ( input_time - 1 ) + newx ) / input_time
        print("mean={:.4f}".format(xbar))
