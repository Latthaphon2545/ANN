"""
P7. SIR model.
Student name: Tenaciti Kurios """
if __name__ == "__main__":
    S, I, R = input("Initial S, I, R: ").split()
    c, r = input("c, r: ").split()
    S = float(S)
    I = float(I)
    R = float(R)
    c = float(c)
    r = float(r)
    periods = int(input("periods: "))
    # Write your code here!
        print("--> It is spreading!")
    # Write your code here!
        print("--> It is contained!")
    with open("epidemic.txt", "w") as f:
        m =  "0: S = {:.0f}; I = {:.0f}; R = {:.0f}\n".format(S, I, R)
        f.write(m)
        for n in range(periods):
# Write your code here!
            m =  "{}: S = {:.0f}; I = {:.0f}; R = {:.0f}\n".format(n+1, S, I, R)
            f.write(m)
        # end for
    # end with