"""
P4. Climate model.
Student name: Latthaphon Phoemmanirat
"""
def nth_root(number, root):
    answer = number**(1/root)
    return answer 

s = float(input("S: "))
a = float(input("a: "))
e = float(input("e: "))
cal_1 = ( ( 1 - a ) * s ) / ( e * ( 5.67 * 10**(-8)) )
t = nth_root(cal_1, 4)
print(f"T= {t:.2f}")