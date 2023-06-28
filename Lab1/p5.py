"""
P5. Old-time sailor.
Student name: Latthaphon Phoemmanirat
"""
def time2min(timetxt):
    meridiem = timetxt[-1]
    h, m = timetxt[:-1].split(":")
    mins = 0
    if meridiem == 'p':
        mins = 720 # offset the pm time to 720 mins
    # end if
    mins += 60* (int(h)%12) + int(m)    # 12:00a = 0:00a
    return mins
if __name__ == "__main__":
    GMTnoon = 720 # noon = 12:00pm = 12*60 = 720 min
    GMT = input("GMT: ")
    mins = time2min(GMT)
    # Write your code here!
    if  mins < GMTnoon:
        direction = "E"
    else:
        direction = "W"
    longitude = abs( (mins - GMTnoon) / 4 )
    print("longitude= {:.2f} {}".format(longitude, direction))
# end __main__