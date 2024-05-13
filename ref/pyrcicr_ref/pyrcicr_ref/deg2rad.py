import math


def deg2rad(deg):
    return (deg * math.pi) / 180

# Example usage
degrees = 180
radians = deg2rad(degrees)
print(radians)  # This will print the equivalent in radians
