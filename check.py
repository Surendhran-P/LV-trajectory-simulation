import numpy as np

velocity = np.array([[100.0, 0.0, 0.0],
                   [50.0, 0, 0]])  # Example velocity vector in m/s

def calculate_mach_number(velocity):
    speed_of_sound = 343.0  # m/s at sea level
    return np.linalg.norm(velocity,axis=1) / speed_of_sound

print("Mach number:", calculate_mach_number(velocity))