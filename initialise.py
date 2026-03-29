import numpy as np

g0 = 9.80665
omega_e=np.array([0, 0, 7.2921159E-5]) # Earth's angular velocity in rads per second
R_e = 6378165.8 # Earth's equitorial radius in meters
R_p = 6356783.8 # Earth's polar radius in meters

k = (R_e/R_p)**2

def initial_state(latitude, longitude, azimuth, cg_pos=0):
    '''
    Calculate initial position and velocity vectors
    Parameters:
    latitude: Geodetic Latitude of the launch site in degrees
    longitude: Longitude of the launch site in degrees
    azimuth: Azimuth angle of the launch in degrees

    Returns:
    Position: position vector in ECI frame
    Velocity: velocity vector in ECI frame
    '''
    phi_d = np.radians(latitude)
    lambda_ = np.radians(longitude)
    Az = np.radians(azimuth)

    phi = np.arctan(k * np.tan(phi_d))

    R_s = R_e / np.sqrt(1 + (k - 1) * np.sin(phi)**2) # R_phi, radius at given latitude

    x = (R_s + cg_pos) * np.cos(phi) * np.cos(lambda_)
    y = (R_s + cg_pos) * np.cos(phi) * np.sin(lambda_)
    z = (R_s + cg_pos) * np.sin(phi)

    r = np.array([x, y, z])

    v = np.cross(omega_e, r)

    return r,v


