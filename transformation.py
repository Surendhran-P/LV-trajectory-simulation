import numpy as np

def IL (latitude, longitude, azimuth, pitch=0.0):
    '''
    Inertial to Launch Pad reference frame
    '''

    _ = pitch

    delta = np.radians(latitude)
    lamb = np.radians(longitude)
    az = np.radians(azimuth)

    I_L = np.array(
        [[-np.sin(az) * np.sin(lamb) - np.cos(az) * np.sin(delta) * np.cos(lamb),
          np.sin(az) * np.cos(lamb) - np.cos(az) * np.sin(delta) * np.sin(lamb),
          np.cos(az) * np.cos(delta)],
         [-np.cos(az) * np.sin(lamb) + np.sin(az) * np.sin(delta) * np.cos(lamb),
          np.cos(az) * np.cos(lamb) + np.sin(az) * np.sin(delta) * np.sin(lamb),
          -np.sin(az) * np.cos(delta)],
         [-np.cos(delta) * np.cos(lamb),
          -np.cos(delta) * np.sin(lamb),
          -np.sin(delta)]]
    )

    return I_L

def IG (latitude, longitude):
    tau = np.radians(latitude)
    lamb = np.radians(longitude)

    I_G = np.array(
        [[-np.sin(tau) * np.cos(lamb), -np.sin(tau) * np.sin(lamb), np.cos(tau)],
         [-np.sin(lamb), np.cos(lamb), 0],
         [-np.cos(tau) * np.cos(lamb), -np.cos(tau) * np.sin(lamb), -np.sin(tau)]]
    )

    return I_G

def GA (azimuth, path_angle):
    az = np.radians(azimuth)
    gamma = np.radians(path_angle)

    G_A = np.array(
        [[np.cos(gamma) * np.cos(az), np.cos(gamma) * np.sin(az), -np.sin(gamma)],
         [-np.sin(az), np.cos(az), 0],
         [np.sin(gamma) * np.cos(az), np.sin(gamma) * np.sin(az), np.cos(gamma)]]
    )

    return G_A

def AB (angle_of_attack, sideslip, roll):
    '''
    Aerodynamic to Body reference frame
    '''
    alpha = np.radians(angle_of_attack)
    beta = np.radians(sideslip)
    sigma = np.radians(roll)

    A_B = np.array(
        [[np.cos(alpha) * np.cos(beta),
          -np.cos(alpha) * np.sin(beta) * np.cos(sigma) + np.sin(alpha) * np.sin(sigma),
          -np.cos(alpha) * np.sin(beta) * np.sin(sigma) - np.sin(alpha) * np.cos(sigma)],
         [np.sin(beta), np.cos(beta) * np.cos(sigma), np.cos(beta) * np.sin(sigma)],
         [np.sin(alpha) * np.cos(beta),
          -np.sin(alpha) * np.sin(beta) * np.cos(sigma) - np.cos(alpha) * np.sin(sigma),
          -np.sin(alpha) * np.sin(beta) * np.sin(sigma) + np.cos(alpha) * np.cos(sigma)]]
    )

    return A_B

def LB (pitch, yaw, roll):
    '''Launch Pad to Body reference frame
    '''

    theta = np.radians(pitch)
    psi = np.radians(yaw)
    phi = np.radians(roll)

    L_B = np.array(
        [[np.cos(psi) * np.cos(theta), np.sin(psi), -np.cos(psi) * np.sin(theta)],
         [-np.cos(phi) * np.sin(psi) * np.cos(theta) + np.sin(phi) * np.sin(theta),
          np.cos(phi) * np.cos(psi),
          np.cos(phi) * np.sin(psi) * np.sin(theta) + np.sin(phi) * np.cos(theta)],
         [np.sin(phi) * np.sin(psi) * np.cos(theta) + np.cos(phi) * np.sin(theta),
          -np.sin(phi) * np.cos(psi),
          -np.sin(phi) * np.sin(psi) * np.sin(theta) + np.cos(phi) * np.cos(theta)]]
    )

    return L_B

# Inverse of matrices

def LI(latitude, longitude, azimuth):
    return IL(latitude, longitude, azimuth).T

def GI(latitude, longitude):
    return IG(latitude, longitude).T

def AG(azimuth, path_angle):
    return GA(azimuth, path_angle).T

def BA(angle_of_attack, sideslip, roll):
    return AB(angle_of_attack, sideslip, roll).T

def BL(pitch, yaw, roll):
    return LB(pitch, yaw, roll).T