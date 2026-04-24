import numpy as np

import initialise
from transformation import *

omega_e = initialise.omega_e
radius_earth = initialise.R_e

class LaunchVehicle:
    def __init__(self, initial_mass, stages, initial_latitude, initial_longitude, initial_azimuth):
        self.initial_mass = initial_mass
        self.stages = stages
        self.latitude = initial_latitude
        self.longitude = initial_longitude
        self.azimuth = initial_azimuth

    def check_staging():
        pass
    
    def _get_aero_coefficients(mach_number):
        # Placeholder function to return aerodynamic coefficients based on Mach number
        # In a real implementation, this would likely involve interpolation from a data table
        if mach_number < 0.8:
            return np.array([0.5, 0.1, 0.05])  # Subsonic coefficients
        elif 0.8 <= mach_number < 1.2:
            return np.array([0.3, 0.05, 0.02])  # Transonic coefficients
        else:
            return np.array([0.1, 0.01, 0.005])  # Supersonic coefficients

    # velocity_rel: relative velocity in ECI frame
    def _calculate_aerodynamics(self, velocity_rel, density, reference_area, pitch, yaw, roll, azimuth, path_angle):
        vel_rel_a = GA(azimuth, path_angle) @ IG(self.latitude, self.longitude) @ velocity_rel
        mach_number = np.linalg.norm(vel_rel_a) / initialise.speed_of_sound

        angle_of_attack = np.arctan2(vel_rel_a[2], vel_rel_a[0]) * 180 / np.pi
        side_slip = np.arcsin(vel_rel_a[1], np.linalg.norm(vel_rel_a)) * 180 / np.pi
        dynamic_pressure = 0.5 * density * np.linalg.norm(vel_rel_a) ** 2

        aero_coefficients = self._get_aero_coefficients(mach_number)

        A_a = dynamic_pressure * reference_area * (aero_coefficients * np.array([-1, 1, -1]))
        A_i = LI(self.latitude, self.longitude, self.azimuth) @ BL(pitch, yaw, roll) @ A_a

        return A_i

    def _calculate_thrust(self,thrust):
        T_b = np.array([thrust, 0, 0])  # Assuming thrust acts along the body x-axis
        T_i = LI(self.latitude, self.longitude, self.azimuth) @ BL(self.pitch, self.yaw, self.roll) @ T_b

        return T_i

    def _calculate_gravity(self, position):
        G_i = -initialise.g0 * position / np.linalg.norm(position)  # Gravity acts towards the center of the Earth
        return G_i

class Environment:
    pass

class FlightSimulation3DOF:
    pass