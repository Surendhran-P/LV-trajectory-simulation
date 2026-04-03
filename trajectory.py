import numpy as np

import initialise
from transformation import *

omega_e = initialise.omega_e
radius_earth = initialise.R_e

class FlightSimulation:
    def __init__(
        self,
        initial_mass,
        mass_flow_rate,
        drag_coefficient,
        initial_position,
        initial_velocity,
        area,
        thrust,
        latitude=0.0,
        longitude=0.0,
        azimuth=90.0,
        pitch=90.0,
        yaw=0.0,
        roll=0.0,
        angle_of_attack=0.0,
        sideslip=0.0,
        aerodynamic_roll=0.0,
    ):

        # Initial conditions
        self.position = np.array(initial_position, dtype=float)  # x, y, z
        self.velocity = np.array(initial_velocity, dtype=float)  # vx, vy, vz
        self.acceleration = np.array([0.0, 0.0, 0.0])  # ax, ay, az

        self.area = area  # Reference area for aerodynamic calculations
        
        self.mass = initial_mass
        self.mass_flow_rate = mass_flow_rate
        self.drag_coefficient = drag_coefficient
        self.thrust = thrust

        # Vehicle orientation and aerodynamic angles in degrees.
        self.latitude = latitude
        self.longitude = longitude
        self.azimuth = azimuth
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.angle_of_attack = angle_of_attack
        self.sideslip = sideslip
        self.aerodynamic_roll = aerodynamic_roll

        self.density = 1.225  # kg/m^3, sea level standard density  ## Needs to be changed

        self.history = {}  # To store the trajectory history


    def _calculate_aerodynamics(self, position, velocity, area):
        velocity_rel = velocity - np.cross(omega_e, position)

        vel_rel_b = np.linalg.multi_dot([LB(self.pitch, self.yaw, self.roll), IL(self.latitude, self.longitude, self.azimuth), velocity_rel])
        u, v, w = vel_rel_b[0], vel_rel_b[1], vel_rel_b[2]
        self.angle_of_attack = np.arctan2(w, u) * 180.0 / np.pi
        self.sideslip = np.arctan2(v, np.sqrt(u**2 + w**2)) * 180.0 / np.pi

        dynamic_pressure = 0.5 * self.density * np.linalg.norm(velocity_rel)**2
        drag_magnitude = dynamic_pressure * self.drag_coefficient * area

        A_a = np.array([-drag_magnitude, 0, 0])  # Assuming drag acts opposite to velocity vector

        A_i = np.linalg.multi_dot([LI(self.latitude, self.longitude, self.azimuth),
                                   BL(self.pitch, self.yaw, self.roll),
                                   AB(self.angle_of_attack, self.sideslip, self.aerodynamic_roll), 
                                   A_a])

        return A_i

    def _calculate_thrust(self, thrust):

        T_b = np.array([thrust, 0, 0])  # Assuming thrust acts along the body x-axis
        T_i = np.linalg.multi_dot([
            LI(self.latitude, self.longitude, self.azimuth),
            BL(self.pitch, self.yaw, self.roll),
            T_b,
        ])

        return T_i
    
    def _calculate_gravity(self, position):
        G_i = -initialise.g0 * position / np.linalg.norm(position)  # Gravity acts towards the center of the Earth
        return G_i

    def _calculate_acceleration(self, position, velocity, mass):

        # Calculate forces
        A_i = self._calculate_aerodynamics(position, velocity, self.area)
        T_i = self._calculate_thrust(self.thrust)
        G_i = self._calculate_gravity(position)

        effective_mass = max(mass, 1e-6)

        # Total acceleration
        self.acceleration = G_i + (A_i + T_i) / effective_mass
        return self.acceleration

    def _state_derivative(self, t, state):
        position = state[:3]
        velocity = state[3:6]
        mass = state[6]
        # mach_number = state[7]
        # rel_latitude = state[8]
        # inertial_latitude = state[9]
        # geodetic_latitude = state[10]
        # geocentric_latitude = state[11]
        # absolute_flight_path_angle = state[12]
        # relative_flight_path_angle = state[13]
        # dynamic_pressure = state[14]

        mass_derivative = -self.mass_flow_rate
        acceleration = self._calculate_acceleration(position, velocity, mass)

        return np.hstack((velocity, acceleration, mass_derivative))

    def rk4(self, derivative_fn, t0, y0, t_final, dt):
        times = [float(t0)]
        states = [np.array(y0, dtype=float)]

        t = float(t0)
        y = np.array(y0, dtype=float)

        while t < t_final:
            h = min(dt, t_final - t)

            k1 = derivative_fn(t, y)
            k2 = derivative_fn(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = derivative_fn(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = derivative_fn(t + h, y + h * k3)

            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + h

            times.append(t)
            states.append(y.copy())

        return np.array(times), np.vstack(states)
    
    def _calculate_mach_number(self, velocity):
        speed_of_sound = 343.0  # m/s at sea level
        return np.linalg.norm(velocity,axis=1) / speed_of_sound
    
    def _update_geographic_coordinates(self, position):
        r = np.linalg.norm(position,axis=1)
        latitude = np.arcsin(position[:, 2] / r) * 180.0 / np.pi # Geodetic latitude in degrees
        longitude = np.arctan2(position[:, 1], position[:, 0]) * 180.0 / np.pi # Inertial longitude in degrees
        return latitude, longitude
    
    def _update_relative_geographic_coordinates(self, latitude, longitude, dt):
        relative_latitude = latitude - self.latitude
        relative_longitude = longitude - omega_e[2] * dt
        return relative_latitude, relative_longitude
        

    def _calculate_altitude(self, position):
        r = np.linalg.norm(position,axis=1)
        altitude = r - radius_earth  # Subtract Earth's radius to get altitude above sea level
        return altitude

    def _calculate_flight_path_angles(self, velocity_rel, position):
        speed = np.linalg.norm(velocity_rel, axis=1)
        absoulute_flight_path_angle = np.arcsin(velocity_rel[:, 2] / speed) * 180.0 / np.pi 
        relative_flight_path_angle = absoulute_flight_path_angle - self.latitude
        return absoulute_flight_path_angle, relative_flight_path_angle
    
    def _calculate_velocity_azimuth(self, velocity):
        absolute_velocity_azimuth = np.arctan2(velocity[:, 1], velocity[:, 0]) * 180.0 / np.pi
        relative_velocity_azimuth = absolute_velocity_azimuth - self.azimuth # check
        return absolute_velocity_azimuth, relative_velocity_azimuth
    
    def _caulculate_dynamic_pressure(self, velocity):
        dynamic_pressure = 0.5 * self.density * np.linalg.norm(velocity, axis=1)**2
        return dynamic_pressure
    
    def execute_flight(self, t_final=10.0, dt=0.1):
        initial_state = np.hstack((self.position, self.velocity, self.mass))

        time_history, state_history = self.rk4(
            self._state_derivative,
            0.0,
            initial_state,
            t_final,
            dt,
        )
       
        self.history = {
            "time": time_history,
            "position": state_history[:, :3],
            "velocity": state_history[:, 3:6],
            "mass": state_history[:, 6],
        }

        self.history["mach_number"] = self._calculate_mach_number(self.history["velocity"])
        self.history["latitude"], self.history["longitude"] = self._update_geographic_coordinates(self.history["position"])
        self.history["relative_latitude"], self.history["relative_longitude"] = self._update_relative_geographic_coordinates(self.history["latitude"], self.history["longitude"], dt)
        self.history["altitude"] = self._calculate_altitude(self.history["position"])
        self.history["absolute_flight_path_angle"], self.history["relative_flight_path_angle"] = self._calculate_flight_path_angles(self.history["velocity"], self.history["position"])
        self.history["absolute_velocity_azimuth"], self.history["relative_velocity_azimuth"] = self._calculate_velocity_azimuth(self.history["velocity"])
        self.history["dynamic_pressure"] = self._caulculate_dynamic_pressure(self.history["velocity"])

        return self.history
