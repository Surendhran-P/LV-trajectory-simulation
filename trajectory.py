import numpy as np

import initialise
from transformation import *

omega_e = initialise.omega_e

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

    def _wrap_to_pi(self, angle_rad):
        return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi

    def _calculate_geospatial_history(self, position_history, time_history):
        x = position_history[:, 0]
        y = position_history[:, 1]
        z = position_history[:, 2]

        radius_xy = np.hypot(x, y)
        radius = np.linalg.norm(position_history, axis=1)

        geocentric_latitude = np.arctan2(z, radius_xy)
        inertial_longitude = np.arctan2(y, x)

        earth_rotation_rate = np.linalg.norm(omega_e)
        geodetic_longitude = self._wrap_to_pi(inertial_longitude - earth_rotation_rate * time_history)
        relative_longitude = self._wrap_to_pi(geodetic_longitude - np.radians(self.longitude))
        absolute_height = radius - initialise.R_e

        return np.hstack(
            (
                geocentric_latitude[:, None],
                geodetic_longitude[:, None],
                inertial_longitude[:, None],
                relative_longitude[:, None],
                absolute_height[:, None],
            )
        )

    def _calculate_flight_kinematics_history(self, position_history, velocity_history):
        radius = np.linalg.norm(position_history, axis=1)
        radius_safe = np.maximum(radius, 1e-12)

        radial_unit = position_history / radius_safe[:, None]
        velocity_rel = velocity_history - np.cross(omega_e, position_history)

        speed_abs = np.linalg.norm(velocity_history, axis=1)
        speed_rel = np.linalg.norm(velocity_rel, axis=1)
        speed_abs_safe = np.maximum(speed_abs, 1e-12)
        speed_rel_safe = np.maximum(speed_rel, 1e-12)

        radial_speed_abs = np.einsum("ij,ij->i", velocity_history, radial_unit)
        radial_speed_rel = np.einsum("ij,ij->i", velocity_rel, radial_unit)

        horizontal_speed_abs = np.sqrt(np.maximum(speed_abs**2 - radial_speed_abs**2, 0.0))
        horizontal_speed_rel = np.sqrt(np.maximum(speed_rel**2 - radial_speed_rel**2, 0.0))

        flight_path_angle_abs = np.arctan2(radial_speed_abs, horizontal_speed_abs)
        flight_path_angle_rel = np.arctan2(radial_speed_rel, horizontal_speed_rel)

        x = position_history[:, 0]
        y = position_history[:, 1]
        z = position_history[:, 2]
        inertial_longitude = np.arctan2(y, x)
        geocentric_latitude = np.arctan2(z, np.hypot(x, y))

        east_unit = np.column_stack((-np.sin(inertial_longitude), np.cos(inertial_longitude), np.zeros_like(inertial_longitude)))
        north_unit = np.column_stack(
            (
                -np.sin(geocentric_latitude) * np.cos(inertial_longitude),
                -np.sin(geocentric_latitude) * np.sin(inertial_longitude),
                np.cos(geocentric_latitude),
            )
        )

        east_speed_abs = np.einsum("ij,ij->i", velocity_history, east_unit)
        north_speed_abs = np.einsum("ij,ij->i", velocity_history, north_unit)
        east_speed_rel = np.einsum("ij,ij->i", velocity_rel, east_unit)
        north_speed_rel = np.einsum("ij,ij->i", velocity_rel, north_unit)

        velocity_azimuth_abs = np.arctan2(east_speed_abs, north_speed_abs)
        velocity_azimuth_rel = np.arctan2(east_speed_rel, north_speed_rel)

        dynamic_pressure = 0.5 * self.density * speed_rel_safe**2

        return np.hstack(
            (
                flight_path_angle_rel[:, None],
                flight_path_angle_abs[:, None],
                velocity_azimuth_rel[:, None],
                velocity_azimuth_abs[:, None],
                dynamic_pressure[:, None],
            )
        )


    def _calculate_aerodynamics(self, position, velocity, area):
        velocity_rel = velocity - np.cross(omega_e, position)
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

        acceleration = self._calculate_acceleration(position, velocity, mass)
        mass_derivative = -self.mass_flow_rate

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

    def execute_flight(self, t_final=10.0, dt=0.1):
        initial_state = np.hstack((self.position, self.velocity, self.mass))

        time_history, state_history = self.rk4(
            self._state_derivative,
            0.0,
            initial_state,
            t_final,
            dt,
        )

        self.position = state_history[-1, :3]
        self.velocity = state_history[-1, 3:6]
        self.mass = state_history[-1, 6]
        self.acceleration = self._calculate_acceleration(self.position, self.velocity, self.mass)

        geospatial_history = self._calculate_geospatial_history(state_history[:, :3], time_history)
        flight_kinematics_history = self._calculate_flight_kinematics_history(
            state_history[:, :3],
            state_history[:, 3:6],
        )

        self.history = {
            "time": time_history,
            "position": state_history[:, :3],
            "velocity": state_history[:, 3:6],
            "mass": state_history[:, 6],
            "geospatial": geospatial_history,
            "geocentric_latitude": geospatial_history[:, 0],
            "geodetic_longitude": geospatial_history[:, 1],
            "inertial_longitude": geospatial_history[:, 2],
            "relative_longitude": geospatial_history[:, 3],
            "absolute_height": geospatial_history[:, 4],
            "flight_kinematics": flight_kinematics_history,
            "relative_flight_path_angle": flight_kinematics_history[:, 0],
            "absolute_flight_path_angle": flight_kinematics_history[:, 1],
            "relative_velocity_azimuth": flight_kinematics_history[:, 2],
            "absolute_velocity_azimuth": flight_kinematics_history[:, 3],
            "dynamic_pressure": flight_kinematics_history[:, 4],
        }

        return self.history
