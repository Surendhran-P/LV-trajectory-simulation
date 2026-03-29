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

        self.history = {
            "time": time_history,
            "position": state_history[:, :3],
            "velocity": state_history[:, 3:6],
            "mass": state_history[:, 6],
        }

        return self.history
