import numpy as np

import initialise
from transformation import *

omega_e = initialise.omega_e
radius_earth = initialise.R_e

class LaunchVehicle:
    def __init__(self, initial_mass, stages, latitude, longitude, azimuth):
        self.initial_mass = initial_mass
        self.stages = stages
        self.latitude = latitude
        self.longitude = longitude
        self.azimuth = azimuth

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

class FlightSimulation:
    def __init__(self, launch_vehicle):
        self.launch_vehicle = launch_vehicle
        self.reference_area = float(getattr(launch_vehicle, "reference_area", 3.14159))
        self.reference_length = float(getattr(launch_vehicle, "reference_length", 20.0))

        inertia_default = np.array([1.20e8, 1.20e8, 2.00e7], dtype=float)
        inertia_diagonal = np.array(
            getattr(launch_vehicle, "inertia_diagonal", inertia_default),
            dtype=float,
        )
        if inertia_diagonal.shape != (3,):
            inertia_diagonal = inertia_default
        inertia_diagonal = np.maximum(inertia_diagonal, 1.0)

        self.inertia = np.diag(inertia_diagonal)
        self.inertia_inv = np.diag(1.0 / inertia_diagonal)

        self.density_sea_level = 1.225
        self.scale_height = 8500.0

        self.control_kp = float(getattr(launch_vehicle, "attitude_kp", 2.0e7))
        self.control_kd = float(getattr(launch_vehicle, "attitude_kd", 2.5e7))
        self.max_control_moment = float(getattr(launch_vehicle, "max_control_moment", 8.0e7))

        self.aero_force_coeffs = {
            "cd0": 0.20,
            "cd_alpha2": 2.5,
            "cd_beta2": 1.5,
            "cy_beta": 0.25,
            "cz_alpha": 2.8,
        }
        self.aero_moment_coeffs = {
            "cl_beta": -0.08,
            "cl_p": -0.60,
            "cm_alpha": -0.90,
            "cm_q": -8.00,
            "cn_beta": 0.06,
            "cn_r": -6.00,
        }

    @staticmethod
    def _normalize(vector, fallback=None):
        vec = np.asarray(vector, dtype=float)
        magnitude = np.linalg.norm(vec)
        if magnitude < 1e-12:
            if fallback is None:
                return np.array([1.0, 0.0, 0.0], dtype=float)
            return np.asarray(fallback, dtype=float)
        return vec / magnitude

    @staticmethod
    def _quat_normalize(quaternion):
        quat = np.asarray(quaternion, dtype=float)
        magnitude = np.linalg.norm(quat)
        if magnitude < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return quat / magnitude

    @staticmethod
    def _quat_multiply(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_to_dcm(quaternion):
        q = FlightSimulation._quat_normalize(quaternion)
        w, x, y, z = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def _dcm_to_quat(dcm):
        c = np.asarray(dcm, dtype=float)
        trace = np.trace(c)

        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (c[2, 1] - c[1, 2]) / s
            qy = (c[0, 2] - c[2, 0]) / s
            qz = (c[1, 0] - c[0, 1]) / s
        elif c[0, 0] > c[1, 1] and c[0, 0] > c[2, 2]:
            s = np.sqrt(1.0 + c[0, 0] - c[1, 1] - c[2, 2]) * 2.0
            qw = (c[2, 1] - c[1, 2]) / s
            qx = 0.25 * s
            qy = (c[0, 1] + c[1, 0]) / s
            qz = (c[0, 2] + c[2, 0]) / s
        elif c[1, 1] > c[2, 2]:
            s = np.sqrt(1.0 + c[1, 1] - c[0, 0] - c[2, 2]) * 2.0
            qw = (c[0, 2] - c[2, 0]) / s
            qx = (c[0, 1] + c[1, 0]) / s
            qy = 0.25 * s
            qz = (c[1, 2] + c[2, 1]) / s
        else:
            s = np.sqrt(1.0 + c[2, 2] - c[0, 0] - c[1, 1]) * 2.0
            qw = (c[1, 0] - c[0, 1]) / s
            qx = (c[0, 2] + c[2, 0]) / s
            qy = (c[1, 2] + c[2, 1]) / s
            qz = 0.25 * s

        return FlightSimulation._quat_normalize(np.array([qw, qx, qy, qz], dtype=float))

    @staticmethod
    def _quat_to_euler_deg(quaternion):
        w, x, y, z = FlightSimulation._quat_normalize(quaternion)

        sin_roll = 2.0 * (w * x + y * z)
        cos_roll = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sin_roll, cos_roll)

        sin_pitch = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))

        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(sin_yaw, cos_yaw)

        return np.degrees(np.array([roll, pitch, yaw], dtype=float))

    def _initial_attitude_quaternion(self, latitude, longitude, azimuth):
        dcm_ib = LI(latitude, longitude, azimuth)
        return self._dcm_to_quat(dcm_ib)

    def _atmospheric_density(self, altitude):
        altitude_clamped = max(float(altitude), 0.0)
        density = self.density_sea_level * np.exp(-altitude_clamped / self.scale_height)
        return float(np.clip(density, 1.0e-6, self.density_sea_level))

    def _guidance_direction(self, phase_name, position_vec, velocity_vec):
        radial_dir = self._normalize(position_vec, fallback=np.array([0.0, 0.0, 1.0]))
        velocity_rel = velocity_vec - np.cross(omega_e, position_vec)
        velocity_dir = self._normalize(velocity_rel, fallback=radial_dir)

        if phase_name == "vertical_ascent":
            return radial_dir

        if phase_name == "gravity_turn":
            blend = 0.10
        elif phase_name == "insertion":
            blend = 0.25
        else:
            blend = 0.15

        return self._normalize((1.0 - blend) * velocity_dir + blend * radial_dir, fallback=velocity_dir)

    def _aero_forces_and_moments(self, position_i, velocity_i, omega_b, quat_bi):
        dcm_ib = self._quat_to_dcm(quat_bi)
        dcm_bi = dcm_ib.T

        velocity_rel_i = velocity_i - np.cross(omega_e, position_i)
        velocity_rel_b = dcm_bi @ velocity_rel_i
        velocity_rel_b = np.clip(velocity_rel_b, -1.5e4, 1.5e4)

        u, v, w = velocity_rel_b
        speed = np.linalg.norm(velocity_rel_b)

        if speed < 1.0e-6:
            zero_vec = np.array([0.0, 0.0, 0.0], dtype=float)
            return zero_vec, zero_vec, 0.0, 0.0, 0.0, 0.0, velocity_rel_i

        alpha = np.arctan2(w, u if abs(u) > 1.0e-9 else 1.0e-9)
        beta = np.arctan2(v, np.sqrt(u * u + w * w) + 1.0e-9)
        alpha = float(np.clip(alpha, -np.radians(85.0), np.radians(85.0)))
        beta = float(np.clip(beta, -np.radians(85.0), np.radians(85.0)))

        radius = np.linalg.norm(position_i)
        altitude = radius - radius_earth
        density = self._atmospheric_density(altitude)
        dynamic_pressure = float(np.clip(0.5 * density * speed * speed, 0.0, 5.0e7))

        cd = (
            self.aero_force_coeffs["cd0"]
            + self.aero_force_coeffs["cd_alpha2"] * alpha * alpha
            + self.aero_force_coeffs["cd_beta2"] * beta * beta
        )
        cy = self.aero_force_coeffs["cy_beta"] * beta
        cz = self.aero_force_coeffs["cz_alpha"] * alpha

        force_b = dynamic_pressure * self.reference_area * np.array([-cd, cy, -cz], dtype=float)

        speed_ref = max(speed, 1.0)
        rate_scale = self.reference_length / (2.0 * speed_ref)
        p_hat = omega_b[0] * rate_scale
        q_hat = omega_b[1] * rate_scale
        r_hat = omega_b[2] * rate_scale

        cl = self.aero_moment_coeffs["cl_beta"] * beta + self.aero_moment_coeffs["cl_p"] * p_hat
        cm = self.aero_moment_coeffs["cm_alpha"] * alpha + self.aero_moment_coeffs["cm_q"] * q_hat
        cn = self.aero_moment_coeffs["cn_beta"] * beta + self.aero_moment_coeffs["cn_r"] * r_hat

        moment_b = dynamic_pressure * self.reference_area * self.reference_length * np.array([cl, cm, cn], dtype=float)

        return force_b, moment_b, alpha, beta, dynamic_pressure, speed, velocity_rel_i

    def execute_flight(self, dt=0.1, phase_timing=None):
        # Tuple format: (phase_name, phase_duration_seconds)
        if phase_timing is None:
            phase_timing = (
                ("vertical_ascent", 5.0),
                ("gravity_turn", 75.0),
                ("insertion", 15.5),
            )

        latitude = self.launch_vehicle.latitude
        longitude = self.launch_vehicle.longitude
        azimuth = self.launch_vehicle.azimuth

        position, velocity = initialise.initial_state(latitude, longitude, azimuth)
        initial_mass = float(self.launch_vehicle.initial_mass)
        quat_bi = self._initial_attitude_quaternion(latitude, longitude, azimuth)
        omega_b = np.array([0.0, 0.0, 0.0], dtype=float)
        state = np.hstack((position, velocity, quat_bi, omega_b, initial_mass))

        stage = self.launch_vehicle.stages[0] if self.launch_vehicle.stages else None
        stage_thrust = float(getattr(stage, "thrust", 0.0)) if stage is not None else 0.0
        stage_mass_flow_rate = float(getattr(stage, "mass_flow_rate", 0.0)) if stage is not None else 0.0

        if stage is not None:
            propellant_mass = max(float(getattr(stage, "propellant_mass", 0.0)), 0.0)
        else:
            propellant_mass = 0.0

        if propellant_mass > 0.0:
            dry_mass = max(initial_mass - propellant_mass, 1.0)
        else:
            dry_mass = max(0.12 * initial_mass, 1.0)
            if dry_mass >= initial_mass:
                dry_mass = 0.5 * initial_mass

        def get_stage_thrust_and_mass_flow(mass_value):
            if mass_value <= dry_mass + 1.0e-6:
                return 0.0, 0.0
            return stage_thrust, stage_mass_flow_rate

        def state_derivative(_, current_state, phase_name):
            pos = current_state[:3]
            vel = current_state[3:6]
            quat = self._quat_normalize(current_state[6:10])
            omega = np.clip(current_state[10:13], -5.0, 5.0)
            mass = max(current_state[13], dry_mass)

            thrust, mass_flow_rate = get_stage_thrust_and_mass_flow(mass)
            thrust_force_b = np.array([thrust, 0.0, 0.0], dtype=float)

            aero_force_b, aero_moment_b, _, _, _, _, _ = self._aero_forces_and_moments(pos, vel, omega, quat)

            dcm_ib = self._quat_to_dcm(quat)
            dcm_bi = dcm_ib.T

            desired_dir_i = self._guidance_direction(phase_name, pos, vel)
            body_x_axis_i = dcm_ib @ np.array([1.0, 0.0, 0.0], dtype=float)
            attitude_error_i = np.cross(body_x_axis_i, desired_dir_i)
            attitude_error_b = dcm_bi @ attitude_error_i

            control_moment_b = self.control_kp * attitude_error_b - self.control_kd * omega
            control_moment_norm = np.linalg.norm(control_moment_b)
            if control_moment_norm > self.max_control_moment:
                control_moment_b = control_moment_b * (self.max_control_moment / control_moment_norm)

            total_force_b = thrust_force_b + aero_force_b
            total_moment_b = aero_moment_b + control_moment_b

            force_i = dcm_ib @ total_force_b

            radius = max(np.linalg.norm(pos), 1.0)
            gravity_magnitude = initialise.g0 * (radius_earth / radius) ** 2
            gravity = -gravity_magnitude * pos / radius

            acceleration_i = gravity + force_i / max(mass, 1.0e-6)

            inertia_omega = self.inertia @ omega
            omega_dot = self.inertia_inv @ (total_moment_b - np.cross(omega, inertia_omega))
            omega_dot = np.clip(omega_dot, -2.0, 2.0)

            omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
            quat_dot = 0.5 * self._quat_multiply(quat, omega_quat)

            mass_dot = -mass_flow_rate
            return np.hstack((vel, acceleration_i, quat_dot, omega_dot, mass_dot))

        def rk4_phase(t_start, y_start, duration, phase_name):
            times = [float(t_start)]
            states = [np.array(y_start, dtype=float)]

            t = float(t_start)
            y = np.array(y_start, dtype=float)
            t_end = t_start + duration

            while t < t_end:
                h = min(dt, t_end - t)

                k1 = state_derivative(t, y, phase_name)
                k2 = state_derivative(t + 0.5 * h, y + 0.5 * h * k1, phase_name)
                k3 = state_derivative(t + 0.5 * h, y + 0.5 * h * k2, phase_name)
                k4 = state_derivative(t + h, y + h * k3, phase_name)

                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                y[6:10] = self._quat_normalize(y[6:10])
                y[13] = max(y[13], dry_mass)
                t = t + h

                times.append(t)
                states.append(y.copy())

            return np.array(times), np.vstack(states)

        phase_results = []
        current_time = 0.0

        for phase_name, phase_duration in phase_timing:
            phase_time, phase_states = rk4_phase(current_time, state, float(phase_duration), phase_name)
            phase_results.append((phase_name, phase_time, phase_states))

            state = phase_states[-1].copy()
            current_time = phase_time[-1]

        all_times = []
        all_states = []
        for idx, (_, phase_time, phase_states) in enumerate(phase_results):
            if idx == 0:
                all_times.append(phase_time)
                all_states.append(phase_states)
            else:
                all_times.append(phase_time[1:])
                all_states.append(phase_states[1:])

        time_history = np.concatenate(all_times)
        state_history = np.vstack(all_states)

        position_history = state_history[:, :3]
        velocity_history = state_history[:, 3:6]
        quaternion_history = state_history[:, 6:10]
        quat_norm = np.linalg.norm(quaternion_history, axis=1, keepdims=True)
        quaternion_history = quaternion_history / np.maximum(quat_norm, 1.0e-12)
        angular_rate_history = state_history[:, 10:13]
        mass_history = state_history[:, 13]

        sample_count = position_history.shape[0]
        angle_of_attack = np.zeros(sample_count, dtype=float)
        sideslip = np.zeros(sample_count, dtype=float)
        dynamic_pressure = np.zeros(sample_count, dtype=float)
        mach_number = np.zeros(sample_count, dtype=float)
        aero_force_body = np.zeros((sample_count, 3), dtype=float)
        aero_moment_body = np.zeros((sample_count, 3), dtype=float)
        velocity_relative = np.zeros((sample_count, 3), dtype=float)

        for idx in range(sample_count):
            force_b, moment_b, alpha, beta, q_dyn, speed_rel, vel_rel_i = self._aero_forces_and_moments(
                position_history[idx],
                velocity_history[idx],
                angular_rate_history[idx],
                quaternion_history[idx],
            )
            aero_force_body[idx] = force_b
            aero_moment_body[idx] = moment_b
            angle_of_attack[idx] = np.degrees(alpha)
            sideslip[idx] = np.degrees(beta)
            dynamic_pressure[idx] = q_dyn
            mach_number[idx] = speed_rel / max(initialise.speed_of_sound, 1.0e-9)
            velocity_relative[idx] = vel_rel_i

        attitude_euler_deg = np.array([self._quat_to_euler_deg(q) for q in quaternion_history], dtype=float)

        speed = np.linalg.norm(velocity_history, axis=1)
        radius = np.linalg.norm(position_history, axis=1)
        altitude = radius - radius_earth

        latitude_hist = np.degrees(np.arcsin(np.clip(position_history[:, 2] / np.maximum(radius, 1e-9), -1.0, 1.0)))
        longitude_hist = np.degrees(np.arctan2(position_history[:, 1], position_history[:, 0]))
        relative_latitude = latitude_hist - latitude
        relative_longitude = longitude_hist - longitude - np.degrees(omega_e[2] * time_history)

        speed_relative = np.linalg.norm(velocity_relative, axis=1)
        flight_path_angle = np.degrees(
            np.arcsin(np.clip(velocity_relative[:, 2] / np.maximum(speed_relative, 1.0e-9), -1.0, 1.0))
        )
        velocity_azimuth = np.degrees(np.arctan2(velocity_relative[:, 1], velocity_relative[:, 0]))

        history = {
            "time": time_history,
            "position": position_history,
            "velocity": velocity_history,
            "mass": mass_history,
            "quaternion_bi": quaternion_history,
            "attitude_euler_deg": attitude_euler_deg,
            "angular_rates_body_rad_s": angular_rate_history,
            "angular_rates_body_deg_s": np.degrees(angular_rate_history),
            "angle_of_attack": angle_of_attack,
            "sideslip": sideslip,
            "aero_force_body": aero_force_body,
            "aero_moment_body": aero_moment_body,
            "velocity_relative": velocity_relative,
            "mach_number": mach_number,
            "latitude": latitude_hist,
            "longitude": longitude_hist,
            "relative_latitude": relative_latitude,
            "relative_longitude": relative_longitude,
            "altitude": altitude,
            "absolute_flight_path_angle": flight_path_angle,
            "relative_flight_path_angle": flight_path_angle - latitude,
            "absolute_velocity_azimuth": velocity_azimuth,
            "relative_velocity_azimuth": velocity_azimuth - azimuth,
            "dynamic_pressure": dynamic_pressure,
            "phase_timing": phase_timing,
        }

        return history
