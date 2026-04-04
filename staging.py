import numpy as np
from scipy.optimize import root_scalar

class RocketStage:
    def __init__(self, name, isp, structural_ratio, thrust):
        self.name = name
        self.isp = isp  # I_sp
        self.thrust = thrust # F
        self.propellant_mass = 0  # m_P
        self.structural_mass = 0  # m_E
        self.exhaust_velocity = isp * 9.81 # Ve
        self.mass_flow_rate = thrust / self.exhaust_velocity # m_dot
        self.mass_ratio = 0 # n
        self.total_mass = 0 # m0
        self.step_mass = 0 # m_i
        self.structural_ratio = structural_ratio # epsilon
        self.stage_payload = 0 # m_pl

class Rocket:
    def __init__(self, payload_mass, target_delta_v):
        self.payload_mass = payload_mass  # Mass of the payload in kg
        self.total_mass = 0 # Total mass of the rocket (payload + stages)
        self.target_delta_v = target_delta_v  # Target delta-v in m/s
        self.stages = []  # List to hold the stages of the rocket

    def add_stage(self, stage: RocketStage):
        self.stages.append(stage)

    def calculate_total_mass(self):
        total_mass = self.payload_mass
        for stage in self.stages:
            total_mass += stage.propellant_mass + stage.structural_mass
        return total_mass
    
    def lagrange_equation(self, eta):
        equation = 0
        for stage in self.stages:
            n_i = (stage.exhaust_velocity * eta - 1) / (stage.exhaust_velocity * eta * stage.structural_ratio)
            equation += stage.exhaust_velocity * np.log(n_i)
        return equation - self.target_delta_v


    def optimize_lagrange(self):
        eta_min = max(1 / (stage.exhaust_velocity * (1 - stage.structural_ratio)) for stage in self.stages) + 1e-6

        eta_max = eta_min * 2

        while self.lagrange_equation(eta_max) < 0:
            eta_max *= 2
            print(eta_max)

        solution = root_scalar(self.lagrange_equation, bracket=[eta_min, eta_max], method='brentq')

        optimal_eta = solution.root

        return optimal_eta
    
    def calculate_stage_masses(self, optimal_eta):
        reversed_stages = self.stages[::-1]
        payload_mass = self.payload_mass

        for stage in reversed_stages:
            stage.stage_payload = payload_mass
            n_i = (stage.exhaust_velocity * optimal_eta - 1) / (stage.exhaust_velocity * optimal_eta * stage.structural_ratio)
            stage.mass_ratio = n_i
            stage.step_mass = ((n_i - 1) / (1 - n_i * stage.structural_ratio)) * payload_mass
            stage.total_mass = stage.step_mass + payload_mass
            payload_mass += stage.step_mass
            

        self.total_mass = self.payload_mass + sum(stage.step_mass for stage in self.stages)

        for stage in self.stages:
            stage.structural_mass = stage.structural_ratio * stage.step_mass
            stage.propellant_mass = stage.step_mass - stage.structural_mass

    def total_burn_time(self, stage):
        if stage.mass_flow_rate > 0:
            return stage.propellant_mass / stage.mass_flow_rate
        else:
            return 0