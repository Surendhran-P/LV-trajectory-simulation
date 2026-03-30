import numpy as np
from staging import Rocket, RocketStage
import initialise
from trajectory import *
from plot_trajectory import plot_

# Sizing of 3 stage rocket
rocket = Rocket(payload_mass=3500, target_delta_v=5199)

stage1 = RocketStage(name="Stage 1", isp=275, structural_ratio=0.15, thrust=2496e3)
stage2 = RocketStage(name="Stage 2", isp=295, structural_ratio=0.12, thrust=250e3)
stage3 = RocketStage(name="Stage 3", isp=292, structural_ratio=0.1, thrust=160e3)

rocket.add_stage(stage1)
rocket.add_stage(stage2)
rocket.add_stage(stage3)

otpimal_eta = rocket.optimize_lagrange()
rocket.calculate_stage_masses(otpimal_eta)

print("Staging completed.")

# Initial conditions for trajectory simulation
result = initialise.initial_state(latitude=8.531, longitude=76.875, azimuth=225, cg_pos=2)
simulate = FlightSimulation(
    initial_mass=rocket.total_mass,
    mass_flow_rate=rocket.stages[0].mass_flow_rate,
    drag_coefficient=0.2,
    initial_position=result[0],
    initial_velocity=result[1],
    area=0.07,
    thrust=103000,
    latitude=8.531,
    longitude=76.875,
    azimuth=225,
)

# Perform trajectory simulation
history = simulate.execute_flight(t_final=100.0, dt=0.1)

print("Completed trajectory simulation.")

# Plot the trajectory
plot_(history)

print("Trajectory plotted.")