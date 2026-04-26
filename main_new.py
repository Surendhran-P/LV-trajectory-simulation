import numpy as np
from staging import Rocket, RocketStage
import initialise
from trajectory import *
from plot_trajectory import plot_

# Sizing of 3 stage rocket
rocket = Rocket(payload_mass=1, target_delta_v=100)

stage1 = RocketStage(name="Stage 1", isp=275, structural_ratio=0.15, thrust=10e3)

rocket.add_stage(stage1)

rocket.calculate_stage_masses(1)

totalburn_time = sum(rocket.total_burn_time(stage) for stage in rocket.stages)
print(f"Total burn time: ", totalburn_time, " seconds")

# Initial conditions for trajectory simulation
model_rocket = LaunchVehicle(
    initial_mass=rocket.total_mass,
    stages=rocket.stages,
    latitude=8.531,
    longitude=76.875,
    azimuth=225,
)

result = initialise.initial_state(latitude=8.543, longitude=76.859, azimuth=225)

simulate = FlightSimulation(model_rocket)

# Perform trajectory simulation
history = simulate.execute_flight(dt=0.1)

print("Completed trajectory simulation.")

# Plot the trajectory
plot_(history)

print("Trajectory plotted.")