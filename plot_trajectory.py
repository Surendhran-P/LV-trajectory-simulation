import numpy as np
import matplotlib.pyplot as plt


def plot_(history):
    time = history["time"]
    position = history["position"]
    velocity = history["velocity"]
    mass = history["mass"]
    mach_number = history["mach_number"]
    latitude = history["latitude"]
    longitude = history["longitude"]
    altitude = history["altitude"]
    absolute_flight_path_angle = history["absolute_flight_path_angle"]
    relative_flight_path_angle = history["relative_flight_path_angle"]
    absolute_velocity_azimuth = history["absolute_velocity_azimuth"]
    relative_velocity_azimuth = history["relative_velocity_azimuth"]
    dynamic_pressure = history["dynamic_pressure"]

    # Inertial-frame trajectory coordinates.
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    launch_lat = latitude[0]
    launch_lon = longitude[0]
    relative_lat = latitude - launch_lat
    relative_lon = (longitude - launch_lon + 180.0) % 360.0 - 180.0
    altitude_mag = np.abs(altitude)

    fig_traj = plt.figure(figsize=(8, 6))

    ax_traj = fig_traj.add_subplot(1, 1, 1, projection="3d")
    ax_traj.plot(
        x,
        y,
        z,
        color="tab:blue",
        linewidth=2,
        label="Trajectory",
    )
    ax_traj.scatter(x[0], y[0], z[0], color="green", s=50, label="Launch")
    ax_traj.scatter(
        x[-1],
        y[-1],
        z[-1],
        color="red",
        s=50,
        label="Final",
    )
    ax_traj.set_title("Trajectory in Inertial Frame")
    ax_traj.set_xlabel("X_inertial (m)")
    ax_traj.set_ylabel("Y_inertial (m)")
    ax_traj.set_zlabel("Z_inertial (m)")
    ax_traj.set_box_aspect((1, 1, 1))
    ax_traj.legend()

    fig_geo = plt.figure(figsize=(8, 6))
    ax_geo = fig_geo.add_subplot(1, 1, 1, projection="3d")
    ax_geo.plot(
        relative_lon,
        relative_lat,
        altitude_mag,
        color="tab:cyan",
        linewidth=2,
        label="Trajectory",
    )
    ax_geo.scatter(0.0, 0.0, altitude_mag[0], color="green", s=50, label="Launch")
    ax_geo.scatter(
        relative_lon[-1],
        relative_lat[-1],
        altitude_mag[-1],
        color="red",
        s=50,
        label="Final",
    )
    ax_geo.set_title("Trajectory in Relative Lat-Lon-Height")
    ax_geo.set_xlabel("Relative Longitude (deg)")
    ax_geo.set_ylabel("Relative Latitude (deg)")
    ax_geo.set_zlabel("|Height| (m)")
    ax_geo.legend()

    fig_metrics = plt.figure(figsize=(12, 8))

    ax_vel = fig_metrics.add_subplot(2, 2, 1)
    ax_vel.plot(time, velocity[:, 0], label="vIx")
    ax_vel.plot(time, velocity[:, 1], label="vIy")
    ax_vel.plot(time, velocity[:, 2], label="vIz")
    ax_vel.set_title("Inertial Velocity Components vs Time")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend()

    ax_speed = fig_metrics.add_subplot(2, 2, 2)
    speed = np.linalg.norm(velocity, axis=1)
    ax_speed.plot(time, speed, color="tab:purple", label="|v|")
    ax_speed.set_title("Speed in Inertial Frame vs Time")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend()

    ax_mass = fig_metrics.add_subplot(2, 2, 4)
    ax_mass.plot(time, mass, color="tab:orange", label="Mass")
    ax_mass.set_title("Mass vs Time")
    ax_mass.set_xlabel("Time (s)")
    ax_mass.set_ylabel("Mass (kg)")
    ax_mass.grid(True, alpha=0.3)
    ax_mass.legend()

    # True altitude change from launch radius.
    altitude_relative = altitude - altitude[0]
    ax_alt = fig_metrics.add_subplot(2, 2, 3)
    ax_alt.plot(time, altitude_relative, color='tab:green', linewidth=2)
    ax_alt.set_title("Altitude Above Launch Point vs Time")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True, alpha=0.3)

    fig_flight = plt.figure(figsize=(12, 10))

    ax_alt_abs = fig_flight.add_subplot(3, 2, 1)
    ax_alt_abs.plot(time, altitude, color="tab:green", linewidth=2)
    ax_alt_abs.set_title("Absolute Altitude vs Time")
    ax_alt_abs.set_xlabel("Time (s)")
    ax_alt_abs.set_ylabel("Altitude (m)")
    ax_alt_abs.grid(True, alpha=0.3)

    ax_fpa = fig_flight.add_subplot(3, 2, 2)
    ax_fpa.plot(time, absolute_flight_path_angle, label="Absolute FPA", color="tab:blue")
    ax_fpa.plot(time, relative_flight_path_angle, label="Relative FPA", color="tab:orange")
    ax_fpa.set_title("Flight Path Angle vs Time")
    ax_fpa.set_xlabel("Time (s)")
    ax_fpa.set_ylabel("Angle (deg)")
    ax_fpa.grid(True, alpha=0.3)
    ax_fpa.legend()

    ax_az = fig_flight.add_subplot(3, 2, 3)
    ax_az.plot(time, absolute_velocity_azimuth, label="Absolute Azimuth", color="tab:red")
    ax_az.plot(time, relative_velocity_azimuth, label="Relative Azimuth", color="tab:purple")
    ax_az.set_title("Velocity Azimuth vs Time")
    ax_az.set_xlabel("Time (s)")
    ax_az.set_ylabel("Angle (deg)")
    ax_az.grid(True, alpha=0.3)
    ax_az.legend()

    ax_q = fig_flight.add_subplot(3, 2, 4)
    ax_q.plot(time, dynamic_pressure, color="tab:brown", linewidth=2)
    ax_q.set_title("Dynamic Pressure vs Time")
    ax_q.set_xlabel("Time (s)")
    ax_q.set_ylabel("Pressure (Pa)")
    ax_q.grid(True, alpha=0.3)

    ax_mach = fig_flight.add_subplot(3, 2, 5)
    ax_mach.plot(time, mach_number, color="tab:gray", linewidth=2)
    ax_mach.set_title("Mach Number vs Time")
    ax_mach.set_xlabel("Time (s)")
    ax_mach.set_ylabel("Mach")
    ax_mach.grid(True, alpha=0.3)

    ax_lon = fig_flight.add_subplot(3, 2, 6)
    ax_lon.plot(time, longitude, color="tab:olive", linewidth=2)
    ax_lon.set_title("Inertial Longitude vs Time")
    ax_lon.set_xlabel("Time (s)")
    ax_lon.set_ylabel("Longitude (deg)")
    ax_lon.grid(True, alpha=0.3)

    # 3. Forces Analysis (Thrust vs Drag vs Gravity)
    # Plot each force component to diagnose issues

    fig_traj.tight_layout()
    fig_geo.tight_layout()
    fig_metrics.tight_layout()
    fig_flight.tight_layout()
    plt.show()

    print("Trajectory and metrics plots generated successfully.")