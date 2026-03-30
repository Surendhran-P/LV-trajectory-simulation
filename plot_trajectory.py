import numpy as np
import matplotlib.pyplot as plt

import initialise


def plot_(history):
    time = history["time"]
    position = history["position"]
    velocity = history["velocity"]
    mass = history["mass"]
    altitude_absolute = history["absolute_height"]
    flight_path_angle_relative = np.degrees(history["relative_flight_path_angle"])
    flight_path_angle_absolute = np.degrees(history["absolute_flight_path_angle"])
    velocity_azimuth_relative = np.degrees(history["relative_velocity_azimuth"])
    velocity_azimuth_absolute = np.degrees(history["absolute_velocity_azimuth"])
    dynamic_pressure = history["dynamic_pressure"]
    geocentric_latitude = np.degrees(history["geocentric_latitude"])
    geodetic_longitude = np.degrees(history["geodetic_longitude"])
    inertial_longitude = np.degrees(history["inertial_longitude"])
    relative_longitude = np.degrees(history["relative_longitude"])

    # Inertial-frame trajectory coordinates.
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]

    omega_earth = float(initialise.omega_e[2])


    def inertial_to_earth_matrix(t):
        theta = omega_earth * t
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])


    position_er = np.zeros_like(position)
    for i, t in enumerate(time):
        i_to_er = inertial_to_earth_matrix(t)
        position_er[i] = i_to_er @ position[i]

    # Convert Earth-rotating Cartesian coordinates to geodetic longitude, latitude, and height.
    a = float(initialise.R_e)
    b = float(initialise.R_p)
    e2 = 1.0 - (b * b) / (a * a)
    ep2 = (a * a - b * b) / (b * b)

    x_er = position_er[:, 0]
    y_er = position_er[:, 1]
    z_er = position_er[:, 2]

    longitude = np.degrees(np.arctan2(y_er, x_er))
    p = np.sqrt(x_er * x_er + y_er * y_er)
    theta = np.arctan2(z_er * a, p * b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    latitude = np.arctan2(z_er + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    sin_lat = np.sin(latitude)
    n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
    height = p / np.maximum(np.cos(latitude), 1e-12) - n
    height_mag = np.abs(height)
    latitude = np.degrees(latitude)

    launch_lat = latitude[0]
    launch_lon = longitude[0]
    relative_lat = latitude - launch_lat
    relative_lon = (longitude - launch_lon + 180.0) % 360.0 - 180.0

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
        height_mag,
        color="tab:cyan",
        linewidth=2,
        label="Trajectory",
    )
    ax_geo.scatter(0.0, 0.0, height_mag[0], color="green", s=50, label="Launch")
    ax_geo.scatter(
        relative_lon[-1],
        relative_lat[-1],
        height_mag[-1],
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

    # True altitude change from launch radius in inertial coordinates.
    altitude = np.linalg.norm(position, axis=1) - np.linalg.norm(position[0])
    ax_alt = fig_metrics.add_subplot(2, 2, 3)
    ax_alt.plot(time, altitude, color='tab:green', linewidth=2)
    ax_alt.set_title("Altitude Above Launch Point vs Time")
    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True, alpha=0.3)

    fig_angles = plt.figure(figsize=(12, 8))

    ax_geo_lat = fig_angles.add_subplot(2, 1, 1)
    ax_geo_lat.plot(time, geocentric_latitude, color="tab:blue", linewidth=2)
    ax_geo_lat.set_title("Geocentric Latitude vs Time")
    ax_geo_lat.set_xlabel("Time (s)")
    ax_geo_lat.set_ylabel("Latitude (deg)")
    ax_geo_lat.grid(True, alpha=0.3)

    ax_lon = fig_angles.add_subplot(2, 1, 2)
    ax_lon.plot(time, geodetic_longitude, label="Geodetic Longitude", color="tab:green")
    ax_lon.plot(time, inertial_longitude, label="Inertial Longitude", color="tab:orange")
    ax_lon.plot(time, relative_longitude, label="Relative Longitude", color="tab:red")
    ax_lon.set_title("Longitude Components vs Time")
    ax_lon.set_xlabel("Time (s)")
    ax_lon.set_ylabel("Longitude (deg)")
    ax_lon.grid(True, alpha=0.3)
    ax_lon.legend()

    fig_flight = plt.figure(figsize=(12, 10))

    ax_alt_abs = fig_flight.add_subplot(3, 2, 1)
    ax_alt_abs.plot(time, altitude_absolute, color="tab:green", linewidth=2)
    ax_alt_abs.set_title("Absolute Altitude vs Time")
    ax_alt_abs.set_xlabel("Time (s)")
    ax_alt_abs.set_ylabel("Altitude (m)")
    ax_alt_abs.grid(True, alpha=0.3)

    ax_fpa = fig_flight.add_subplot(3, 2, 2)
    ax_fpa.plot(time, flight_path_angle_absolute, label="Absolute FPA", color="tab:blue")
    ax_fpa.plot(time, flight_path_angle_relative, label="Relative FPA", color="tab:orange")
    ax_fpa.set_title("Flight Path Angle vs Time")
    ax_fpa.set_xlabel("Time (s)")
    ax_fpa.set_ylabel("Angle (deg)")
    ax_fpa.grid(True, alpha=0.3)
    ax_fpa.legend()

    ax_az = fig_flight.add_subplot(3, 2, 3)
    ax_az.plot(time, velocity_azimuth_absolute, label="Absolute Azimuth", color="tab:red")
    ax_az.plot(time, velocity_azimuth_relative, label="Relative Azimuth", color="tab:purple")
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

    # 2. Acceleration Magnitude vs Time
    # (You'll need to compute this from your simulation's acceleration history)
    # acceleration_mag = np.linalg.norm(acceleration_history, axis=1)
    # ax_accel = fig_metrics.add_subplot(2, 2, 4)
    # ax_accel.plot(time, acceleration_mag, color='tab:red', label="|acceleration|")
    # ax_accel.set_title("Net Acceleration vs Time")
    # ax_accel.set_xlabel("Time (s)")
    # ax_accel.set_ylabel("Acceleration (m/s²)")
    # ax_accel.grid(True, alpha=0.3)
    # ax_accel.legend()

    # 3. Forces Analysis (Thrust vs Drag vs Gravity)
    # Plot each force component to diagnose issues

    fig_traj.tight_layout()
    fig_geo.tight_layout()
    fig_metrics.tight_layout()
    fig_angles.tight_layout()
    fig_flight.tight_layout()
    plt.show()

    print("Trajectory and metrics plots generated successfully.")