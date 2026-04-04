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
    relative_lat = history["relative_latitude"]
    relative_lon = history["relative_longitude"]
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
    speed = np.linalg.norm(velocity, axis=1)
    position_magnitude = np.linalg.norm(position, axis=1)

    altitude_mag = np.abs(altitude) / 1000.0

    # Window 1: relative latitude/longitude with altitude magnitude.
    fig_window1 = plt.figure(figsize=(9, 7))
    ax_window1 = fig_window1.add_subplot(1, 1, 1, projection="3d")
    ax_window1.plot(
        relative_lat,
        relative_lon,
        altitude_mag,
        color="tab:blue",
        linewidth=2,
        label="Trajectory",
    )
    ax_window1.scatter(0.0, 0.0, altitude_mag[0], color="green", s=55, label="Launch")
    ax_window1.scatter(
        relative_lat[-1],
        relative_lon[-1],
        altitude_mag[-1],
        color="red",
        s=55,
        label="Final",
    )
    ax_window1.set_title("Window 1: Relative Lat-Lon-Altitude Magnitude")
    ax_window1.set_xlabel("Relative Latitude (deg)")
    ax_window1.set_ylabel("Relative Longitude (deg)")
    ax_window1.set_zlabel("|Altitude| (km)")
    ax_window1.legend()

    # Window 2: inertial longitude/geodetic latitude with altitude.
    fig_window2 = plt.figure(figsize=(9, 7))
    ax_window2 = fig_window2.add_subplot(1, 1, 1, projection="3d")
    ax_window2.plot(
        longitude,
        latitude,
        np.absolute(altitude) / 1000.0,
        color="tab:cyan",
        linewidth=2,
        label="Trajectory",
    )
    ax_window2.scatter(longitude[0], latitude[0], np.absolute(altitude[0]) / 1000.0, color="green", s=55, label="Launch")
    ax_window2.scatter(
        longitude[-1],
        latitude[-1],
        np.absolute(altitude[-1]) / 1000.0,
        color="red",
        s=55,
        label="Final",
    )
    ax_window2.set_title("Window 2: Inertial Longitude-Geodetic Latitude-Altitude")
    ax_window2.set_xlabel("Inertial Longitude (deg)")
    ax_window2.set_ylabel("Geodetic Latitude (deg)")
    ax_window2.set_zlabel("Altitude (km)")
    ax_window2.legend()

    # Window 3: requested flight metrics with time on x-axis.
    fig_window3, axs_window3 = plt.subplots(2, 2, figsize=(12, 8))

    axs_window3[0, 0].plot(time, np.absolute(altitude) / 1000.0, color="tab:green", linewidth=2)
    axs_window3[0, 0].set_title("Altitude vs Time")
    axs_window3[0, 0].set_xlabel("Time (s)")
    axs_window3[0, 0].set_ylabel("Altitude (km)")
    axs_window3[0, 0].grid(True, alpha=0.3)

    axs_window3[0, 1].plot(time, absolute_flight_path_angle, color="tab:blue", linewidth=2)
    axs_window3[0, 1].plot(
        time,
        relative_flight_path_angle,
        color="tab:orange",
        linestyle="--",
        linewidth=1.5,
    )
    axs_window3[0, 1].set_title("Flight Path Angle vs Time")
    axs_window3[0, 1].set_xlabel("Time (s)")
    axs_window3[0, 1].set_ylabel("Angle (deg)")
    axs_window3[0, 1].grid(True, alpha=0.3)
    axs_window3[0, 1].legend(["Absolute", "Relative"])

    axs_window3[1, 0].plot(time, absolute_velocity_azimuth, color="tab:red", linewidth=2)
    axs_window3[1, 0].plot(
        time,
        relative_velocity_azimuth,
        color="tab:purple",
        linestyle="--",
        linewidth=1.5,
    )
    axs_window3[1, 0].set_title("Velocity Azimuth vs Time")
    axs_window3[1, 0].set_xlabel("Time (s)")
    axs_window3[1, 0].set_ylabel("Angle (deg)")
    axs_window3[1, 0].grid(True, alpha=0.3)
    axs_window3[1, 0].legend(["Absolute", "Relative"])

    axs_window3[1, 1].plot(time, dynamic_pressure, color="tab:brown", linewidth=2)
    axs_window3[1, 1].set_title("Dynamic Pressure vs Time")
    axs_window3[1, 1].set_xlabel("Time (s)")
    axs_window3[1, 1].set_ylabel("Pressure (Pa)")
    axs_window3[1, 1].grid(True, alpha=0.3)

    # Window 4: mass, velocity magnitude, position magnitude variations.
    fig_window4, axs_window4 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs_window4[0].plot(time, mass, color="tab:orange", linewidth=2)
    axs_window4[0].set_title("Mass vs Time")
    axs_window4[0].set_ylabel("Mass (kg)")
    axs_window4[0].grid(True, alpha=0.3)

    axs_window4[1].plot(time, speed, color="tab:purple", linewidth=2)
    axs_window4[1].set_title("Velocity Magnitude vs Time")
    axs_window4[1].set_ylabel("|Velocity| (m/s)")
    axs_window4[1].grid(True, alpha=0.3)

    axs_window4[2].plot(time, position_magnitude, color="tab:gray", linewidth=2)
    axs_window4[2].set_title("Position Magnitude vs Time")
    axs_window4[2].set_xlabel("Time (s)")
    axs_window4[2].set_ylabel("|Position| (m)")
    axs_window4[2].grid(True, alpha=0.3)

    fig_window1.tight_layout()
    fig_window2.tight_layout()
    fig_window3.tight_layout()
    fig_window4.tight_layout()
    plt.show()

    print("Trajectory plots generated successfully.")