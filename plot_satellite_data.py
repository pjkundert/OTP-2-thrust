#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates
from pykalman import KalmanFilter
from scipy.optimize import minimize, curve_fit

def parse_timestamp(timestamp_str):
    """Parse timestamp from format '2025-09-22T16:23:43+00:00 (25265.68313586)'"""
    datetime_part = timestamp_str.split(' (')[0]
    return pd.to_datetime(datetime_part)

def calculate_velocity_change_rate(timestamps, velocities):
    """Calculate the rate of change in velocity (acceleration)"""
    if len(timestamps) < 2:
        return np.array([])

    # Convert timestamps to seconds for proper time differential
    time_seconds = [(t - timestamps[0]).total_seconds() for t in timestamps]

    # Calculate acceleration using numpy gradient
    acceleration = np.gradient(velocities, time_seconds)

    return acceleration

def pykalman_filter(timestamps, altitudes, velocities):
    """
    Simple smoothing using pykalman library for orbital data.

    Uses independent filtering for altitude and velocity since orbital mechanics
    relationships are complex and our simple constant velocity model doesn't apply well.
    """
    if len(timestamps) < 2:
        return np.array(altitudes), np.array(velocities)

    # Filter altitude independently
    altitude_observations = np.array(altitudes).reshape(-1, 1)

    # Simple random walk model for altitude (no velocity coupling)
    kf_alt = KalmanFilter(
        transition_matrices=np.array([[1.0]]),      # altitude[t] = altitude[t-1]
        observation_matrices=np.array([[1.0]]),     # we observe altitude directly
        initial_state_mean=np.array([altitudes[0]]),
        initial_state_covariance=np.array([[10.0]]),
        transition_covariance=np.array([[0.1]]),    # small process noise
        observation_covariance=np.array([[2.0]])    # measurement noise
    )

    alt_states, _ = kf_alt.smooth(altitude_observations)
    filtered_altitudes = alt_states.flatten()

    # Filter velocity independently
    velocity_observations = np.array(velocities).reshape(-1, 1)

    kf_vel = KalmanFilter(
        transition_matrices=np.array([[1.0]]),      # velocity[t] = velocity[t-1]
        observation_matrices=np.array([[1.0]]),     # we observe velocity directly
        initial_state_mean=np.array([velocities[0]]),
        initial_state_covariance=np.array([[100.0]]),
        transition_covariance=np.array([[0.5]]),    # small process noise
        observation_covariance=np.array([[10.0]])   # measurement noise
    )

    vel_states, _ = kf_vel.smooth(velocity_observations)
    filtered_velocities = vel_states.flatten()

    return filtered_altitudes, filtered_velocities

def fit_velocity_altitude_relationship(velocities, altitudes):
    """
    Deduce the relationship between velocity and altitude using curve fitting.

    Tests multiple orbital mechanics models and returns the best fit:
    1. Vis-viva equation: v = sqrt(μ/(R + h)) where h is altitude, R is Earth radius
    2. Power law: v = a * h^b
    3. Inverse relationship: v = a / (h + b)
    4. Exponential: v = a * exp(b * h)
    """
    velocities = np.array(velocities)
    altitudes = np.array(altitudes)

    # Convert velocity from km/h to m/s for physics calculations
    velocities_ms = velocities * 1000 / 3600
    altitudes_m = altitudes * 1000  # Convert to meters

    print("Fitting velocity-altitude relationships...")

    # Model 1: Vis-viva equation v = sqrt(μ/(R + h))
    def vis_viva_model(h, mu_eff, R_eff):
        """Orbital velocity model based on vis-viva equation"""
        # Prevent division by zero and negative values
        denominator = np.maximum(R_eff + h, 1e6)
        return np.sqrt(np.maximum(mu_eff / denominator, 0))

    # Model 2: Power law v = a * h^b
    def power_law_model(h, a, b):
        """Power law relationship"""
        return a * np.power(np.maximum(h, 1), b)

    # Model 3: Inverse relationship v = a / (h + b)
    def inverse_model(h, a, b):
        """Inverse relationship"""
        return a / np.maximum(h + b, 1)

    # Model 4: Exponential v = a * exp(b * h)
    def exponential_model(h, a, b):
        """Exponential relationship"""
        return a * np.exp(b * h)

    models = [
        ("Vis-viva (orbital)", vis_viva_model, [3.986e14, 6.371e6]),  # Initial: GM_earth, R_earth
        ("Power law", power_law_model, [30000, -0.5]),                # Initial: reasonable values
        ("Inverse", inverse_model, [2e10, 6.371e6]),                  # Initial: a, offset
        ("Exponential", exponential_model, [28000, -1e-6])            # Initial: a, decay rate
    ]

    best_model = None
    best_params = None
    best_rmse = float('inf')
    best_name = None

    for name, model_func, initial_params in models:
        try:
            # Fit the model
            popt, pcov = curve_fit(
                model_func,
                altitudes_m,
                velocities_ms,
                p0=initial_params,
                maxfev=5000,
                bounds=([0] * len(initial_params), [np.inf] * len(initial_params))
            )

            # Calculate RMSE
            predicted = model_func(altitudes_m, *popt)
            rmse = np.sqrt(np.mean((velocities_ms - predicted)**2))

            # Calculate R²
            ss_res = np.sum((velocities_ms - predicted)**2)
            ss_tot = np.sum((velocities_ms - np.mean(velocities_ms))**2)
            r_squared = 1 - (ss_res / ss_tot)

            print(f"  {name:15s}: RMSE = {rmse:.2f} m/s, R² = {r_squared:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_func
                best_params = popt
                best_name = name

        except Exception as e:
            print(f"  {name:15s}: Failed to fit ({e})")

    if best_model is None:
        print("  Warning: No model could be fitted, using linear approximation")
        # Fallback to simple linear fit
        coeffs = np.polyfit(altitudes, velocities, 1)
        best_model = lambda h, a, b: a * h + b
        best_params = coeffs
        best_name = "Linear fallback"

    print(f"\nBest model: {best_name} (RMSE: {best_rmse:.2f} m/s)")
    print(f"Parameters: {best_params}")

    def predict_altitude_from_velocity(vel_kmh):
        """
        Predict altitude from velocity using the fitted model.
        Uses numerical root finding since we fitted v = f(h), but need h = f^(-1)(v)
        """
        vel_ms = np.array(vel_kmh) * 1000 / 3600
        predicted_altitudes = []

        for v in vel_ms:
            def objective(h):
                if best_name == "Linear fallback":
                    predicted_v = best_model(h/1000, *best_params)  # Convert back to km for linear
                else:
                    predicted_v = best_model(h, *best_params)
                return (predicted_v - v)**2

            # Search for altitude that gives the target velocity
            from scipy.optimize import minimize_scalar

            # Search in reasonable range (400-600 km altitude)
            result = minimize_scalar(objective, bounds=(400000, 600000), method='bounded')

            if result.success:
                predicted_altitudes.append(result.x / 1000)  # Convert back to km
            else:
                # Fallback: use average altitude
                predicted_altitudes.append(np.mean(altitudes))

        return np.array(predicted_altitudes)

    return predict_altitude_from_velocity, best_name, best_params, best_rmse

def main():
    # Read the tab-separated CSV file
    print("Reading data from 2025-052AB-data.csv...")

    # Read the file with tab separator and proper column names
    df = pd.read_csv('2025-052AB-data.csv',
                     sep='\t',
                     header=None,
                     names=['satellite', 'timestamp', 'altitude', 'velocity', 'time_ago'])

    print(f"Loaded {len(df)} data points")

    # Parse timestamps
    df['datetime'] = df['timestamp'].apply(parse_timestamp)

    # Sort by timestamp (oldest first)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Fit velocity-altitude relationship
    predict_altitude_func, best_model_name, best_params, best_rmse = fit_velocity_altitude_relationship(
        df['velocity'].tolist(), df['altitude'].tolist()
    )

    # Predict altitude from velocity using the fitted model
    df['predicted_altitude'] = predict_altitude_func(df['velocity'])

    # Calculate prediction accuracy
    altitude_prediction_rmse = np.sqrt(np.mean((df['altitude'] - df['predicted_altitude'])**2))
    print(f"\nAltitude prediction accuracy: {altitude_prediction_rmse:.3f} km RMSE")

    # Apply Kalman filtering to smooth the data
    print("Applying PyKalman smoothing (forward-backward pass)...")
    filtered_alt, filtered_vel = pykalman_filter(
        df['datetime'].tolist(),
        df['altitude'].tolist(),
        df['velocity'].tolist()
    )
    df['filtered_altitude'] = filtered_alt
    df['filtered_velocity'] = filtered_vel

    # Calculate noise reduction metrics
    altitude_noise_reduction = (df['altitude'].std() - df['filtered_altitude'].std()) / df['altitude'].std() * 100
    velocity_noise_reduction = (df['velocity'].std() - df['filtered_velocity'].std()) / df['velocity'].std() * 100

    print(f"Kalman smoothing results:")
    print(f"  Raw altitude std: {df['altitude'].std():.3f} km")
    print(f"  Filtered altitude std: {df['filtered_altitude'].std():.3f} km")
    print(f"  Altitude noise reduction: {altitude_noise_reduction:.1f}%")
    print(f"  Raw velocity std: {df['velocity'].std():.3f} km/h")
    print(f"  Filtered velocity std: {df['filtered_velocity'].std():.3f} km/h")
    print(f"  Velocity noise reduction: {velocity_noise_reduction:.1f}%")

    # Calculate velocity change rate (acceleration) for both raw and filtered data
    acceleration = calculate_velocity_change_rate(df['datetime'].tolist(), df['velocity'].tolist())
    filtered_acceleration = calculate_velocity_change_rate(df['datetime'].tolist(), df['filtered_velocity'].tolist())

    # Filter data for post-September 1st analysis (thruster test period)
    sept_1_2025 = pd.to_datetime('2025-09-01', utc=True)
    df_post_sept1 = df[df['datetime'] >= sept_1_2025].copy()

    if len(df_post_sept1) > 0:
        post_sept1_acceleration = calculate_velocity_change_rate(
            df_post_sept1['datetime'].tolist(),
            df_post_sept1['velocity'].tolist()
        )
        post_sept1_filtered_acceleration = calculate_velocity_change_rate(
            df_post_sept1['datetime'].tolist(),
            df_post_sept1['filtered_velocity'].tolist()
        )
        print(f"Post-Sept 1 data points: {len(df_post_sept1)}")
    else:
        post_sept1_acceleration = np.array([])
        post_sept1_filtered_acceleration = np.array([])
        print("No data found after September 1st")

    # Create the plot with 6 subplots (2x3 layout for better comparison)
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('OTP-2 Satellite Orbital Data Analysis with PyKalman Smoothing', fontsize=16, fontweight='bold')

    # Plot 1: Raw Altitude with Velocity-based Prediction
    ax1.plot(df['datetime'], df['altitude'], 'b-', linewidth=1, marker='o', markersize=3,
             label='Measured Altitude')
    ax1.plot(df['datetime'], df['predicted_altitude'], 'purple', linewidth=2, alpha=0.8,
             label=f'Predicted from Velocity ({best_model_name})')
    ax1.set_ylabel('Altitude (km)', fontweight='bold')
    ax1.set_title('Altitude vs Time: Measured vs Predicted from Velocity')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # Plot 2: PyKalman Smoothed Altitude
    ax2.plot(df['datetime'], df['filtered_altitude'], 'r-', linewidth=2)
    ax2.set_ylabel('Altitude (km)', fontweight='bold')
    ax2.set_title('PyKalman Smoothed Altitude vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Raw vs PyKalman Smoothed Velocity
    ax3.plot(df['datetime'], df['velocity'], 'b-', linewidth=1, marker='o', markersize=2,
             alpha=0.7, label='Raw Data')
    ax3.plot(df['datetime'], df['filtered_velocity'], 'r-', linewidth=2,
             label='PyKalman Smoothed')
    ax3.set_ylabel('Velocity (km/h)', fontweight='bold')
    ax3.set_title('Velocity vs Time (Raw vs PyKalman Smoothed)')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    # Plot 4: Raw Acceleration
    if len(acceleration) > 0:
        ax4.plot(df['datetime'], acceleration, 'g-', linewidth=1, marker='o', markersize=2)
        ax4.set_ylabel('Acceleration (km/h²)', fontweight='bold')
        ax4.set_title('Raw Acceleration vs Time')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.axvline(x=sept_1_2025, color='r', linestyle=':', alpha=0.7)

    # Plot 5: PyKalman Smoothed Acceleration
    if len(filtered_acceleration) > 0:
        ax5.plot(df['datetime'], filtered_acceleration, 'orange', linewidth=2)
        ax5.set_ylabel('Acceleration (km/h²)', fontweight='bold')
        ax5.set_title('PyKalman Smoothed Acceleration vs Time')
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax5.tick_params(axis='x', rotation=45)
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax5.axvline(x=sept_1_2025, color='r', linestyle=':', alpha=0.7,
                   label='Sept 1 (Thruster Test Period)')
        ax5.legend()

    # Plot 6: Focused acceleration analysis (post-September 1st)
    if len(df_post_sept1) > 0 and len(post_sept1_acceleration) > 0:
        # Plot both raw and filtered acceleration
        ax6.plot(df_post_sept1['datetime'], post_sept1_acceleration,
                'lightblue', linewidth=1, marker='o', markersize=3, alpha=0.7,
                label='Raw Acceleration')

        if len(post_sept1_filtered_acceleration) > 0:
            ax6.plot(df_post_sept1['datetime'], post_sept1_filtered_acceleration,
                    'orange', linewidth=2, marker='s', markersize=4,
                    label='PyKalman Smoothed')

        # Use PyKalman smoothed data for anomaly detection (optimal signal with forward-backward pass)
        analysis_data = post_sept1_filtered_acceleration if len(post_sept1_filtered_acceleration) > 0 else post_sept1_acceleration
        window_size = min(5, len(analysis_data))
        if window_size >= 3:
            rolling_mean = pd.Series(analysis_data).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(analysis_data).rolling(window=window_size, center=True).std()

            # Plot rolling mean
            ax6.plot(df_post_sept1['datetime'], rolling_mean,
                    'red', linewidth=2, alpha=0.7, label=f'Rolling Mean ({window_size}pt)')

            # Highlight potential anomalies (>2 standard deviations from rolling mean)
            anomaly_threshold = 2
            upper_bound = rolling_mean + anomaly_threshold * rolling_std
            lower_bound = rolling_mean - anomaly_threshold * rolling_std

            # Fill the normal range
            ax6.fill_between(df_post_sept1['datetime'], lower_bound, upper_bound,
                           alpha=0.2, color='gray', label=f'±{anomaly_threshold}σ range')

            # Mark potential thruster events (acceleration spikes)
            for i, (dt, acc, mean_val, std_val) in enumerate(zip(
                df_post_sept1['datetime'], analysis_data, rolling_mean, rolling_std)):
                if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                    if abs(acc - mean_val) > anomaly_threshold * std_val:
                        ax6.scatter(dt, acc, color='red', s=100, marker='*',
                                  zorder=5, alpha=0.8)

        ax6.set_ylabel('Acceleration (km/h²)', fontweight='bold')
        ax6.set_title('Focused Acceleration Analysis (Post-Sept 1, PyKalman Smoothed)',
                     fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax6.tick_params(axis='x', rotation=45)
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax6.legend()

        # Print potential thruster events
        if window_size >= 3:
            print(f"\nThruster Test Period Analysis (Post-Sept 1):")
            print(f"Data points: {len(df_post_sept1)}")
            print(f"Raw acceleration range: {post_sept1_acceleration.min():.8f} to {post_sept1_acceleration.max():.8f} km/h²")
            print(f"Raw acceleration std dev: {post_sept1_acceleration.std():.8f} km/h²")
            if len(post_sept1_filtered_acceleration) > 0:
                print(f"Filtered acceleration range: {post_sept1_filtered_acceleration.min():.8f} to {post_sept1_filtered_acceleration.max():.8f} km/h²")
                print(f"Filtered acceleration std dev: {post_sept1_filtered_acceleration.std():.8f} km/h²")

            # Find and report significant acceleration events (using analysis_data)
            anomalies = []
            for i, (dt, acc, mean_val, std_val) in enumerate(zip(
                df_post_sept1['datetime'], analysis_data, rolling_mean, rolling_std)):
                if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                    if abs(acc - mean_val) > anomaly_threshold * std_val:
                        anomalies.append((dt, acc, abs(acc - mean_val) / std_val))

            if anomalies:
                print(f"\nPotential thruster events detected ({len(anomalies)} anomalies):")
                for dt, acc, sigma in sorted(anomalies, key=lambda x: x[2], reverse=True):
                    print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S')}: {acc:.8f} km/h² ({sigma:.1f}σ)")
            else:
                print("\nNo significant acceleration anomalies detected")
    else:
        ax6.text(0.5, 0.5, 'No data available\nfor post-Sept 1 analysis',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Focused Acceleration Analysis (Post-Sept 1)')

    # Set x-axis labels for bottom row
    ax4.set_xlabel('Time', fontweight='bold')
    ax5.set_xlabel('Time', fontweight='bold')
    ax6.set_xlabel('Time', fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display statistics
    print(f"\nOverall Data Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"\nVelocity-Altitude Model:")
    print(f"  Best fit model: {best_model_name}")
    print(f"  Model RMSE: {best_rmse:.2f} m/s")
    print(f"  Altitude prediction RMSE: {altitude_prediction_rmse:.3f} km")
    print(f"\nRaw Data:")
    print(f"  Altitude range: {df['altitude'].min():.1f} - {df['altitude'].max():.1f} km")
    print(f"  Predicted altitude range: {df['predicted_altitude'].min():.1f} - {df['predicted_altitude'].max():.1f} km")
    print(f"  Velocity range: {df['velocity'].min():.1f} - {df['velocity'].max():.1f} km/h")
    if len(acceleration) > 0:
        print(f"  Acceleration range: {acceleration.min():.8f} - {acceleration.max():.8f} km/h²")
        print(f"  Average acceleration: {acceleration.mean():.8f} km/h²")
        print(f"  Acceleration std deviation: {acceleration.std():.8f} km/h²")

    print(f"\nPyKalman Smoothed Data:")
    print(f"  Altitude range: {df['filtered_altitude'].min():.1f} - {df['filtered_altitude'].max():.1f} km")
    print(f"  Velocity range: {df['filtered_velocity'].min():.1f} - {df['filtered_velocity'].max():.1f} km/h")
    if len(filtered_acceleration) > 0:
        print(f"  Acceleration range: {filtered_acceleration.min():.8f} - {filtered_acceleration.max():.8f} km/h²")
        print(f"  Average acceleration: {filtered_acceleration.mean():.8f} km/h²")
        print(f"  Acceleration std deviation: {filtered_acceleration.std():.8f} km/h²")

    # Acceleration noise reduction metrics
    if len(acceleration) > 0 and len(filtered_acceleration) > 0:
        acc_noise_reduction = (acceleration.std() - filtered_acceleration.std()) / acceleration.std() * 100
        print(f"\nPyKalman Smoothing Performance:")
        print(f"  Acceleration noise reduction: {acc_noise_reduction:.1f}%")
        print(f"  Forward-backward smoothing provides optimal estimates using complete dataset")

    # Save the plot
    plt.savefig('satellite_analysis_with_thruster.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'satellite_analysis_with_thruster.png'")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
