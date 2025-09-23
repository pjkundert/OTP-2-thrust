#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

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

def main():
    # Read the tab-separated CSV file
    print("Reading data from 2025-052AB-data2.csv...")

    # Read the file with tab separator and proper column names
    df = pd.read_csv('2025-052AB-data2.csv',
                     sep='\t',
                     header=None,
                     names=['satellite', 'timestamp', 'altitude', 'velocity', 'time_ago'])

    print(f"Loaded {len(df)} data points")

    # Parse timestamps
    df['datetime'] = df['timestamp'].apply(parse_timestamp)

    # Sort by timestamp (oldest first)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Calculate velocity change rate (acceleration)
    acceleration = calculate_velocity_change_rate(df['datetime'].tolist(), df['velocity'].tolist())

    # Filter data for post-September 1st analysis (thruster test period)
    sept_1_2025 = pd.to_datetime('2025-09-01', utc=True)
    df_post_sept1 = df[df['datetime'] >= sept_1_2025].copy()

    if len(df_post_sept1) > 0:
        post_sept1_acceleration = calculate_velocity_change_rate(
            df_post_sept1['datetime'].tolist(),
            df_post_sept1['velocity'].tolist()
        )
        print(f"Post-Sept 1 data points: {len(df_post_sept1)}")
    else:
        post_sept1_acceleration = np.array([])
        print("No data found after September 1st")

    # Create the plot with 4 subplots (added focused acceleration plot)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OTP-2 Satellite Orbital Data Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Altitude over time
    ax1.plot(df['datetime'], df['altitude'], 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_ylabel('Altitude (km)', fontweight='bold')
    ax1.set_title('Altitude vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Velocity over time
    ax2.plot(df['datetime'], df['velocity'], 'r-', linewidth=2, marker='o', markersize=4)
    ax2.set_ylabel('Velocity (km/h)', fontweight='bold')
    ax2.set_title('Velocity vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Rate of change in velocity (acceleration) - Full timeline
    if len(acceleration) > 0:
        ax3.plot(df['datetime'], acceleration, 'g-', linewidth=2, marker='o', markersize=4)
        ax3.set_ylabel('Acceleration (km/h²)', fontweight='bold')
        ax3.set_title('Acceleration vs Time (Full Timeline)')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax3.tick_params(axis='x', rotation=45)

        # Add horizontal line at zero for reference
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Mark September 1st as potential thruster test start
        ax3.axvline(x=sept_1_2025, color='r', linestyle=':', alpha=0.7,
                   label='Sept 1 (Thruster Test Period)')
        ax3.legend()

    ax3.set_xlabel('Time', fontweight='bold')

    # Plot 4: Focused acceleration analysis (post-September 1st)
    if len(df_post_sept1) > 0 and len(post_sept1_acceleration) > 0:
        # Main acceleration plot
        ax4.plot(df_post_sept1['datetime'], post_sept1_acceleration,
                'orange', linewidth=2, marker='o', markersize=5, label='Acceleration')

        # Calculate rolling statistics for anomaly detection
        window_size = min(5, len(post_sept1_acceleration))
        if window_size >= 3:
            rolling_mean = pd.Series(post_sept1_acceleration).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(post_sept1_acceleration).rolling(window=window_size, center=True).std()

            # Plot rolling mean
            ax4.plot(df_post_sept1['datetime'], rolling_mean,
                    'red', linewidth=2, alpha=0.7, label=f'Rolling Mean ({window_size}pt)')

            # Highlight potential anomalies (>2 standard deviations from rolling mean)
            anomaly_threshold = 2
            upper_bound = rolling_mean + anomaly_threshold * rolling_std
            lower_bound = rolling_mean - anomaly_threshold * rolling_std

            # Fill the normal range
            ax4.fill_between(df_post_sept1['datetime'], lower_bound, upper_bound,
                           alpha=0.2, color='gray', label=f'±{anomaly_threshold}σ range')

            # Mark potential thruster events (acceleration spikes)
            for i, (dt, acc, mean_val, std_val) in enumerate(zip(
                df_post_sept1['datetime'], post_sept1_acceleration, rolling_mean, rolling_std)):
                if not pd.isna(mean_val) and not pd.isna(std_val) and std_val > 0:
                    if abs(acc - mean_val) > anomaly_threshold * std_val:
                        ax4.scatter(dt, acc, color='red', s=100, marker='*',
                                  zorder=5, alpha=0.8)

        ax4.set_ylabel('Acceleration (km/h²)', fontweight='bold')
        ax4.set_title('Focused Acceleration Analysis (Post-Sept 1, Thruster Test Period)',
                     fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax4.tick_params(axis='x', rotation=45)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.legend()

        # Print potential thruster events
        if window_size >= 3:
            print(f"\nThruster Test Period Analysis (Post-Sept 1):")
            print(f"Data points: {len(df_post_sept1)}")
            print(f"Acceleration range: {post_sept1_acceleration.min():.8f} to {post_sept1_acceleration.max():.8f} km/h²")
            print(f"Acceleration std dev: {post_sept1_acceleration.std():.8f} km/h²")

            # Find and report significant acceleration events
            anomalies = []
            for i, (dt, acc, mean_val, std_val) in enumerate(zip(
                df_post_sept1['datetime'], post_sept1_acceleration, rolling_mean, rolling_std)):
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
        ax4.text(0.5, 0.5, 'No data available\nfor post-Sept 1 analysis',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Focused Acceleration Analysis (Post-Sept 1)')

    ax4.set_xlabel('Time', fontweight='bold')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display statistics
    print(f"\nOverall Data Statistics:")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['datetime'].min().strftime('%Y-%m-%d %H:%M')} to {df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    print(f"Altitude range: {df['altitude'].min():.1f} - {df['altitude'].max():.1f} km")
    print(f"Velocity range: {df['velocity'].min():.1f} - {df['velocity'].max():.1f} km/h")
    if len(acceleration) > 0:
        print(f"Acceleration range: {acceleration.min():.8f} - {acceleration.max():.8f} km/h²")
        print(f"Average acceleration: {acceleration.mean():.8f} km/h²")
        print(f"Acceleration std deviation: {acceleration.std():.8f} km/h²")

    # Save the plot
    plt.savefig('satellite_analysis_with_thruster.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'satellite_analysis_with_thruster.png'")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()