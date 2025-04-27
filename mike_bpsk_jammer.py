# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:22:05 2025

@author: mtm4h
"""

import numpy as np
import scipy.signal as sig

def create_bpsk_jammer_iq_duty_cycle(bit_duration_seconds=0.00025, sampling_rate_hz=32000.0, bandwidth_hz=4000.0, pulse_width_microseconds=1000.0, duty_cycle=0.5):
    """
    Generates an IQ file for a baseband BPSK jammer signal with a duty cycle.

    Args:
        bit_duration_seconds (float): Duration of each bit in the BPSK signal (in seconds).
        sampling_rate_hz (float): Sampling rate for the generated IQ signal (in Hz).
        bandwidth_hz (float): Bandwidth of the jamming signal (in Hz).
        pulse_width_microseconds (float): Duration of the jamming pulse within each bit period (in microseconds).
        duty_cycle (float): The fraction of the bit duration for which the jammer is active (0.0 to 1.0).

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - i_data (numpy.ndarray): In-phase (I) component of the jamming signal.
               - q_data (numpy.ndarray): Quadrature (Q) component of the jamming signal (all zeros for baseband real signal).
    """

    pulse_width_seconds = pulse_width_microseconds / 1e6

    if pulse_width_seconds > bit_duration_seconds * duty_cycle:
        raise ValueError("Pulse width cannot be greater than the active portion of the bit duration (based on duty cycle).")
    if pulse_width_seconds <= 0 or duty_cycle <= 0 or duty_cycle > 1.0:
        raise ValueError("Pulse width and duty cycle must be positive, and duty cycle must be between 0.0 and 1.0.")

    samples_per_bit = int(sampling_rate_hz * bit_duration_seconds)
    samples_per_pulse = int(sampling_rate_hz * pulse_width_seconds)
    active_samples = int(sampling_rate_hz * bit_duration_seconds * duty_cycle)
    silence_samples = samples_per_bit - active_samples
    silence_after_pulse = active_samples - samples_per_pulse

    # Generate a pseudo-random sequence of bits for the jammer
    num_bits = 100  # Generate a larger number of bits for a more continuous jammer
    jammer_bits = np.random.randint(0, 2, num_bits) * 2 - 1  # Generates -1 and 1

    i_data = np.array([])
    for bit in jammer_bits:
        # Create a pulse for the current bit
        pulse = bit * np.ones(samples_per_pulse)
        # Add silence after the pulse within the active period
        silence_within = np.zeros(silence_after_pulse)
        # Add silence for the inactive period
        silence_inactive = np.zeros(silence_samples)

        i_data = np.concatenate((i_data, pulse, silence_within, silence_inactive))

    # The jamming signal is at baseband, so the Q component is zero
    q_data = np.zeros_like(i_data)

    print(f"Generated IQ data with {len(i_data)} samples.")
    print(f"Each bit of the target BPSK signal is {bit_duration_seconds} seconds long ({samples_per_bit} samples).")
    print(f"Jamming pulse width: {pulse_width_microseconds} microseconds ({samples_per_pulse} samples).")
    print(f"Duty cycle: {duty_cycle * 100:.2f}% (active for {bit_duration_seconds * duty_cycle:.6f} seconds per bit).")
    print(f"Sampling rate: {sampling_rate_hz} Hz.")
    print(f"Intended bandwidth: {bandwidth_hz} Hz.")

    return i_data, q_data

def save_iq_to_file(filename, i_data, q_data):
    """
    Saves the I and Q data to a binary file.

    Args:
        filename (str): The name of the file to save to.
        i_data (numpy.ndarray): In-phase (I) component of the signal.
        q_data (numpy.ndarray): Quadrature (Q) component of the signal.
    """
    iq_data = np.empty(i_data.shape[0] + q_data.shape[0], dtype=np.float32)
    iq_data[0::2] = i_data.astype(np.float32) #Assigns the i_data (converted to np.float32) to the even indices of iq_data.
    iq_data[1::2] = q_data.astype(np.float32) #Assigns the q_data (converted to np.float32) to the odd indices of iq_data. This interleaves the I and Q samples.
    iq_data.tofile(filename)
    print(f"IQ data saved to '{filename}' as interleaved floats.")

if __name__ == "__main__": #This block of code will only execute when the script is run directly (not when imported as a module).100

    pulse_width_input_us = float(input("Enter the pulse width for the jamming signal (in microseconds): "))
    duty_cycle_input = float(input("Enter the duty cycle for the jamming signal (0.0 to 1.0): "))

    # Define parameters based on 8 samples/bit at 32k sps
    bit_duration = 8.0 / 32000.0  # 0.00025 seconds
    sampling_rate = 32000.0  # Hz
    bandwidth = 4000.0  # Hz
    output_filename = "mike_bpsk_jammer.iq"

    # Generate the IQ data with duty cycling
    i_data, q_data = create_bpsk_jammer_iq_duty_cycle(bit_duration, sampling_rate, bandwidth, pulse_width_input_us, duty_cycle_input)

    # Save the IQ data to a file
    save_iq_to_file(output_filename, i_data, q_data)

    print("Script finished.")