# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:22:05 2025

@author: mtm4h
"""

import numpy as np
import scipy.signal as sig

def create_bpsk_jammer_iq(bit_duration_seconds=8.0, sampling_rate_hz=32000.0, bandwidth_hz=4000.0, pulse_width_microseconds=1000.0):
    """
    Generates an IQ file for a baseband BPSK jammer signal.

    Args:
        bit_duration_seconds (float): Duration of each bit in the BPSK signal (in seconds).
        sampling_rate_hz (float): Sampling rate for the generated IQ signal (in Hz).
        bandwidth_hz (float): Bandwidth of the jamming signal (in Hz).
        pulse_width_microseconds (float): Duration of the jamming pulse within each bit period (in microseconds).

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - i_data (numpy.ndarray): In-phase (I) component of the jamming signal.
               - q_data (numpy.ndarray): Quadrature (Q) component of the jamming signal (all zeros for baseband real signal).
    """

    pulse_width_seconds = pulse_width_microseconds / 1e6

    if pulse_width_seconds > bit_duration_seconds:
        raise ValueError("Pulse width cannot be greater than the bit duration.")
    if pulse_width_seconds <= 0:
        raise ValueError("Pulse width must be a positive value.")

    samples_per_bit = int(sampling_rate_hz * bit_duration_seconds)
    samples_per_pulse = int(sampling_rate_hz * pulse_width_seconds)
    silence_samples = samples_per_bit - samples_per_pulse

    # Generate a pseudo-random sequence of bits for the jammer
    num_bits = 10  # Generate a few bits for demonstration
    jammer_bits = np.random.randint(0, 2, num_bits) * 2 - 1  # Generates -1 and 1

    i_data = np.array([])
    for bit in jammer_bits:
        # Create a pulse for the current bit
        pulse = bit * np.ones(samples_per_pulse)
        # Add silence after the pulse
        silence = np.zeros(silence_samples)
        i_data = np.concatenate((i_data, pulse, silence))

    # The jamming signal is at baseband, so the Q component is zero
    q_data = np.zeros_like(i_data)

    print(f"Generated IQ data with {len(i_data)} samples.")
    print(f"Each bit of the target BPSK signal is {bit_duration_seconds} seconds long ({samples_per_bit} samples).")
    print(f"Jamming pulse width: {pulse_width_microseconds} microseconds ({samples_per_pulse} samples).")
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
    iq_data[0::2] = i_data.astype(np.float32)
    iq_data[1::2] = q_data.astype(np.float32)
    iq_data.tofile(filename)
    print(f"IQ data saved to '{filename}' as interleaved floats.")

if __name__ == "__main__":
    pulse_width_input_us = float(input("Enter the pulse width for the jamming signal (in microseconds): "))

    # Define parameters
    bit_duration = 8.0  # seconds
    sampling_rate = 32000.0  # Hz
    bandwidth = 4000.0  # Hz
    output_filename = "mike_bpsk_jammer.iq"

    # Generate the IQ data
    i_data, q_data = create_bpsk_jammer_iq(bit_duration, sampling_rate, bandwidth, pulse_width_input_us)

    # Save the IQ data to a file
    save_iq_to_file(output_filename, i_data, q_data)

    print("Script finished.")