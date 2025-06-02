import numpy as np
import matplotlib.pyplot as plt
import time  # To estimate runtime

import nengo  # Only for NeuronType and its dependencies if not fully standalone
from nengo.params import (
    NumberParam, IntParam, StringParam, BoolParam, Unconfigurable
)
from nengo.neurons import NeuronType, settled_firingrate
from nengo.dists import Choice


# Note: No need for scipy.optimize.brentq in this specific F-I plotting script

# --- Definition of your SCTNNeuronType (from your previous complete script) ---
class SCTNNeuronType(NeuronType):
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    theta = NumberParam("theta", default=0.0, readonly=True)
    reset_to = NumberParam("reset_to", default=0.0, readonly=True)
    min_clip = NumberParam("min_clip", default=-524287.0, readonly=True)
    max_clip = NumberParam("max_clip", default=524287.0, readonly=True)
    leakage_factor = IntParam("leakage_factor", default=0, low=0, readonly=True)
    leakage_period = IntParam("leakage_period", default=1, low=1, readonly=True)
    threshold_pulse = NumberParam("threshold_pulse", default=0.0, readonly=True)
    activation_function_str = StringParam("activation_function_str", default="IDENTITY", readonly=True)
    membrane_should_reset = BoolParam("membrane_should_reset", default=True, readonly=True)
    amplitude = NumberParam("amplitude", default=1.0, low=0.000001, readonly=True)
    identity_const = NumberParam("identity_const", default=32767.0, readonly=True)
    gaussian_rand_order = IntParam("gaussian_rand_order", default=8, low=1, readonly=True)

    state = {
        "voltage": Choice([0.0]),
        "leakage_timer": Choice([0.0]),
        "rand_gauss_var_state": Choice([0.0]),
        "pn_generator_state": Choice([1.0]),
    }
    spiking = True

    def __init__(self,
                 theta=Unconfigurable,
                 reset_to=Unconfigurable,
                 min_clip=Unconfigurable,
                 max_clip=Unconfigurable,
                 leakage_factor=Unconfigurable,
                 leakage_period=Unconfigurable,
                 threshold_pulse=Unconfigurable,
                 activation_function="IDENTITY",
                 membrane_should_reset=Unconfigurable,
                 amplitude=1.0,
                 identity_const=Unconfigurable,
                 gaussian_rand_order=Unconfigurable,
                 initial_state=None):
        super().__init__(initial_state=initial_state)

        if theta is not Unconfigurable: self.theta = float(theta)
        if reset_to is not Unconfigurable: self.reset_to = float(reset_to)
        if min_clip is not Unconfigurable: self.min_clip = float(min_clip)
        if max_clip is not Unconfigurable: self.max_clip = float(max_clip)
        if leakage_factor is not Unconfigurable: self.leakage_factor = int(leakage_factor)
        if leakage_period is not Unconfigurable: self.leakage_period = int(leakage_period)
        if threshold_pulse is not Unconfigurable: self.threshold_pulse = float(threshold_pulse)

        self.activation_function_str = str(activation_function)

        if membrane_should_reset is not Unconfigurable: self.membrane_should_reset = bool(membrane_should_reset)
        if amplitude is not Unconfigurable: self.amplitude = float(amplitude)
        if self.amplitude <= 1e-9:
            self.amplitude = 1e-9

        if identity_const is not Unconfigurable: self.identity_const = float(identity_const)
        if gaussian_rand_order is not Unconfigurable: self.gaussian_rand_order = int(gaussian_rand_order)

        act_str_upper = self.activation_function_str.upper()
        if act_str_upper == "BINARY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_BINARY
        elif act_str_upper == "IDENTITY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_IDENTITY
        elif act_str_upper == "SIGMOID":
            self.current_activation_fn_id = SCTNNeuronType.ACT_SIGMOID
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function_str}.")

        if self.leakage_period < 1:
            self.leakage_period = 1

    # We only need _get_rate_for_NEF_current and its dependency (step, make_state)
    # The full gain_bias and rates methods are not strictly needed for this F-I plotting script.

    def step(self, dt, J_nef, output, voltage, leakage_timer, rand_gauss_var_state, pn_generator_state):
        current_total = J_nef + self.theta  # J_nef is the direct current here, theta is added

        voltage_arr = np.asarray(voltage)
        leakage_timer_arr = np.asarray(leakage_timer)
        rand_gauss_var_state_arr = np.asarray(rand_gauss_var_state)
        pn_generator_state_arr = np.asarray(pn_generator_state).astype(np.int32)
        # J_nef is already the current for this neuron, ensure it's an array if step handles multiple
        J_nef_arr = np.asarray(J_nef)
        output_arr = np.asarray(output)

        effective_input_current = np.zeros_like(J_nef_arr)
        current_leakage_factor_val = int(self.leakage_factor)

        if current_leakage_factor_val < 3:
            effective_input_current[:] = current_total
        else:
            lf_scale = float(2 ** (current_leakage_factor_val - 3))
            effective_input_current[:] = current_total * lf_scale

        voltage_arr += effective_input_current * dt
        voltage_arr[:] = np.clip(voltage_arr, self.min_clip, self.max_clip)

        emit_spike_flags = np.zeros_like(voltage_arr, dtype=bool)

        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            emit_spike_flags[:] = voltage_arr > self.threshold_pulse

        elif self.current_activation_fn_id == SCTNNeuronType.ACT_IDENTITY:
            const_val = float(self.identity_const)
            mask_gt_const = voltage_arr > const_val
            emit_spike_flags[mask_gt_const] = True
            rand_gauss_var_state_arr[mask_gt_const] = const_val

            mask_lt_neg_const = voltage_arr < -const_val
            rand_gauss_var_state_arr[mask_lt_neg_const] = const_val

            mask_else = ~(mask_gt_const | mask_lt_neg_const)
            if np.any(mask_else):
                c_val_else = voltage_arr[mask_else] + const_val
                increment_else = c_val_else + 1.0
                rgv_subset = rand_gauss_var_state_arr[mask_else].copy()
                rgv_subset += increment_else

                overflow_mask_subset = rgv_subset >= 65536.0
                if np.any(overflow_mask_subset):
                    rgv_subset[overflow_mask_subset] = np.mod(rgv_subset[overflow_mask_subset], 65536.0)
                    original_else_indices = np.where(mask_else)[0]
                    spiking_else_indices = original_else_indices[overflow_mask_subset]
                    emit_spike_flags[spiking_else_indices] = True

                rand_gauss_var_state_arr[mask_else] = rgv_subset

        elif self.current_activation_fn_id == SCTNNeuronType.ACT_SIGMOID:
            sum_for_sigmoid = np.zeros_like(voltage_arr)
            temp_pn_gen = np.copy(pn_generator_state_arr)

            for _ in range(int(self.gaussian_rand_order)):
                sum_for_sigmoid += temp_pn_gen & 0x1fff
                feedback_bit_14 = (temp_pn_gen >> 14) & 1
                feedback_bit_0 = temp_pn_gen & 1
                new_msb = feedback_bit_14 ^ feedback_bit_0
                temp_pn_gen = (temp_pn_gen >> 1) | (new_msb << 14)
                temp_pn_gen &= 0x7FFF

            pn_generator_state_arr[:] = temp_pn_gen
            rand_gauss_var_state_arr[:] = sum_for_sigmoid
            emit_spike_flags[:] = voltage_arr > rand_gauss_var_state_arr

        if current_leakage_factor_val > 0:
            current_leakage_period_val = int(self.leakage_period)
            leak_now_mask = leakage_timer_arr >= current_leakage_period_val

            if np.any(leak_now_mask):
                decay_multiplier = (1.0 - (1.0 / (2 ** current_leakage_factor_val)))
                voltage_arr[leak_now_mask] *= decay_multiplier
                leakage_timer_arr[leak_now_mask] = 0.0

            leakage_timer_arr[~leak_now_mask] += 1.0

        output_arr[:] = 0.0
        output_arr[emit_spike_flags] = self.amplitude / dt

        if self.membrane_should_reset:
            voltage_arr[emit_spike_flags] = self.reset_to

        voltage[:] = voltage_arr
        leakage_timer[:] = leakage_timer_arr
        rand_gauss_var_state[:] = rand_gauss_var_state_arr
        pn_generator_state[:] = pn_generator_state_arr.astype(float)
        output[:] = output_arr

    def _get_rate_for_NEF_current(self, J_nef, dt_sim=0.001, settle_time=0.3, sim_time=1.5):
        J_total_for_step = J_nef + self.theta
        J_array = np.array([J_total_for_step], dtype=float)  # settled_firingrate expects array

        # For this isolated test, make_state needs a dummy rng
        rng_for_calc = np.random.RandomState(seed=12345)
        sim_state_single = self.make_state(n_neurons=1, rng=rng_for_calc, dtype=J_array.dtype)
        if "output" not in sim_state_single:  # Ensure 'output' key exists
            sim_state_single["output"] = np.zeros(1, dtype=J_array.dtype)

        # Optional: print progress for long simulations
        print(f"    Calculating rate for J_nef = {J_nef:.2f} (J_total = {J_total_for_step:.2f}) "
              f"with sim_time = {sim_time}s...")

        rate = settled_firingrate(
            self.step,
            J_array,
            sim_state_single,
            dt=dt_sim,
            settle_time=settle_time,
            sim_time=sim_time
        )
        calculated_rate = rate[0] if hasattr(rate, 'size') and rate.size > 0 else 0.0
        print(f"    J_nef = {J_nef:.2f} -> Rate = {calculated_rate:.2f} Hz")
        return calculated_rate


# --- Script to Plot F-I Curve ---
if __name__ == "__main__":
    print("Starting F-I Curve Plotting Script for SCTNNeuronType (IDENTITY)...")

    # --- Neuron Parameters to Test ---
    test_identity_const = 10  # The value from your last test
    # test_identity_const = 100.0 # Example of a smaller value to try
    test_theta = 0.05
    test_leakage_factor = 2
    test_leakage_period = 1
    test_amplitude = 1.0

    # Accuracy parameters for _get_rate_for_NEF_current
    # Using longer times for better accuracy as requested
    param_dt_sim = 0.001
    param_settle_time = 0.3  # s
    param_sim_time = 1.5  # s
    print(f"Using settle_time={param_settle_time}s, sim_time={param_sim_time}s for rate calculations.")

    # Instantiate the neuron
    neuron_to_test = SCTNNeuronType(
        activation_function="SIGMOID",
        identity_const=test_identity_const,
        theta=test_theta,
        leakage_factor=test_leakage_factor,
        leakage_period=test_leakage_period,
        amplitude=test_amplitude
    )
    print(f"Testing Neuron: IDENTITY, id_const={test_identity_const}, theta={test_theta}")

    # --- Define J_nef range based on identity_const ---
    # This is the wide range you requested for exploration
    min_J_nef =0
    max_J_nef = 2500
    num_J_points = 1000  # Number of current levels to test (reduce if too slow)

    # If identity_const is very small (e.g. 0), the range above might be too small or zero.
    if abs(max_J_nef - min_J_nef) < 1.0:  # If range is too small, use a default range
        print(f"Warning: Calculated J_nef range is too small or zero based on identity_const={test_identity_const}.")
        min_J_nef = -50.0  # Default fallback range
        max_J_nef = 50.0
        print(f"Using fallback J_nef range: [{min_J_nef}, {max_J_nef}]")
        if test_identity_const == 0:
            print(
                "Note: identity_const is 0, so 'voltage > const_val' and 'voltage < -const_val' might behave like 'voltage > 0' and 'voltage < 0'.")

    J_nef_values = np.linspace(min_J_nef, max_J_nef, num_J_points)
    rates_out = np.zeros_like(J_nef_values)

    print(f"\nCalculating F-I curve for {num_J_points} J_nef points from {min_J_nef:.2f} to {max_J_nef:.2f}...")
    start_calc_time = time.time()

    for i, j_nef in enumerate(J_nef_values):
        rates_out[i] = neuron_to_test._get_rate_for_NEF_current(
            j_nef,
            dt_sim=param_dt_sim,
            settle_time=param_settle_time,
            sim_time=param_sim_time
        )

    end_calc_time = time.time()
    print(f"F-I curve calculation finished in {end_calc_time - start_calc_time:.2f} seconds.")

    # --- Plotting the F-I Curve ---
    plt.figure(figsize=(12, 7))
    plt.plot(J_nef_values, rates_out, marker='o', linestyle='-')
    plt.title(f"F-I Curve for SCTNNeuronType (IDENTITY)\n"
              f"id_const={test_identity_const}, theta={test_theta}, "
              f"lk_factor={test_leakage_factor}, lk_period={test_leakage_period}\n"
              f"(settle_time={param_settle_time}s, sim_time={param_sim_time}s per point)")
    plt.xlabel("J_nef (Input Current before intrinsic theta)")
    plt.ylabel("Firing Rate (Hz)")
    plt.grid(True)
    plt.axhline(0, color='black', lw=0.5)  # Zero rate line
    plt.axvline(0, color='black', lw=0.5)  # Zero current line

    # Annotate min/max rates found
    min_rate_found = rates_out.min()
    max_rate_found = rates_out.max()
    plt.figtext(0.01, 0.01, f"Min rate found: {min_rate_found:.2f} Hz, Max rate found: {max_rate_found:.2f} Hz "
                            f"over J_nef range [{min_J_nef:.2f}, {max_J_nef:.2f}]")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

    print("\n--- F-I Curve Plotting Script Finished ---")
    print("Analyze the plot to determine:")
    print("1. The actual range of J_nef where the neuron's firing rate is modulated.")
    print("2. The J_nef value where firing starts (e.g., > 0.1 Hz).")
    print("3. The J_nef value where the firing rate saturates or reaches typical max_rates (e.g., 50-150 Hz).")
    print("4. If the curve is monotonic and suitable for brentq root-finding.")
    print("Use this information to set accurate J_nef_search_min/max_default in your main gain_bias method.")