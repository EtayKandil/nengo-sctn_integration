import numpy as np
import matplotlib.pyplot as plt
import time

import nengo
from nengo.params import (
    NumberParam, IntParam, StringParam, BoolParam, Unconfigurable
)
from nengo.neurons import NeuronType, settled_firingrate
from nengo.dists import Choice, Uniform
from nengo.exceptions import ValidationError
from scipy.optimize import brentq  # For the custom gain_bias


# --- Definition of your SCTNNeuronType ---
class SCTNNeuronType(NeuronType):
    """
    A custom, highly configurable spiking neuron model for the Nengo framework.

    This neuron model extends Nengo's base NeuronType to include several
    unique features, such as multiple activation functions, a configurable
    leakage mechanism, and a custom gain-bias calculation process. It is
    designed to simulate complex neural dynamics that can be mapped to
    specialized hardware like FPGAs.

    Key Features:
    - Multiple activation functions: BINARY, IDENTITY, SIGMOID.
    - A unique leakage mechanism controlled by `leakage_factor` and `leakage_period`.
    - Custom `gain_bias` method that empirically determines the neuron's
      tuning curve using a root-finding algorithm, which is more robust for
      complex neuron models than the default analytical approach.
    """
    # --- Activation Function IDs ---
    # Static constants to represent the different activation functions.
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    # --- Nengo Parameters ---
    # These parameters define the neuron's properties and are exposed to the Nengo
    # build system. They are declared as readonly=True because they are intended
    # to be set at initialization and not changed during the simulation.
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

    # --- Nengo State Variables ---
    # Defines the state variables that need to be tracked for each neuron
    # during the simulation.
    state = {
        "voltage": Choice([0.0]),
        "leakage_timer": Choice([0.0]),
        "rand_gauss_var_state": Choice([0.0]),
        "pn_generator_state": Choice([1.0]),
    }
    spiking = True  # Indicates to Nengo that this is a spiking neuron model.

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
        """Initializes the neuron's parameters."""
        super().__init__(initial_state=initial_state)

        # Set parameters only if they are explicitly provided, otherwise they
        # will keep their default values defined in the class.
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
            print(f"Warning: Amplitude {self.amplitude} is too small, setting to 1e-9.")
            self.amplitude = 1e-9

        if identity_const is not Unconfigurable: self.identity_const = float(identity_const)
        if gaussian_rand_order is not Unconfigurable: self.gaussian_rand_order = int(gaussian_rand_order)

        # Set an integer ID for the current activation function for efficient
        # switching within the `step` method.
        act_str_upper = self.activation_function_str.upper()
        if act_str_upper == "BINARY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_BINARY
        elif act_str_upper == "IDENTITY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_IDENTITY
        elif act_str_upper == "SIGMOID":
            self.current_activation_fn_id = SCTNNeuronType.ACT_SIGMOID
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function_str}.")

        # Validate leakage_period to prevent division by zero or invalid behavior.
        if self.leakage_period < 1:
            print(f"Warning: leakage_period {self.leakage_period} is < 1. Setting to 1.")
            self.leakage_period = 1

    def rates(self, x, gain, bias, dt=0.001):
        """
        Calculates the steady-state firing rates for given inputs.

        This method is used by the Nengo builder to compute decoders. It uses
        the `settled_firingrate` utility, which repeatedly calls the `step`
        method to find the average firing rate for a constant input current.
        """
        # Calculate the total input current J for the given encoded values 'x'.
        J_for_step_method = self.current(x, gain, bias)

        # Define simulation parameters for finding the settled rate.
        settle_time_for_rates = 0.02
        sim_time_for_rates = 0.2

        J_for_step_method = np.asarray(J_for_step_method)
        if J_for_step_method.ndim == 0:
            J_for_step_method = J_for_step_method[np.newaxis]

        # Handle single vector of currents (for one set of inputs).
        if J_for_step_method.ndim == 1:
            n_neurons_calc = J_for_step_method.shape[0]
            rng_for_rates = np.random.RandomState(seed=12345)
            sim_state_calc = self.make_state(n_neurons=n_neurons_calc, rng=rng_for_rates, dtype=J_for_step_method.dtype)
            if "output" not in sim_state_calc: sim_state_calc["output"] = np.zeros(n_neurons_calc,
                                                                                   dtype=J_for_step_method.dtype)
            return settled_firingrate(
                self.step, J_for_step_method, sim_state_calc, dt=dt,
                settle_time=settle_time_for_rates, sim_time=sim_time_for_rates)
        # Handle a matrix of currents (for multiple sets of inputs).
        elif J_for_step_method.ndim == 2:
            n_samples, n_neurons_calc = J_for_step_method.shape
            output_rates = np.zeros_like(J_for_step_method)
            rng_for_rates = np.random.RandomState(seed=12345)
            for i in range(n_samples):
                J_sample = J_for_step_method[i, :]
                sim_state_calc = self.make_state(n_neurons=n_neurons_calc, rng=rng_for_rates, dtype=J_sample.dtype)
                if "output" not in sim_state_calc: sim_state_calc["output"] = np.zeros(n_neurons_calc,
                                                                                       dtype=J_sample.dtype)
                output_rates[i, :] = settled_firingrate(
                    self.step, J_sample, sim_state_calc, dt=dt,
                    settle_time=settle_time_for_rates, sim_time=sim_time_for_rates)
            return output_rates
        else:
            raise ValueError(f"Unexpected J dimensions in rates: {J_for_step_method.ndim}")

    def step(self, dt, J_nef, output, voltage, leakage_timer, rand_gauss_var_state, pn_generator_state):
        """
        Simulates one time step of the SCTN neuron dynamics.
        """
        # --- 1. Calculate Total and Effective Input Current ---
        current_total = J_nef + self.theta

        # Ensure all state variables are numpy arrays for vectorized operations.
        voltage_arr = np.asarray(voltage)
        leakage_timer_arr = np.asarray(leakage_timer)
        rand_gauss_var_state_arr = np.asarray(rand_gauss_var_state)
        pn_generator_state_arr = np.asarray(pn_generator_state).astype(np.int32)
        J_nef_arr = np.asarray(J_nef)
        output_arr = np.asarray(output)

        # Scale the input current based on the leakage_factor. This is a custom
        # mechanism that can amplify or attenuate the input.
        effective_input_current = np.zeros_like(J_nef_arr)
        current_leakage_factor_val = int(self.leakage_factor)

        if current_leakage_factor_val < 3:
            effective_input_current[:] = current_total
        else:
            lf_scale = float(2 ** (current_leakage_factor_val - 3))
            effective_input_current[:] = current_total * lf_scale

        # --- 2. Integrate Membrane Voltage ---
        voltage_arr += effective_input_current * dt
        voltage_arr[:] = np.clip(voltage_arr, self.min_clip, self.max_clip)

        # --- 3. Spike Generation based on Activation Function ---
        emit_spike_flags = np.zeros_like(voltage_arr, dtype=bool)

        # A. BINARY activation: Spike if voltage exceeds a fixed threshold.
        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            emit_spike_flags[:] = voltage_arr > self.threshold_pulse

        # B. IDENTITY activation: A probabilistic spiking mechanism.
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

                # Check for overflow, which triggers a spike.
                overflow_mask_subset = rgv_subset >= 65536.0
                if np.any(overflow_mask_subset):
                    rgv_subset[overflow_mask_subset] = np.mod(rgv_subset[overflow_mask_subset], 65536.0)
                    original_else_indices = np.where(mask_else)[0]
                    spiking_else_indices = original_else_indices[overflow_mask_subset]
                    emit_spike_flags[spiking_else_indices] = True

                rand_gauss_var_state_arr[mask_else] = rgv_subset

        # C. SIGMOID activation: Spike if voltage exceeds a pseudo-random threshold.
        elif self.current_activation_fn_id == SCTNNeuronType.ACT_SIGMOID:
            sum_for_sigmoid = np.zeros_like(voltage_arr)
            temp_pn_gen = np.copy(pn_generator_state_arr)

            # Generate a pseudo-random number by summing outputs of a Linear
            # Feedback Shift Register (LFSR). This approximates a Gaussian distribution.
            for _ in range(int(self.gaussian_rand_order)):
                sum_for_sigmoid += temp_pn_gen & 0x1fff
                feedback_bit_14 = (temp_pn_gen >> 14) & 1
                feedback_bit_0 = temp_pn_gen & 1
                new_msb = feedback_bit_14 ^ feedback_bit_0
                temp_pn_gen = (temp_pn_gen >> 1) | (new_msb << 14)
                temp_pn_gen &= 0x7FFF

            sum_for_sigmoid -= 32768  # Center the distribution around zero
            pn_generator_state_arr[:] = temp_pn_gen
            rand_gauss_var_state_arr[:] = sum_for_sigmoid
            emit_spike_flags[:] = voltage_arr > rand_gauss_var_state_arr

        # --- 4. Apply Leakage ---
        if current_leakage_factor_val > 0:
            current_leakage_period_val = int(self.leakage_period)
            leak_now_mask = leakage_timer_arr >= current_leakage_period_val

            if np.any(leak_now_mask):
                decay_multiplier = (1.0 - (1.0 / (2 ** current_leakage_factor_val)))
                voltage_arr[leak_now_mask] *= decay_multiplier
                leakage_timer_arr[leak_now_mask] = 0.0

            leakage_timer_arr[~leak_now_mask] += 1.0

        # --- 5. Set Output and Reset Voltage ---
        output_arr[:] = 0.0
        output_arr[emit_spike_flags] = self.amplitude / dt  # Output is a spike impulse

        if self.membrane_should_reset:
            voltage_arr[emit_spike_flags] = self.reset_to

        # --- 6. Update State Variables ---
        # Copy the modified local arrays back to the Nengo state dictionary views.
        voltage[:] = voltage_arr
        leakage_timer[:] = leakage_timer_arr
        rand_gauss_var_state[:] = rand_gauss_var_state_arr
        pn_generator_state[:] = pn_generator_state_arr.astype(float)
        output[:] = output_arr

    # --- Custom gain_bias and helpers ---
    def _get_rate_for_NEF_current(self, J_nef, dt_sim=0.001, settle_time=0.3, sim_time=1.5):
        """
        Helper function to calculate the firing rate for a single J_nef value.
        It runs a mini-simulation to find the settled firing rate.
        """
        J_total_for_step = J_nef + self.theta
        J_array = np.array([J_total_for_step], dtype=float)
        rng_for_calc = np.random.RandomState(seed=12345)
        sim_state_single = self.make_state(n_neurons=1, rng=rng_for_calc, dtype=J_array.dtype)
        if "output" not in sim_state_single:
            sim_state_single["output"] = np.zeros(1, dtype=J_array.dtype)

        if sim_time > 0.5:
            print(f"    _get_rate_for_NEF_current: J_nef={J_nef:.2f}, sim_time={sim_time}s (this may take a moment)...")

        rate = settled_firingrate(
            self.step, J_array, sim_state_single, dt=dt_sim,
            settle_time=settle_time, sim_time=sim_time
        )
        calculated_rate = rate[0] if hasattr(rate, 'size') and rate.size > 0 else 0.0
        if sim_time > 0.5:
            print(f"    _get_rate_for_NEF_current: J_nef={J_nef:.2f} -> rate={calculated_rate:.2f} Hz")
        return calculated_rate

    def _find_J_nef_for_target_rate(self, target_rate, dt_sim,
                                    J_nef_search_min, J_nef_search_max,
                                    settle_time_gb=0.3, sim_time_gb=1.5,
                                    xtol=1e-4, rtol=1e-4):
        """
        Uses a numerical solver (Brent's method) to find the input current J_nef
        that results in a specific `target_rate`. This is the inverse of the
        neuron's response function.
        """
        # Define the objective function for the root-finder: f(J) = rate(J) - target_rate
        objective_fn = lambda J_nef_candidate: self._get_rate_for_NEF_current(
            J_nef_candidate, dt_sim, settle_time_gb, sim_time_gb
        ) - target_rate

        # Evaluate the function at the search bounds to ensure the root is bracketed.
        rate_at_min = self._get_rate_for_NEF_current(J_nef_search_min, dt_sim, settle_time_gb, sim_time_gb)
        rate_at_max = self._get_rate_for_NEF_current(J_nef_search_max, dt_sim, settle_time_gb, sim_time_gb)
        obj_at_min = rate_at_min - target_rate
        obj_at_max = rate_at_max - target_rate

        print(
            f"  _find_J_nef: Target={target_rate:.2f}Hz. Range J_nef=[{J_nef_search_min:.2f} (rate {rate_at_min:.2f}Hz), "
            f"{J_nef_search_max:.2f} (rate {rate_at_max:.2f}Hz)]")

        try:
            # Handle cases where the root is not bracketed by the search range.
            if np.sign(obj_at_min) == np.sign(obj_at_max) and abs(obj_at_min) > xtol and abs(obj_at_max) > xtol:
                failing_print_str = (f"Warning: Root for target rate {target_rate:.2f} Hz not bracketed. ")
                if obj_at_min > 0:
                    print(
                        failing_print_str + f"Target likely at or below J_min. Returning J_min ({J_nef_search_min:.2f}).")
                    return J_nef_search_min
                else:
                    print(
                        failing_print_str + f"Target likely at or above J_max. Returning J_max ({J_nef_search_max:.2f}).")
                    return J_nef_search_max

            if abs(obj_at_min) < xtol: return J_nef_search_min
            if abs(obj_at_max) < xtol: return J_nef_search_max

            # Use brentq to find the root.
            j_nef_solution = brentq(objective_fn, J_nef_search_min, J_nef_search_max, xtol=xtol, rtol=rtol)
            print(f"  _find_J_nef: Success for Target={target_rate:.2f}Hz -> J_nef={j_nef_solution:.3f}")
            return j_nef_solution
        except ValueError as e:
            print(
                f"Warning: brentq failed for target_rate {target_rate:.2f} Hz. Error: {e}. Returning best guess from bounds.")
            return J_nef_search_min if abs(obj_at_min) < abs(obj_at_max) else J_nef_search_max

    def gain_bias(self, max_rates, intercepts):
        """
        Overrides the default Nengo gain_bias calculation.

        This method empirically determines the gain and bias for each neuron
        to match the desired `max_rates` and `intercepts`. It does this by:
        1. Finding the current `J_nef` that makes the neuron fire at a near-zero rate (at its intercept).
        2. Finding the current `J_nef` that makes the neuron fire at its max_rate.
        3. Solving a system of two linear equations to find the `gain` and `bias`
           that satisfy these two points.
        """
        print(f"\nSCTNNeuronType: CUSTOM gain_bias called for {self.activation_function_str.upper()} activation.")
        max_rates_arr = np.asarray(np.atleast_1d(max_rates), dtype=float)
        intercepts_arr = np.asarray(np.atleast_1d(intercepts), dtype=float)

        gains = np.zeros_like(max_rates_arr)
        biases_nef = np.zeros_like(max_rates_arr)

        # Use longer simulation times for more accurate rate calculations.
        dt_for_calc = 0.001
        settle_time_gb = 0.3
        sim_time_gb = 1.5
        print(f"  Using settle_time={settle_time_gb}s, sim_time={sim_time_gb}s for rate calculations in gain_bias.")

        rate_at_intercept_target = 0.1  # A small, non-zero rate to represent the threshold.

        # CRITICAL: These search ranges must be manually tuned for each activation function
        # by plotting the F-I curve (Firing rate vs. Input current).
        active_activation = self.activation_function_str.upper()
        if active_activation == "IDENTITY":
            J_nef_search_min_default = -1000 * self.identity_const
            J_nef_search_max_default = 1000 * self.identity_const
        elif active_activation == "BINARY":
            J_nef_search_min_default = -65000
            J_nef_search_max_default = 65000
        else:  # SIGMOID or other
            J_nef_search_min_default = -100000
            J_nef_search_max_default = 100000

        print(
            f"  gain_bias: Using J_nef search range approx [{J_nef_search_min_default:.2f}, {J_nef_search_max_default:.2f}] for activation '{active_activation}'.")

        # Iterate through each neuron to calculate its specific gain and bias.
        for i in range(len(max_rates_arr)):
            max_rate_i = max_rates_arr[i]
            intercept_i = intercepts_arr[i]
            print(f"\n  Neuron {i}: Target max_rate={max_rate_i:.2f} Hz, intercept={intercept_i:.2f}")

            # Adjust search range slightly based on target max_rate.
            current_J_nef_search_max = J_nef_search_max_default + (max_rate_i / 2.0)
            current_J_nef_search_min = J_nef_search_min_default - (max_rate_i / 10.0)

            # Special handling for neurons with very low max firing rates.
            if max_rate_i < rate_at_intercept_target * 1.5:
                J_nef_for_low_max_rate = self._find_J_nef_for_target_rate(
                    max_rate_i, dt_for_calc,
                    current_J_nef_search_min, current_J_nef_search_max,
                    settle_time_gb, sim_time_gb)

                if J_nef_for_low_max_rate is None:
                    gains[i] = 0.0;
                    biases_nef[i] = current_J_nef_search_min
                    continue
                gains[i] = 1e-9;
                biases_nef[i] = J_nef_for_low_max_rate - gains[i] * intercept_i
                continue

            # 1. Find the current that produces a near-zero firing rate.
            J_nef_at_threshold = self._find_J_nef_for_target_rate(
                rate_at_intercept_target, dt_for_calc,
                current_J_nef_search_min, current_J_nef_search_max,
                settle_time_gb, sim_time_gb)

            # 2. Find the current that produces the maximum firing rate.
            J_nef_at_max = self._find_J_nef_for_target_rate(
                max_rate_i, dt_for_calc,
                max(current_J_nef_search_min,
                    J_nef_at_threshold if J_nef_at_threshold is not None else current_J_nef_search_min),
                current_J_nef_search_max,
                settle_time_gb, sim_time_gb)

            if J_nef_at_threshold is None or J_nef_at_max is None:
                print(
                    f"  Warning: Custom gain_bias J_nef root-finding critically failed for neuron {i}. Setting defaults.")
                gains[i] = 0.0;
                biases_nef[i] = (current_J_nef_search_min + current_J_nef_search_max) / 2.0
                continue

            # Handle cases where the F-I curve is flat or non-monotonic.
            if J_nef_at_max <= J_nef_at_threshold + 1e-6:
                print(
                    f"  Warning: J_nef_at_max ({J_nef_at_max:.3f}) not > J_nef_at_threshold ({J_nef_at_threshold:.3f}) for neuron {i}.")
                if max_rate_i > rate_at_intercept_target:
                    gains[i] = np.clip((max_rate_i - rate_at_intercept_target) / 20.0, 1e-9, None)
                    biases_nef[i] = J_nef_at_max - gains[i] * 1.0
                else:
                    gains[i] = 1e-9
                    biases_nef[i] = J_nef_at_threshold - gains[i] * intercept_i
                print(f"     Fallback gain={gains[i]:.3e}, bias_nef={biases_nef[i]:.3f}")
                continue

            # 3. Solve for gain and bias using the two points found.
            # J_thresh = gain * intercept + bias
            # J_max    = gain * 1.0       + bias
            safe_intercept_i = np.clip(intercept_i, -0.99999, 0.99999)
            denominator = 1.0 - safe_intercept_i
            if abs(denominator) < 1e-7:
                gain_i = 1e-9
            else:
                gain_i = (J_nef_at_max - J_nef_at_threshold) / denominator
            bias_nef_i = J_nef_at_threshold - gain_i * safe_intercept_i

            # Clamp negative gains which are physically unrealistic.
            if gain_i < 0:
                print(f"  Warning: Negative gain {gain_i:.4f} for neuron {i}. Clamping & re-adjusting bias.")
                gain_i = 1e-9
                bias_nef_i = J_nef_at_threshold - gain_i * safe_intercept_i

            gains[i] = gain_i
            biases_nef[i] = bias_nef_i
            print(
                f"  Neuron {i}: Found J_thresh={J_nef_at_threshold:.3f}, J_max={J_nef_at_max:.3f} -> gain={gain_i:.3e}, bias_nef={bias_nef_i:.3f}")

        return gains, biases_nef


# --- Simulation Function ---
def run_and_plot_simulation(n_neurons, neuron_params):
    """
    A helper function to build a simple Nengo network, run a simulation,
    and plot the results to test the SCTNNeuronType's performance.
    """
    # 1. Build the Nengo model
    with nengo.Network(seed=123) as model:
        sctn_neuron_type = SCTNNeuronType(**neuron_params)
        input_node = nengo.Node(output=lambda t: np.sin(2 * np.pi * t))
        sctn_ensemble = nengo.Ensemble(
            n_neurons=n_neurons,
            dimensions=1,
            neuron_type=sctn_neuron_type,
            label=f"SCTN Ensemble ({n_neurons} neurons)"
        )
        nengo.Connection(input_node, sctn_ensemble)

        # Probes to record data from the simulation
        input_probe = nengo.Probe(input_node, synapse=0.01)
        output_probe = nengo.Probe(sctn_ensemble, synapse=0.03)

    # 2. Run the simulation
    sim_duration = 5.0
    with nengo.Simulator(model) as sim:
        sim.run(sim_duration)

    # 3. Analyze and plot the results
    true_signal = sim.data[input_probe]
    predicted_signal = sim.data[output_probe]
    mse = np.mean((true_signal - predicted_signal) ** 2)

    print(f"\nResults for {n_neurons} neurons:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")

    plt.figure(figsize=(12, 6))
    plt.plot(sim.trange(), true_signal, label="Input Sine Wave", linewidth=2)
    plt.plot(sim.trange(), predicted_signal, label=f"Output from {n_neurons}-Neuron Ensemble", linewidth=2,
             linestyle='--')
    plt.title(f"Sine Wave Transmission (with {n_neurons} Neurons)\nMSE: {mse:.4f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)


# --- Main script entry point ---
if __name__ == '__main__':
    # Define the number of neurons to test for each simulation run.
    neuron_counts = [5, 10, 100]

    # Define the specific parameters for the SCTN neuron model.
    sctn_params = {
        "theta": 0.0,
        "leakage_period": 3,
        "leakage_factor": 2,
        "threshold_pulse": 0.0005,
        "reset_to": 0.0,
        "activation_function": "BINARY",
        "membrane_should_reset": True,
        "min_clip": -10.0,
        "max_clip": 110.0,
        "amplitude": 1.0
    }

    # Run a separate simulation for each neuron count.
    for n in neuron_counts:
        run_and_plot_simulation(n, sctn_params)

    # Display all the generated plots.
    plt.show()
