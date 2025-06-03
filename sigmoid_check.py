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
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    # --- Nengo Parameters ---
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
            print(f"Warning: Amplitude {self.amplitude} is too small, setting to 1e-9.")
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
            print(f"Warning: leakage_period {self.leakage_period} is < 1. Setting to 1.")
            self.leakage_period = 1

    def rates(self, x, gain, bias, dt=0.001):
        J_for_step_method = self.current(x, gain, bias)

        # These are for Nengo's general call to rates (e.g. for decoders)
        # For gain_bias, we will use longer times inside that specific method.
        settle_time_for_rates = 0.02
        sim_time_for_rates = 0.2

        J_for_step_method = np.asarray(J_for_step_method)
        if J_for_step_method.ndim == 0:
            J_for_step_method = J_for_step_method[np.newaxis]

        if J_for_step_method.ndim == 1:
            n_neurons_calc = J_for_step_method.shape[0]
            rng_for_rates = np.random.RandomState(seed=12345)
            sim_state_calc = self.make_state(n_neurons=n_neurons_calc, rng=rng_for_rates, dtype=J_for_step_method.dtype)
            if "output" not in sim_state_calc: sim_state_calc["output"] = np.zeros(n_neurons_calc,
                                                                                   dtype=J_for_step_method.dtype)
            return settled_firingrate(
                self.step, J_for_step_method, sim_state_calc, dt=dt,
                settle_time=settle_time_for_rates, sim_time=sim_time_for_rates)
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
        current_total = J_nef + self.theta

        voltage_arr = np.asarray(voltage)
        leakage_timer_arr = np.asarray(leakage_timer)
        rand_gauss_var_state_arr = np.asarray(rand_gauss_var_state)
        pn_generator_state_arr = np.asarray(pn_generator_state).astype(np.int32)
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
                if np.any(overflow_mask_subset):  # Check if any element is True before indexing
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


            sum_for_sigmoid-=32768
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

    # --- Custom gain_bias and helpers ---
    def _get_rate_for_NEF_current(self, J_nef, dt_sim=0.001, settle_time=0.3, sim_time=1.5):  # INCREASED TIMES
        """
        Calculates the firing rate for a given J_nef component.
        J_nef is the current part: gain * x_encoded + bias_nef.
        self.theta is added internally to get the total current for the step method.
        """
        J_total_for_step = J_nef + self.theta
        J_array = np.array([J_total_for_step], dtype=float)
        rng_for_calc = np.random.RandomState(seed=12345)
        sim_state_single = self.make_state(n_neurons=1, rng=rng_for_calc, dtype=J_array.dtype)
        if "output" not in sim_state_single:
            sim_state_single["output"] = np.zeros(1, dtype=J_array.dtype)

        # Print a message if sim_time is long, to indicate activity
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
                                    settle_time_gb=0.3, sim_time_gb=1.5,  # INCREASED TIMES
                                    xtol=1e-4, rtol=1e-4):
        objective_fn = lambda J_nef_candidate: self._get_rate_for_NEF_current(
            J_nef_candidate, dt_sim, settle_time_gb, sim_time_gb  # Pass increased times
        ) - target_rate

        # Initial evaluation at bounds
        rate_at_min = self._get_rate_for_NEF_current(J_nef_search_min, dt_sim, settle_time_gb, sim_time_gb)
        rate_at_max = self._get_rate_for_NEF_current(J_nef_search_max, dt_sim, settle_time_gb, sim_time_gb)
        obj_at_min = rate_at_min - target_rate
        obj_at_max = rate_at_max - target_rate

        print(
            f"  _find_J_nef: Target={target_rate:.2f}Hz. Range J_nef=[{J_nef_search_min:.2f} (rate {rate_at_min:.2f}Hz), "
            f"{J_nef_search_max:.2f} (rate {rate_at_max:.2f}Hz)]")

        try:
            if np.sign(obj_at_min) == np.sign(obj_at_max) and abs(obj_at_min) > xtol and abs(
                    obj_at_max) > xtol:  # Root not bracketed and not already at a bound
                failing_print_str = (f"Warning: Root for target rate {target_rate:.2f} Hz not bracketed. ")
                if obj_at_min > 0:  # Both rates above target
                    print(
                        failing_print_str + f"Target likely at or below J_min. Returning J_min ({J_nef_search_min:.2f}).")
                    return J_nef_search_min
                else:  # Both rates below target
                    print(
                        failing_print_str + f"Target likely at or above J_max. Returning J_max ({J_nef_search_max:.2f}).")
                    return J_nef_search_max

            # If one of the bounds is already very close to the target rate
            if abs(obj_at_min) < xtol: return J_nef_search_min
            if abs(obj_at_max) < xtol: return J_nef_search_max

            j_nef_solution = brentq(objective_fn, J_nef_search_min, J_nef_search_max, xtol=xtol, rtol=rtol)
            print(f"  _find_J_nef: Success for Target={target_rate:.2f}Hz -> J_nef={j_nef_solution:.3f}")
            return j_nef_solution
        except ValueError as e:  # Typically means f(a) and f(b) must have different signs for brentq
            print(f"Warning: brentq failed for target_rate {target_rate:.2f} Hz. Error: {e}. "
                  f"Returning best guess from bounds.")
            return J_nef_search_min if abs(obj_at_min) < abs(obj_at_max) else J_nef_search_max

    # This OVERRIDES the default NeuronType.gain_bias
    def gain_bias(self, max_rates, intercepts):
        print(f"\nSCTNNeuronType: CUSTOM gain_bias called for {self.activation_function_str.upper()} activation.")
        max_rates_arr = np.asarray(np.atleast_1d(max_rates), dtype=float)
        intercepts_arr = np.asarray(np.atleast_1d(intercepts), dtype=float)

        gains = np.zeros_like(max_rates_arr)
        biases_nef = np.zeros_like(max_rates_arr)

        # --- INCREASED SIMULATION TIMES FOR ACCURACY ---
        dt_for_calc = 0.001
        settle_time_gb = 0.3  # Increased from 0.05
        sim_time_gb = 1.5  # Increased from 0.2
        print(f"  Using settle_time={settle_time_gb}s, sim_time={sim_time_gb}s for rate calculations in gain_bias.")

        rate_at_intercept_target = 0.1

        # !!! CRITICAL USER INPUT: Define default search range for J_nef values !!!
        # These ranges MUST be tuned by manually plotting the F-I curve for EACH activation type
        # using the longer settle_time_gb and sim_time_gb.
        active_activation = self.activation_function_str.upper()
        if active_activation == "IDENTITY":
            # MANUALLY PLOT F-I FOR IDENTITY AND SET THESE BASED ON OBSERVED RANGE
            J_nef_search_min_default = -1000*self.identity_const  # VERY ROUGH GUESS
            J_nef_search_max_default = 1000*(self.identity_const)   # VERY ROUGH GUESS
            print(
                f"  gain_bias: Initial J_nef search range for IDENTITY: approx [{J_nef_search_min_default:.2f}, {J_nef_search_max_default:.2f}]")
            print(
                "  REMINDER: This IDENTITY search range is a rough guess and likely needs significant tuning based on F-I plotting!")
        elif active_activation == "BINARY":
            # This range also needs verification by plotting F-I for BINARY
            J_nef_search_min_default = -65000
            J_nef_search_max_default = 65000
        else:  # SIGMOID or other (generic, needs tuning)
            J_nef_search_min_default = -100000
            J_nef_search_max_default =-10000000

        print(
            f"  gain_bias: Using J_nef search range approx [{J_nef_search_min_default:.2f}, {J_nef_search_max_default:.2f}] "
            f"for activation '{active_activation}' before per-neuron adjustment.")

        for i in range(len(max_rates_arr)):
            max_rate_i = max_rates_arr[i]
            intercept_i = intercepts_arr[i]
            print(f"\n  Neuron {i}: Target max_rate={max_rate_i:.2f} Hz, intercept={intercept_i:.2f}")

            current_J_nef_search_max = J_nef_search_max_default + (max_rate_i / 2.0)  # Wider for high rates
            current_J_nef_search_min = J_nef_search_min_default - (max_rate_i / 10.0)  # Slightly wider for low rates

            if max_rate_i < rate_at_intercept_target * 1.5:  # Handle very low max_rates
                J_nef_for_low_max_rate = self._find_J_nef_for_target_rate(
                    max_rate_i, dt_for_calc,
                    current_J_nef_search_min, current_J_nef_search_max,
                    settle_time_gb, sim_time_gb)

                if J_nef_for_low_max_rate is None:
                    print(
                        f"  Warning (low max_rate): Custom gain_bias search failed for neuron {i}, max_rate={max_rate_i:.2f}. Setting defaults.")
                    gains[i] = 0.0;
                    biases_nef[i] = current_J_nef_search_min
                    continue
                gains[i] = 1e-9;
                biases_nef[i] = J_nef_for_low_max_rate - gains[i] * intercept_i
                continue

            J_nef_at_threshold = self._find_J_nef_for_target_rate(
                rate_at_intercept_target, dt_for_calc,
                current_J_nef_search_min, current_J_nef_search_max,
                settle_time_gb, sim_time_gb)

            J_nef_at_max = self._find_J_nef_for_target_rate(
                max_rate_i, dt_for_calc,
                # For J_nef_at_max, ensure the search starts from at least J_nef_at_threshold if found
                # This helps if J_nef_at_threshold was found at the upper bound of the initial search.
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

            safe_intercept_i = np.clip(intercept_i, -0.99999, 0.99999)
            denominator = 1.0 - safe_intercept_i
            if abs(denominator) < 1e-7:
                gain_i = 1e-9
            else:
                gain_i = (J_nef_at_max - J_nef_at_threshold) / denominator
            bias_nef_i = J_nef_at_threshold - gain_i * safe_intercept_i

            if gain_i < 0:
                print(f"  Warning: Negative gain {gain_i:.4f} for neuron {i}. Clamping & re-adjusting bias.")
                gain_i = 1e-9
                bias_nef_i = J_nef_at_threshold - gain_i * safe_intercept_i  # Recalc bias

            gains[i] = gain_i
            biases_nef[i] = bias_nef_i
            print(
                f"  Neuron {i}: Found J_thresh={J_nef_at_threshold:.3f}, J_max={J_nef_at_max:.3f} -> gain={gain_i:.3e}, bias_nef={bias_nef_i:.3f}")

        return gains, biases_nef


# --- Main script ---
# --- Main script ---
# --- Main script ---
# --- Main script ---
if __name__ == "__main__":
    print("Starting Nengo SCTNNeuronType SIGMOID diagnostic script with J_nef probe...")

    n_neurons_test = 5
    dimensions_test = 1
    sim_duration = 1.0  # s
    dt_model = 0.001  # s
    # num_timesteps = int(sim_duration / dt_model) # Not explicitly needed here

    # --- CHOOSE ACTIVATION TO TEST ---
    sctn_activation = "IDENTITY"

    sctn_theta = 0.05
    sctn_leakage_factor = 2
    sctn_leakage_period = 5
    sctn_threshold_pulse = 0.5
    sctn_identity_const = 10.0
    sctn_gaussian_rand_order = 8

    sctn_neuron_params = SCTNNeuronType(
        activation_function=sctn_activation,
        theta=sctn_theta,
        leakage_factor=sctn_leakage_factor,
        leakage_period=int(sctn_leakage_period),
        threshold_pulse=sctn_threshold_pulse,
        identity_const=sctn_identity_const,
        gaussian_rand_order=sctn_gaussian_rand_order
    )
    print(f"Using SCTNNeuronType with: {sctn_activation}, theta={sctn_theta}, "
          f"lk_factor={sctn_leakage_factor}, lk_period={int(sctn_leakage_period)}, "
          f"gauss_order={sctn_gaussian_rand_order}")

    with nengo.Network(label="SCTN Sigmoid J_nef Test", seed=34) as model:  # Updated seed for new test run
        input_node = nengo.Node(lambda t: (2 * t / sim_duration) - 1 if t < sim_duration else 0)

        sctn_ensemble = nengo.Ensemble(
            n_neurons=n_neurons_test,
            dimensions=dimensions_test,
            neuron_type=sctn_neuron_params,
            max_rates=Uniform(50, 100),
            intercepts=Uniform(-0.8, 0.8),
            seed=46  # Updated seed
        )
        nengo.Connection(input_node, sctn_ensemble, synapse=None)

        # --- Probes for SIGMOID diagnostics ---
        input_stim_probe = nengo.Probe(input_node, synapse=None)  # The x(t) stimulus
        # Probe the 'input' to the neurons, which is J_nef = gain*(encoder*x) + bias_nef
        sctn_Jnef_probe = nengo.Probe(sctn_ensemble.neurons, "input", synapse=None)
        sctn_voltage_probe = nengo.Probe(sctn_ensemble.neurons, "voltage", synapse=None)
        sctn_rand_gauss_var_probe = nengo.Probe(sctn_ensemble.neurons, "rand_gauss_var_state", synapse=None)
        sctn_spikes_probe = nengo.Probe(sctn_ensemble.neurons, "output", synapse=None)
        sctn_filtered_activity = nengo.Probe(sctn_ensemble.neurons, synapse=0.01)

    print("\nBuilding the Nengo simulator (with custom gain_bias for SIGMOID)...")
    start_time = time.time()
    try:
        with nengo.Simulator(model, dt=dt_model, seed=54) as sim:  # Updated seed
            build_time = time.time() - start_time
            print(f"Simulator build time: {build_time:.4f} seconds")
            print(f"Running simulation for {sim_duration} seconds...")
            sim.run(sim_duration)
            print("Simulation complete.")

            # --- Plotting probed data ---
            print("Plotting probed data for SIGMOID activation...")
            # Increased figure height to accommodate the new J_nef plot
            fig, axs = plt.subplots(6, 1, figsize=(12, 15), sharex=True)

            axs[0].plot(sim.trange(), sim.data[input_stim_probe], label="Input Stimulus x(t)")
            axs[0].set_title("Input Stimulus x(t)")
            axs[0].set_ylabel("Value")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(sim.trange(), sim.data[sctn_Jnef_probe])
            axs[1].set_title(f"Neuron Input Current J_nef (gain*enc*x + bias_nef) (N={n_neurons_test})")
            axs[1].set_ylabel("Current J_nef")
            axs[1].grid(True)

            axs[2].plot(sim.trange(), sim.data[sctn_voltage_probe])
            axs[2].set_title(f"Neuron Voltages (SIGMOID, N={n_neurons_test})")
            axs[2].set_ylabel("Voltage")
            axs[2].grid(True)

            axs[3].plot(sim.trange(), sim.data[sctn_rand_gauss_var_probe])
            axs[3].set_title("Neuron rand_gauss_var_state (Shifted Threshold)")
            axs[3].set_ylabel("Shifted Threshold")
            axs[3].grid(True)

            axs[4].plot(sim.trange(), sim.data[sctn_filtered_activity])
            axs[4].set_title("Neuron Filtered Output Activity")
            axs[4].set_ylabel("Filtered Activity (Hz)")
            axs[4].grid(True)
            if not np.any(sim.data[sctn_spikes_probe] > 0):
                print("SIGMOID (with J_nef probe): NO Spikes detected during simulation.")
            else:
                print("SIGMOID (with J_nef probe): Spikes were detected!")

            # --- Histogram of rand_gauss_var_state ---
            rand_gauss_data = sim.data[sctn_rand_gauss_var_probe].flatten()
            mean_rgv = np.mean(rand_gauss_data)
            std_rgv = np.std(rand_gauss_data)
            min_rgv = np.min(rand_gauss_data)
            max_rgv = np.max(rand_gauss_data)

            print(f"\nStatistics for SHIFTED rand_gauss_var_state (thresholds):")
            print(f"  Mean: {mean_rgv:.2f}")
            print(f"  Std Dev: {std_rgv:.2f}")
            print(f"  Min: {min_rgv:.2f}")
            print(f"  Max: {max_rgv:.2f}")

            axs[5].hist(rand_gauss_data, bins=50, density=True, alpha=0.7, label='Shifted Threshold')
            from scipy.stats import norm  # Keep this import local if only used here

            xmin_hist, xmax_hist = axs[5].get_xlim()  # Use axis limits for PDF plotting range
            x_pdf = np.linspace(xmin_hist, xmax_hist, 100)
            p_pdf = norm.pdf(x_pdf, mean_rgv, std_rgv)
            axs[5].plot(x_pdf, p_pdf, 'k', linewidth=2, label=f'Gaussian PDF Fit')
            axs[5].set_title(f"Histogram of Shifted Threshold (mean={mean_rgv:.1f}, std={std_rgv:.1f})")
            axs[5].set_xlabel("Value of Shifted Threshold")
            axs[5].set_ylabel("Density")
            axs[5].legend()
            axs[5].grid(True)
            axs[5].set_xlabel("Time (s)")  # Common x-axis label

            plt.tight_layout()
            plt.show()

            # Response curve plot (still useful to see what gain_bias did)
            print("Generating and plotting response curves for SIGMOID...")
            plt.figure(figsize=(12, 8))
            eval_points, activities = nengo.utils.ensemble.tuning_curves(sctn_ensemble, sim)
            plt.plot(eval_points, activities, lw=1.5)
            plt.title(f"Response Curves for SCTN Ensemble (Custom gain_bias, {sctn_activation})")
            plt.xlabel("Input Stimulus (x)")
            plt.ylabel("Firing Rate (Hz)")
            plt.grid(True)
            # ... (fig_text_str and other plot details remain the same) ...
            fig_text_str = (f"Build: {build_time:.2f}s. N={n_neurons_test}. Act: {sctn_activation}, "
                            f"theta: {sctn_theta}, lk_f: {sctn_leakage_factor}, lk_p: {int(sctn_leakage_period)}")
            plt.figtext(0.5, 0.01, fig_text_str, wrap=True, horizontalalignment='center', fontsize=8)
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            plt.show()

    except ValidationError as e:
        build_time = time.time() - start_time
        print(f"!!! Nengo ValidationError during build (time: {build_time:.2f}s) with custom gain_bias !!!")
        print(f"{type(e).__name__}: {e}")
    except Exception as e:
        build_time = time.time() - start_time
        print(f"--- An error occurred (time: {build_time:.2f}s) with custom gain_bias ---")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("\n--- SIGMOID Diagnostic Script with J_nef probe Finished ---")
    print("What to check:")
    print("1. J_nef Plot: What are the values of J_nef? Are they mostly positive, negative, or small?")
    print("2. Voltages: How do they behave now, in relation to J_nef and theta?")
    print("3. Shifted Threshold (RandGaussVarState): Confirm its mean is near zero.")
    print("4. Output Activity: Any spikes now? Compare voltages to the shifted thresholds.")
    print("5. Response Curves: Any improvement?")