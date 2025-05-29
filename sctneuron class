import numpy as np
import nengo
from nengo.neurons import NeuronType, settled_firingrate
from nengo.dists import Choice
from nengo.utils.numpy import clip
# Note: matplotlib.pyplot is not needed for the model definition itself

# --- SCTNNeuronType Definition ---
class SCTNNeuronType(NeuronType):
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    state = {
        "voltage": Choice([0.0]),
        "leakage_timer": Choice([0]),
        "rand_gauss_var_state": Choice([0]), # For IDENTITY and SIGMOID
        "pn_generator_state": Choice([1]),   # For SIGMOID, initial seed typically > 0
    }
    spiking = True

    def __init__(self,
                 theta=0.0,
                 reset_to=0.0,
                 min_clip=-524287.0,
                 max_clip=524287.0,
                 leakage_factor=0,
                 leakage_period=1,
                 threshold_pulse=0.0,
                 activation_function="BINARY",
                 membrane_should_reset=True,
                 amplitude=1.0,
                 # New parameters for IDENTITY and SIGMOID
                 identity_const=32767,
                 gaussian_rand_order=8,
                 initial_state=None):
        super().__init__(initial_state=initial_state)
        self.theta_val = float(theta)
        self.reset_to_val = float(reset_to)
        self.min_clip_val = float(min_clip)
        self.max_clip_val = float(max_clip)
        self.leakage_factor_val = np.int16(leakage_factor)
        self.leakage_period_val = np.int16(leakage_period)
        if self.leakage_period_val < 1:
            self.leakage_period_val = np.int16(1)
        self.threshold_pulse_val = float(threshold_pulse) # Used by BINARY
        self.activation_function_str = activation_function.upper()
        self.membrane_should_reset_val = bool(membrane_should_reset)
        self.amplitude_val = float(amplitude)

        # Store new parameters
        self.identity_const_val = np.int32(identity_const) # Match original type
        self.gaussian_rand_order_val = np.int32(gaussian_rand_order)

        if self.activation_function_str == "BINARY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_BINARY
        elif self.activation_function_str == "IDENTITY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_IDENTITY
            # No NotImplementedError now, we will implement it
        elif self.activation_function_str == "SIGMOID":
            self.current_activation_fn_id = SCTNNeuronType.ACT_SIGMOID
            # No NotImplementedError now, we will implement it
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function_str}.")

    def rates(self, x, gain, bias, dt=0.001):
        J_all_samples = self.current(x, gain, bias)

        if J_all_samples.ndim == 1:
            n_neurons = J_all_samples.shape[0]
            sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_all_samples.dtype)
            sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_all_samples.dtype)
            return settled_firingrate(
                self.step, J_all_samples, sim_state_1d, dt=dt,
                settle_time=0.05, sim_time=0.5
            )
        elif J_all_samples.ndim == 2:
            n_samples, n_neurons = J_all_samples.shape
            output_rates = np.zeros_like(J_all_samples)
            for i in range(n_samples):
                J_sample = J_all_samples[i, :]
                sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_sample.dtype)
                sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_sample.dtype)
                rates_for_sample = settled_firingrate(
                    self.step, J_sample, sim_state_1d, dt=dt,
                    settle_time=0.05, sim_time=0.5
                )
                output_rates[i, :] = rates_for_sample
            return output_rates
        else:
            if J_all_samples.ndim == 0:
                n_neurons = 1
                J_sample_1d = J_all_samples.reshape((1,))
                sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_sample_1d.dtype)
                sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_sample_1d.dtype)
                return settled_firingrate(
                    self.step, J_sample_1d, sim_state_1d, dt=dt,
                    settle_time=0.05, sim_time=0.5
                )
            else:
                raise ValueError(f"Unexpected J shape in rates method: {J_all_samples.shape}")

    def step(self, dt, J, output, voltage, leakage_timer, rand_gauss_var_state, pn_generator_state):
        # Note: rand_gauss_var_state and pn_generator_state are now expected based on self.state
        
        current_plus_theta = J + self.theta_val
        if self.leakage_factor_val < 3:
            effective_input_current = current_plus_theta
        else:
            lf = float(2**(int(self.leakage_factor_val) - 3))
            effective_input_current = current_plus_theta * lf

        voltage += effective_input_current * dt
        voltage[:] = clip(voltage, self.min_clip_val, self.max_clip_val)

        # Initialize emit_spike_flags for all neurons
        emit_spike_flags = np.zeros_like(voltage, dtype=bool)

        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            emit_spike_flags = voltage > self.threshold_pulse_val

        elif self.current_activation_fn_id == SCTNNeuronType.ACT_IDENTITY:
            # Vectorized IDENTITY logic
            const = self.identity_const_val # scalar
            
            # Condition 1: voltage > const
            mask_gt_const = voltage > const
            emit_spike_flags[mask_gt_const] = True
            rand_gauss_var_state[mask_gt_const] = const # Uses float assignment

            # Condition 2: voltage < -const
            mask_lt_neg_const = voltage < -const
            # emit_spike_flags[mask_lt_neg_const] is already False from initialization
            rand_gauss_var_state[mask_lt_neg_const] = const # Uses float assignment
            
            # Condition 3: -const <= voltage <= const
            mask_else = ~(mask_gt_const | mask_lt_neg_const)
            
            if np.any(mask_else):
                # Original: self.rand_gauss_var = int(self.rand_gauss_var + c + 1)
                # With modulo: if self.rand_gauss_var >= 65536: self.rand_gauss_var -= 65536; emit_spike = 1
                # This part is tricky to vectorize perfectly while matching int behavior if states are float.
                # Using a loop for the neurons matching mask_else to ensure exact logic.
                for i in np.where(mask_else)[0]:
                    c_val = voltage[i] + const # c_val is float
                    # Accumulate as float, then check threshold. Original used int accumulation.
                    # This might differ if rand_gauss_var_state was meant to be strictly integer with overflow.
                    rand_gauss_var_state[i] = rand_gauss_var_state[i] + c_val + 1.0
                    if rand_gauss_var_state[i] >= 65536.0:
                        rand_gauss_var_state[i] -= 65536.0
                        emit_spike_flags[i] = True
                    # else: emit_spike_flags[i] is already False
            
        elif self.current_activation_fn_id == SCTNNeuronType.ACT_SIGMOID:
            # rand_gauss_var_state is reset to 0 for each neuron then accumulated
            # In original, self.rand_gauss_var = 0 was at the start of _activation_function_sigmoid
            # Here, rand_gauss_var_state is persistent. If it should reset each step for SIGMOID,
            # it should be: current_rand_gauss_sum = np.zeros_like(voltage)
            # If it's an accumulator that resets elsewhere or based on other conditions, this is different.
            # Assuming for now it's an accumulator for the sum within the gaussian_rand_order loop.
            # Let's follow original: rand_gauss_var is reset for this calculation.
            current_rand_gauss_sum_for_sigmoid = np.zeros_like(voltage) 
            
            # Using a loop for pn_generator_state update for clarity and bitwise correctness
            # as Nengo states are float by default.
            temp_pn_generator = pn_generator_state.astype(np.int32) # Work with int copy

            for _ in range(int(self.gaussian_rand_order_val)):
                current_rand_gauss_sum_for_sigmoid += temp_pn_generator & 0x1fff
                
                feedback_bit_from_14 = (temp_pn_generator >> 14) & 1 
                feedback_bit_from_0 = temp_pn_generator & 1
                new_msb = feedback_bit_from_14 ^ feedback_bit_from_0
                
                temp_pn_generator = (temp_pn_generator >> 1) | (new_msb << 14) 
                temp_pn_generator &= 0x7FFF # Assuming 15-bit LFSR, adjust if different
            
            pn_generator_state[:] = temp_pn_generator.astype(float) # Update the actual state array
            rand_gauss_var_state[:] = current_rand_gauss_sum_for_sigmoid # Store the sum for this step
            emit_spike_flags = voltage > rand_gauss_var_state


        leak_now_mask = leakage_timer >= self.leakage_period_val
        if self.leakage_factor_val > 0:
            decay_multiplier = (1.0 - (1.0 / (2**int(self.leakage_factor_val))))
            voltage[leak_now_mask] *= decay_multiplier

        leakage_timer[leak_now_mask] = 0
        leakage_timer[~leak_now_mask] += 1

        output[:] = 0.0
        output[emit_spike_flags] = self.amplitude_val / dt

        if self.membrane_should_reset_val:
            voltage[emit_spike_flags] = self.reset_to_val
# --- End of SCTNNeuronType Definition ---
