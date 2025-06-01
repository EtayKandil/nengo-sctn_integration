import numpy as np
import nengo
from nengo.neurons import NeuronType, settled_firingrate
from nengo.dists import Choice
from nengo.utils.numpy import clip
import matplotlib.pyplot as plt


# --- SCTNNeuronType Definition (from nengo_sctn_model_definition artifact) ---
class SCTNNeuronType(NeuronType):
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    state = {
        "voltage": Choice([0.0]),
        "leakage_timer": Choice([0]),
        "rand_gauss_var_state": Choice([0.0]),
        "pn_generator_state": Choice([1.0]),
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
        self.threshold_pulse_val = float(threshold_pulse)
        self.activation_function_str = activation_function.upper()
        self.membrane_should_reset_val = bool(membrane_should_reset)
        self.amplitude_val = float(amplitude)
        self.identity_const_val = np.int32(identity_const)
        self.gaussian_rand_order_val = np.int32(gaussian_rand_order)

        if self.activation_function_str == "BINARY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_BINARY
        elif self.activation_function_str == "IDENTITY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_IDENTITY
        elif self.activation_function_str == "SIGMOID":
            self.current_activation_fn_id = SCTNNeuronType.ACT_SIGMOID
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
                settle_time=0.05, sim_time=0.5)
        elif J_all_samples.ndim == 2:
            n_samples, n_neurons = J_all_samples.shape
            output_rates = np.zeros_like(J_all_samples)
            for i in range(n_samples):
                J_sample = J_all_samples[i, :]
                sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_sample.dtype)
                sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_sample.dtype)
                rates_for_sample = settled_firingrate(
                    self.step, J_sample, sim_state_1d, dt=dt,
                    settle_time=0.05, sim_time=0.5)
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
                    settle_time=0.05, sim_time=0.5)
            else:
                raise ValueError(f"Unexpected J shape in rates method: {J_all_samples.shape}")

    def step(self, dt, J, output, voltage, leakage_timer, rand_gauss_var_state, pn_generator_state):
        current_plus_theta = J + self.theta_val
        if self.leakage_factor_val < 3:
            effective_input_current = current_plus_theta
        else:
            lf = float(2 ** (int(self.leakage_factor_val) - 3))
            effective_input_current = current_plus_theta * lf
        voltage += effective_input_current * dt
        voltage[:] = clip(voltage, self.min_clip_val, self.max_clip_val)
        emit_spike_flags = np.zeros_like(voltage, dtype=bool)
        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            emit_spike_flags = voltage > self.threshold_pulse_val
        elif self.current_activation_fn_id == SCTNNeuronType.ACT_IDENTITY:
            const = float(self.identity_const_val)
            mask_gt_const = voltage > const
            emit_spike_flags[mask_gt_const] = True
            rand_gauss_var_state[mask_gt_const] = const
            mask_lt_neg_const = voltage < -const
            rand_gauss_var_state[mask_lt_neg_const] = const
            mask_else = ~(mask_gt_const | mask_lt_neg_const)
            for i in np.where(mask_else)[0]:
                c_val = voltage[i] + const
                current_rand_val_int = np.int32(rand_gauss_var_state[i])
                c_val_int = np.int32(c_val)
                accumulated_int = current_rand_val_int + c_val_int + 1
                if accumulated_int >= 65536:
                    accumulated_int -= 65536
                    emit_spike_flags[i] = True
                rand_gauss_var_state[i] = float(accumulated_int)
        elif self.current_activation_fn_id == SCTNNeuronType.ACT_SIGMOID:
            current_rand_gauss_sum_for_sigmoid = np.zeros_like(voltage)
            temp_pn_generator = pn_generator_state.astype(np.int32)
            for _ in range(int(self.gaussian_rand_order_val)):
                current_rand_gauss_sum_for_sigmoid += temp_pn_generator & 0x1fff
                feedback_bit_14 = (temp_pn_generator >> 14) & 1
                feedback_bit_0 = temp_pn_generator & 1
                new_msb = feedback_bit_14 ^ feedback_bit_0
                temp_pn_generator = (temp_pn_generator >> 1) | (new_msb << 14)
                temp_pn_generator &= 0x7FFF
            pn_generator_state[:] = temp_pn_generator.astype(float)
            rand_gauss_var_state[:] = current_rand_gauss_sum_for_sigmoid
            emit_spike_flags = voltage > rand_gauss_var_state
        leak_now_mask = leakage_timer >= self.leakage_period_val
        if self.leakage_factor_val > 0:
            decay_multiplier = (1.0 - (1.0 / (2 ** int(self.leakage_factor_val))))
            voltage[leak_now_mask] *= decay_multiplier
        leakage_timer[leak_now_mask] = 0
        leakage_timer[~leak_now_mask] += 1
        output[:] = 0.0
        output[emit_spike_flags] = self.amplitude_val / dt
        if self.membrane_should_reset_val:
            voltage[emit_spike_flags] = self.reset_to_val


# --- End of SCTNNeuronType Definition ---


# --- Oscillator Parameters (from your LIF script) ---
sim_run_time = 5  # in seconds
sim_n_neurons = 100  # Number of neurons per 1D ensemble
sim_dt = 1e-3  # Simulation timestep
sim_tau = 0.1  # Synaptic time constant for recurrent connections
sim_seed = 1771
sim_A = np.array([[1, 1], [-1, 1]])  # Dynamics matrix

# SCTN neuron parameters (tune as needed)
sctn_params_for_oscillator = {
    "theta": 0.0,
    "leakage_period": 10,  # Might need to be longer or factor smaller for sustained oscillation
    "leakage_factor": 1,  # Less aggressive leakage
    "threshold_pulse": 0.1,  # Adjust based on typical voltage range for SCTN
    "reset_to": 0.0,
    "activation_function": "BINARY",  # Using BINARY for this test
    "membrane_should_reset": True,
    "min_clip": -1.5,  # Allow some negative voltage if helpful
    "max_clip": 1.5,  # Keep within a reasonable range for representation
    "amplitude": 1.0
}


def input_func(t):
    # First element is 1.0 whenever t<0.1, second element is always 0.0
    if t < 0.1:
        return [1.0, 0.0]  # Kick x1
    else:
        return [0.0, 0.0]


# --- Model Definition ---
with nengo.Network(label="SCTN_Matrix_A_Oscillator", seed=sim_seed) as model:
    input_node = nengo.Node(output=input_func, label="KickInput")

    # Two 1-D ensembles using SCTNNeuronType
    x1_ensemble = nengo.Ensemble(
        sim_n_neurons,
        dimensions=1,
        neuron_type=SCTNNeuronType(**sctn_params_for_oscillator),
        label="x1_SCTN"
    )
    x2_ensemble = nengo.Ensemble(
        sim_n_neurons,
        dimensions=1,
        neuron_type=SCTNNeuronType(**sctn_params_for_oscillator),
        label="x2_SCTN"
    )

    # Connect initial kick
    # Input[0] (kick for x1) -> x1_ensemble
    nengo.Connection(input_node[0], x1_ensemble, synapse=None)
    # Input[1] (kick for x2, which is 0) -> x2_ensemble
    nengo.Connection(input_node[1], x2_ensemble, synapse=None)

    # Implement recurrent dynamics:
    # dx1/dt ~ (1/tau) * (A[0,0]*x1 + A[0,1]*x2)
    # dx2/dt ~ (1/tau) * (A[1,0]*x1 + A[1,1]*x2)

    # Connections to x1_ensemble (target for first row of matrix A)
    nengo.Connection(x1_ensemble, x1_ensemble, transform=sim_A[0, 0], synapse=sim_tau)
    nengo.Connection(x2_ensemble, x1_ensemble, transform=sim_A[0, 1], synapse=sim_tau)

    # Connections to x2_ensemble (target for second row of matrix A)
    nengo.Connection(x1_ensemble, x2_ensemble, transform=sim_A[1, 0], synapse=sim_tau)
    nengo.Connection(x2_ensemble, x2_ensemble, transform=sim_A[1, 1], synapse=sim_tau)

    # --- Probes ---
    probe_input = nengo.Probe(input_node, "output", sample_every=0.01)
    probe_x1 = nengo.Probe(x1_ensemble, synapse=0.03, sample_every=0.01)  # Smoothed decoded output
    probe_x2 = nengo.Probe(x2_ensemble, synapse=0.03, sample_every=0.01)  # Smoothed decoded output
    # Optional: Probe spikes from a few neurons in x1
    # probe_x1_spikes = nengo.Probe(x1_ensemble.neurons[::sim_n_neurons//10], "output", sample_every=0.01)

# --- Simulation ---
with nengo.Simulator(model, dt=sim_dt, seed=sim_seed) as sim:
    sim.run(sim_run_time)

# --- Plotting ---
time_vector = sim.trange(sample_every=0.01)  # Match probe sample_every

fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f"SCTN Oscillator (Matrix A method, tau={sim_tau})", fontsize=16)

# 1. Plot Kick Input
axs[0].plot(time_vector, sim.data[probe_input][:, 0], label="Kick to x1")
axs[0].plot(time_vector, sim.data[probe_input][:, 1], label="Kick to x2")
axs[0].set_title("Initial Kick Input")
axs[0].set_ylabel("Input Value")
axs[0].legend(loc='upper right')
axs[0].grid(True)

# 2. Plot Oscillator Decoded Output (x1 and x2)
axs[1].plot(time_vector, sim.data[probe_x1], label="x1 (decoded)")
axs[1].plot(time_vector, sim.data[probe_x2], label="x2 (decoded)")
axs[1].set_title("Oscillator Ensembles Decoded Output")
axs[1].set_ylabel("State Value")
axs[1].legend(loc='upper right')
axs[1].grid(True)

# 3. Plot Phase Portrait (x2 vs x1)
axs[2].plot(sim.data[probe_x1], sim.data[probe_x2])
axs[2].set_title("Phase Portrait (x2 vs x1)")
axs[2].set_xlabel("x1")
axs[2].set_ylabel("x2")
axs[2].axis('equal')
axs[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
