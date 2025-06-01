import numpy as np
import nengo
from nengo.neurons import NeuronType, settled_firingrate
from nengo.dists import Choice
from nengo.utils.numpy import clip
# Import Parameter types and the Unconfigurable sentinel
from nengo.params import (
    NumberParam, BoolParam, StringParam, IntParam, Unconfigurable
)
import matplotlib.pyplot as plt


# --- SCTNNeuronType Definition ---
class SCTNNeuronType(NeuronType):
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    # --- Nengo Parameters (Defaults match original SCTN where specified) ---
    theta = NumberParam("theta", default=0.0, readonly=True)
    reset_to = NumberParam("reset_to", default=0.0, readonly=True)
    min_clip = NumberParam("min_clip", default=-524287.0, readonly=True)
    max_clip = NumberParam("max_clip", default=524287.0, readonly=True)
    leakage_factor = IntParam("leakage_factor", default=0, low=0, readonly=True)
    leakage_period = IntParam("leakage_period", default=1, low=1, readonly=True)
    threshold_pulse = NumberParam("threshold_pulse", default=0.0, readonly=True)  # For BINARY
    activation_function_str = StringParam("activation_function_str", default="IDENTITY", readonly=True)
    membrane_should_reset = BoolParam("membrane_should_reset", default=True, readonly=True)
    amplitude = NumberParam("amplitude", default=1.0, low=0, readonly=True)
    identity_const = NumberParam("identity_const", default=32767.0, readonly=True)  # Original SCTN default
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
                 activation_function=Unconfigurable,
                 membrane_should_reset=Unconfigurable,
                 amplitude=Unconfigurable,
                 identity_const=Unconfigurable,
                 gaussian_rand_order=Unconfigurable,
                 initial_state=None):
        super().__init__(initial_state=initial_state)

        if theta is not Unconfigurable: self.theta = theta
        if reset_to is not Unconfigurable: self.reset_to = reset_to
        if min_clip is not Unconfigurable: self.min_clip = min_clip
        if max_clip is not Unconfigurable: self.max_clip = max_clip
        if leakage_factor is not Unconfigurable: self.leakage_factor = leakage_factor
        if leakage_period is not Unconfigurable: self.leakage_period = leakage_period
        if threshold_pulse is not Unconfigurable: self.threshold_pulse = threshold_pulse
        if activation_function is not Unconfigurable: self.activation_function_str = activation_function
        if membrane_should_reset is not Unconfigurable: self.membrane_should_reset = membrane_should_reset
        if amplitude is not Unconfigurable: self.amplitude = amplitude
        if identity_const is not Unconfigurable: self.identity_const = identity_const
        if gaussian_rand_order is not Unconfigurable: self.gaussian_rand_order = gaussian_rand_order

        act_str_upper = self.activation_function_str.upper()
        if act_str_upper == "BINARY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_BINARY
        elif act_str_upper == "IDENTITY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_IDENTITY
        elif act_str_upper == "SIGMOID":
            self.current_activation_fn_id = SCTNNeuronType.ACT_SIGMOID
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function_str}.")

    def rates(self, x, gain, bias, dt=0.001):
        # This method is crucial for NEF processes like decoder solving.
        # If the natural spiking rate is too low with PDM parameters,
        # this can lead to "all zero activities" errors during build.
        J_all_samples = self.current(x, gain, bias)
        if J_all_samples.ndim == 1:
            n_neurons = J_all_samples.shape[0]
            sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_all_samples.dtype)
            sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_all_samples.dtype)
            return settled_firingrate(
                self.step, J_all_samples, sim_state_1d, dt=dt,
                settle_time=0.02, sim_time=0.2)  # Shorter sim_time for build
        elif J_all_samples.ndim == 2:
            n_samples, n_neurons = J_all_samples.shape
            output_rates = np.zeros_like(J_all_samples)
            for i in range(n_samples):
                J_sample = J_all_samples[i, :]
                sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_sample.dtype)
                sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_sample.dtype)
                rates_for_sample = settled_firingrate(
                    self.step, J_sample, sim_state_1d, dt=dt,
                    settle_time=0.02, sim_time=0.2)
                output_rates[i, :] = rates_for_sample
            return output_rates
        else:  # Scalar J implies n_neurons=1
            n_neurons = 1
            J_sample_1d = np.atleast_1d(J_all_samples)  # Ensure J is at least 1D
            sim_state_1d = self.make_state(n_neurons=n_neurons, rng=np.random, dtype=J_sample_1d.dtype)
            sim_state_1d["output"] = np.zeros(n_neurons, dtype=J_sample_1d.dtype)
            return settled_firingrate(
                self.step, J_sample_1d, sim_state_1d, dt=dt,
                settle_time=0.02, sim_time=0.2)

    def step(self, dt, J, output, voltage, leakage_timer, rand_gauss_var_state, pn_generator_state):
        current_plus_theta = J + self.theta

        effective_input_current = 0.0
        current_leakage_factor = int(self.leakage_factor)
        if current_leakage_factor < 3:
            effective_input_current = current_plus_theta
        else:
            lf = float(2 ** (current_leakage_factor - 3))
            effective_input_current = current_plus_theta * lf

        voltage += effective_input_current * dt
        voltage[:] = clip(voltage, self.min_clip, self.max_clip)

        emit_spike_flags = np.zeros_like(voltage, dtype=bool)

        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            emit_spike_flags = voltage > self.threshold_pulse
        elif self.current_activation_fn_id == SCTNNeuronType.ACT_IDENTITY:
            const = float(self.identity_const)

            mask_gt_const = voltage > const
            emit_spike_flags[mask_gt_const] = True
            rand_gauss_var_state[mask_gt_const] = const

            mask_lt_neg_const = voltage < -const
            rand_gauss_var_state[mask_lt_neg_const] = const

            mask_else = ~(mask_gt_const | mask_lt_neg_const)

            for i in np.where(mask_else)[0]:
                c_val = voltage[i] + const
                increment = c_val + 1.0

                rand_gauss_var_state[i] += increment

                if rand_gauss_var_state[i] >= 65536.0:  # Fixed overflow threshold
                    rand_gauss_var_state[i] = np.mod(rand_gauss_var_state[i], 65536.0)
                    emit_spike_flags[i] = True

        elif self.current_activation_fn_id == SCTNNeuronType.ACT_SIGMOID:
            current_rand_gauss_sum_for_sigmoid = np.zeros_like(voltage)
            temp_pn_generator = pn_generator_state.astype(np.int32)
            current_gaussian_rand_order_val = int(self.gaussian_rand_order)

            for _ in range(current_gaussian_rand_order_val):
                current_rand_gauss_sum_for_sigmoid += temp_pn_generator & 0x1fff
                feedback_bit_14 = (temp_pn_generator >> 14) & 1
                feedback_bit_0 = temp_pn_generator & 1
                new_msb = feedback_bit_14 ^ feedback_bit_0
                temp_pn_generator = (temp_pn_generator >> 1) | (new_msb << 14)
                temp_pn_generator &= 0x7FFF
            pn_generator_state[:] = temp_pn_generator.astype(float)
            rand_gauss_var_state[:] = current_rand_gauss_sum_for_sigmoid
            emit_spike_flags = voltage > rand_gauss_var_state

        if current_leakage_factor > 0:
            current_leakage_period = int(self.leakage_period)
            leak_now_mask = leakage_timer >= current_leakage_period
            decay_multiplier = (1.0 - (1.0 / (2 ** current_leakage_factor)))
            voltage[leak_now_mask] *= decay_multiplier
            leakage_timer[leak_now_mask] = 0
            leakage_timer[~leak_now_mask] += 1.0

        output[:] = 0.0
        output[emit_spike_flags] = self.amplitude / dt
        if self.membrane_should_reset:
            voltage[emit_spike_flags] = self.reset_to
        # --- End of SCTNNeuronType Definition ---


# --- PDM Model Setup ---
N_PDM_NEURONS = 1
SIMULATION_TIME = 5 # Simulation time in seconds
DT = 0.001  # Standard Nengo timestep

# Parameters for the SCTNNeuronType instance in this PDM simulation
# The CLASS default for identity_const is 32767.0.
# For THIS SIMULATION, we override it to a value suitable for PDM
# with input signals that result in small voltage swings.
sctn_pdm_params = {
    "theta": 0.0,
    "reset_to": 0.0,
    "min_clip": -2.0,
    "max_clip": 2.0,
    "leakage_factor": 0,  # No leakage for clearer PDM based on integration
    "leakage_period": 1,  # Irrelevant if leakage_factor is 0
    "activation_function": "IDENTITY",
    "membrane_should_reset": True,
    "amplitude": 1.0,
    "identity_const": 0.5,  # << CHOSEN FOR PDM with small voltage swings
}

with nengo.Network(label="SCTN_PDM_Network", seed=79) as model:
    input_freq = 0.5
    input_amp = 0.5  # Input signal amplitude
    input_node = nengo.Node(output=lambda t: input_amp * np.sin(2 * np.pi * input_freq * t),
                            label="AnalogInput")

    pdm_ensemble = nengo.Ensemble(
        n_neurons=N_PDM_NEURONS,
        dimensions=1,
        neuron_type=SCTNNeuronType(**sctn_pdm_params),
        # Ensure J to neuron = input_node value
        encoders=nengo.dists.Choice([[1.0]]),
        gain=nengo.dists.Choice([50.0] * N_PDM_NEURONS),
        bias=nengo.dists.Choice([0.0] * N_PDM_NEURONS),
        radius=input_amp * 1.5,  # Set radius based on input amplitude
        label="PDM_Ensemble"
    )

    nengo.Connection(input_node, pdm_ensemble, synapse=None)

    # --- Probes ---
    probe_analog_input = nengo.Probe(input_node, "output", sample_every=0.005)
    probe_pdm_voltage = nengo.Probe(pdm_ensemble.neurons, "voltage", sample_every=0.005)
    probe_pdm_rand_gauss = nengo.Probe(pdm_ensemble.neurons, "rand_gauss_var_state", sample_every=0.005)
    probe_pdm_spikes = nengo.Probe(pdm_ensemble.neurons, "output", sample_every=DT)

    # probe_pdm_decoded = nengo.Probe(pdm_ensemble, synapse=0.05, sample_every=0.005)

# --- Simulation ---
with nengo.Simulator(model, dt=DT) as sim:
    sim.run(SIMULATION_TIME)

# --- Plotting ---
time_vector_analog = sim.trange(sample_every=0.005)
time_vector_spikes = sim.trange(sample_every=DT)

num_subplots = 4

fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 3 * num_subplots + 3), sharex=True)
fig.suptitle(f"SCTN PDM (IDENTITY, sim_identity_const={sctn_pdm_params['identity_const']}, input_amp={input_amp})",
             fontsize=16)

axs[0].plot(time_vector_analog, sim.data[probe_analog_input], label="Analog Input", color="blue")
axs[0].set_title("Original Analog Input Signal")
axs[0].set_ylabel("Amplitude")
axs[0].legend(loc='upper right')
axs[0].grid(True)

axs[1].plot(time_vector_analog, sim.data[probe_pdm_voltage][:, 0], label=f"Neuron 0 Voltage", color="red")
axs[1].set_title("PDM Neuron Membrane Voltage")
axs[1].set_ylabel("Voltage (V)")
axs[1].axhline(0, color='grey', linestyle=':', linewidth=0.8)
axs[1].axhline(sctn_pdm_params["identity_const"], color='grey', linestyle='--', linewidth=0.8,
               label=f"Identity Const (sim)")
axs[1].axhline(-sctn_pdm_params["identity_const"], color='grey', linestyle='--', linewidth=0.8)
axs[1].legend(loc='upper right')
axs[1].grid(True)

axs[2].plot(time_vector_analog, sim.data[probe_pdm_rand_gauss][:, 0], label=f"Neuron 0 RandGaussVar", color="green")
axs[2].set_title("PDM Neuron 'rand_gauss_var_state' (Accumulator)")
axs[2].set_ylabel("Accumulator Value")
axs[2].axhline(65536, color='grey', linestyle='--', linewidth=0.8, label="Overflow (65536)")
axs[2].legend(loc='upper right')
axs[2].grid(True)

plot_idx_spikes = 3
if N_PDM_NEURONS == 1:
    axs[3].plot(time_vector_spikes, sim.data[probe_pdm_spikes][:, 0], label="PDM Spikes", color="purple",
                drawstyle="steps-post")
else:
    from nengo.utils.matplotlib import rasterplot

    rasterplot(time_vector_spikes, sim.data[probe_pdm_spikes], ax=axs[plot_idx_spikes],
               colors=['purple'] * N_PDM_NEURONS)
axs[plot_idx_spikes].set_title("PDM Neuron Spike Output")
axs[plot_idx_spikes].set_ylabel(f"Spike (0 or {int(pdm_ensemble.neuron_type.amplitude / DT)})")
if N_PDM_NEURONS == 1: axs[plot_idx_spikes].legend(loc='upper right')
axs[plot_idx_spikes].grid(True)
axs[plot_idx_spikes].set_xlabel("Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
