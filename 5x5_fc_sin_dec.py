import numpy as np
import nengo
from nengo.neurons import NeuronType, settled_firingrate  # Import settled_firingrate
from nengo.dists import Choice
from nengo.utils.numpy import clip
import matplotlib.pyplot as plt


# --- SCTNNeuronType Definition (from your working script) ---
class SCTNNeuronType(NeuronType):
    ACT_IDENTITY = 0
    ACT_BINARY = 1
    ACT_SIGMOID = 2

    state = {
        "voltage": Choice([0.0]),
        "leakage_timer": Choice([0]),
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

        if self.activation_function_str == "BINARY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_BINARY
        elif self.activation_function_str == "IDENTITY":
            self.current_activation_fn_id = SCTNNeuronType.ACT_IDENTITY
            raise NotImplementedError("IDENTITY activation not fully implemented yet.")
        elif self.activation_function_str == "SIGMOID":
            self.current_activation_fn_id = SCTNNeuronType.ACT_SIGMOID
            raise NotImplementedError("SIGMOID activation not fully implemented yet.")
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

    def step(self, dt, J, output, voltage, leakage_timer):
        current_plus_theta = J + self.theta_val
        if self.leakage_factor_val < 3:
            effective_input_current = current_plus_theta
        else:
            lf = float(2 ** (int(self.leakage_factor_val) - 3))
            effective_input_current = current_plus_theta * lf

        voltage += effective_input_current * dt
        voltage[:] = clip(voltage, self.min_clip_val, self.max_clip_val)

        spiked_mask = np.zeros_like(voltage, dtype=bool)
        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            spiked_mask = voltage > self.threshold_pulse_val

        leak_now_mask = leakage_timer >= self.leakage_period_val
        if self.leakage_factor_val > 0:
            decay_multiplier = (1.0 - (1.0 / (2 ** int(self.leakage_factor_val))))
            voltage[leak_now_mask] *= decay_multiplier

        leakage_timer[leak_now_mask] = 0
        leakage_timer[~leak_now_mask] += 1

        output[:] = 0.0
        output[spiked_mask] = self.amplitude_val / dt

        if self.membrane_should_reset_val:
            voltage[spiked_mask] = self.reset_to_val


# --- End of SCTNNeuronType Definition ---

# --- Model Setup ---
N_NEURONS = 5

with nengo.Network(seed=123) as model:
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

    sine_freq = 5
    input_node_a = nengo.Node(output=lambda t: np.sin(2 * np.pi * sine_freq * t))

    ensemble_a = nengo.Ensemble(
        n_neurons=N_NEURONS,
        dimensions=1,
        neuron_type=SCTNNeuronType(**sctn_params),
        gain=nengo.dists.Choice([1.0] * N_NEURONS),
        bias=nengo.dists.Choice([0.0] * N_NEURONS),
        label="Ensemble_A"
    )
    nengo.Connection(input_node_a, ensemble_a, synapse=None)

    ensemble_b = nengo.Ensemble(
        n_neurons=N_NEURONS,
        dimensions=1,
        neuron_type=SCTNNeuronType(**sctn_params),
        gain=nengo.dists.Choice([1.0] * N_NEURONS),
        bias=nengo.dists.Choice([0.0] * N_NEURONS),
        label="Ensemble_B"
    )

    direct_conn_ab = nengo.Connection(
        ensemble_a.neurons,
        ensemble_b.neurons,
        transform=np.ones((N_NEURONS, N_NEURONS)),  # CHANGED: Fully connected with weights of 1
        synapse=None
    )

    # --- Probes ---
    probe_input_signal = nengo.Probe(input_node_a, 'output')

    probe_a_input_J = nengo.Probe(ensemble_a.neurons, 'input')
    probe_a_voltage = nengo.Probe(ensemble_a.neurons, 'voltage')
    probe_a_spikes = nengo.Probe(ensemble_a.neurons, 'output')

    probe_conn_signal_j_for_b = nengo.Probe(direct_conn_ab)
    probe_b_voltage = nengo.Probe(ensemble_b.neurons, 'voltage')
    probe_b_spikes = nengo.Probe(ensemble_b.neurons, 'output')

    probe_b_decoded_output = nengo.Probe(ensemble_b, synapse=0.01)

# --- Simulation ---
sim_dt = 0.001
simulation_time = 0.5

with nengo.Simulator(model, dt=sim_dt) as sim:
    sim.run(simulation_time)

# --- Plotting ---
time_vector = sim.trange()

fig, axs = plt.subplots(8, 1, figsize=(14, 24), sharex=True)
fig.suptitle(f"SCTNNeuronType: 2-Ensemble ({N_NEURONS}-Neuron) Fully Connected (Sine Input)", fontsize=16)

# 0. Plot Input Sine Wave Signal
axs[0].plot(time_vector, sim.data[probe_input_signal], label="Input Sine Wave", color='magenta')
axs[0].set_title("Input Signal to Ensemble A (Sine Wave)")
axs[0].set_ylabel("Signal Value")
axs[0].legend(loc='upper right')
axs[0].grid(True)

# 1. Plot Ensemble A Input Current (J) - Plotting for first neuron of A
axs[1].plot(time_vector, sim.data[probe_a_input_J][:, 0], label="Neuron A[0] Input J", color='teal')
axs[1].set_title("Ensemble A Neuron[0]: Input Current (J)")
axs[1].set_ylabel("Current (J)")
axs[1].legend(loc='upper right')
axs[1].grid(True)

# 2. Plot Ensemble A Voltage - Plotting for first neuron of A
axs[2].plot(time_vector, sim.data[probe_a_voltage][:, 0], label="Neuron A[0] Voltage", color='orange')
axs[2].set_title("Ensemble A Neuron[0]: Membrane Voltage")
axs[2].set_ylabel("Voltage (V)")
axs[2].axhline(0, color='grey', linestyle=':', linewidth=0.8)
axs[2].axhline(sctn_params["threshold_pulse"], color='grey', linestyle='--', linewidth=0.8,
               label=f"Threshold A ({sctn_params['threshold_pulse']:.4f})")
axs[2].legend(loc='upper right')
axs[2].grid(True)

# 3. Plot Ensemble A Spikes - Plotting for first neuron of A
axs[3].plot(time_vector, sim.data[probe_a_spikes][:, 0], label="Neuron A[0] Spikes", color='blue')
axs[3].set_title("Ensemble A Neuron[0] Output Spikes")
axs[3].set_ylabel(f"Spike (0 or {int(sctn_params['amplitude'] / sim_dt)})")
axs[3].legend(loc='upper right')
axs[3].grid(True)

# 4. Plot Signal on Connection (Input J for Ensemble B's Neuron[0])
# This now represents the sum of outputs from all A neurons (weighted by 1)
axs[4].plot(time_vector, sim.data[probe_conn_signal_j_for_b][:, 0], label="J for Neuron B[0]", color='green')
axs[4].set_title("Signal on Connection (Input Current J for Neuron B[0])")
axs[4].set_ylabel("Current (J)")
axs[4].legend(loc='upper right')
axs[4].grid(True)

# 5. Plot Ensemble B Voltage - Plotting for first neuron of B
axs[5].plot(time_vector, sim.data[probe_b_voltage][:, 0], label="Neuron B[0] Voltage", color='red')
axs[5].set_title("Ensemble B Neuron[0] Membrane Voltage")
axs[5].set_ylabel("Voltage (V)")
axs[5].axhline(0, color='grey', linestyle=':', linewidth=0.8)
axs[5].axhline(sctn_params["threshold_pulse"], color='grey', linestyle='--', linewidth=0.8,
               label=f"Threshold B ({sctn_params['threshold_pulse']:.4f})")
axs[5].legend(loc='upper right')
axs[5].grid(True)

# 6. Plot Ensemble B Spikes - Plotting for first neuron of B
axs[6].plot(time_vector, sim.data[probe_b_spikes][:, 0], label="Neuron B[0] Spikes", color='purple')
axs[6].set_title("Ensemble B Neuron[0] Output Spikes")
axs[6].set_ylabel(f"Spike (0 or {int(sctn_params['amplitude'] / sim_dt)})")
axs[6].legend(loc='upper right')
axs[6].grid(True)

# 7. Plot Ensemble B Decoded Output
axs[7].plot(time_vector, sim.data[probe_b_decoded_output], label="Ensemble B Decoded Output", color='brown')
axs[7].set_title("Ensemble B Decoded Output (Value Represented)")
axs[7].set_xlabel("Time (s)")
axs[7].set_ylabel("Decoded Value")
axs[7].legend(loc='upper right')
axs[7].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Verification Print (Optional) ---
print(f"--- SCTN Neuron Parameters (used for both A and B) ---")
for key, value in sctn_params.items():
    print(f"{key}: {value}")

# Verification for the first neuron in ensemble A
first_a_spike_indices = np.where(sim.data[probe_a_spikes][:, 0] > 0)[0]
if len(first_a_spike_indices) > 0:
    first_a_spike_idx = first_a_spike_indices[0]
    print(f"\n--- Verification at Timestep of First Neuron A[0] Spike ---")
    print(f"Time of first Neuron A[0] spike: {time_vector[first_a_spike_idx]:.4f} s (index {first_a_spike_idx})")

    val_input_signal = sim.data[probe_input_signal][first_a_spike_idx, 0]
    print(f"Input Signal (sine wave) value at this time: {val_input_signal:.4f}")
    val_a_input_j = sim.data[probe_a_input_J][first_a_spike_idx, 0]
    print(f"Neuron A[0] Input J at this time: {val_a_input_j:.4f}")
    val_a_voltage = sim.data[probe_a_voltage][first_a_spike_idx, 0]
    print(f"Neuron A[0] Voltage at this time (after spike & reset): {val_a_voltage:.4f}")

    # Sum of all spikes from ensemble A at this timestep
    sum_a_spikes_at_idx = np.sum(sim.data[probe_a_spikes][first_a_spike_idx, :])
    print(f"Sum of Neuron A spike outputs at this time: {sum_a_spikes_at_idx:.1f}")

    val_j_for_b_neuron0 = sim.data[probe_conn_signal_j_for_b][first_a_spike_idx, 0]
    print(f"Connection signal value (current J for B[0]): {val_j_for_b_neuron0:.1f}")

    # With a fully connected transform of ones, J for any B neuron is the sum of all A spikes
    if np.isclose(sum_a_spikes_at_idx, val_j_for_b_neuron0):
        print("SUCCESS: J for Neuron B[0] matches the SUM of Neuron A's spike output values.")
    else:
        print("NOTE: J for Neuron B[0] might not directly match a single A neuron's spike due to full connectivity.")
        print(f"      (Expected sum from A: {sum_a_spikes_at_idx}, Got J for B[0]: {val_j_for_b_neuron0})")

    if first_a_spike_idx > 0:
        voltage_b_before_spike = sim.data[probe_b_voltage][first_a_spike_idx - 1, 0]
        print(f"Neuron B[0] voltage just BEFORE this J pulse: {voltage_b_before_spike:.4f} V")

    voltage_b_at_spike_time = sim.data[probe_b_voltage][first_a_spike_idx, 0]
    print(f"Neuron B[0] voltage AT TIMESTEP of J pulse: {voltage_b_at_spike_time:.4f} V")

    if sim.data[probe_b_spikes][first_a_spike_idx, 0] > 0:
        print("Neuron B[0] spiked at this timestep due to the J pulse.")
        print(
            f"Neuron B[0] voltage (after spike and reset): {voltage_b_at_spike_time:.4f} V (should be reset_to_val: {sctn_params['reset_to']})")
    else:
        print(
            "Ensemble A's neurons did not spike significantly in the simulation period. Check input J and voltage for neuron 0.")
        if len(time_vector) > 0:
            print(f"Initial Input Signal (sine wave, t={time_vector[0]:.4f}): {sim.data[probe_input_signal][0, 0]:.4f}")
            print(f"Initial J for Neuron A[0] (t={time_vector[0]:.4f}): {sim.data[probe_a_input_J][0, 0]:.4f}")
            print(f"Initial Voltage for Neuron A[0] (t={time_vector[0]:.4f}): {sim.data[probe_a_voltage][0, 0]:.4f}")
