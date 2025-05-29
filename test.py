import numpy as np
import nengo
from nengo.neurons import NeuronType, settled_firingrate  # Import settled_firingrate
from nengo.dists import Choice
from nengo.utils.numpy import clip
import matplotlib.pyplot as plt

# --- SCTNNeuronType Definition (from previous step) ---
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
        J_all_samples = self.current(x, gain, bias)  # J can be (n_samples, n_neurons) or (n_neurons,)

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
            lf = float(2**(int(self.leakage_factor_val) - 3))
            effective_input_current = current_plus_theta * lf

        voltage += effective_input_current * dt
        voltage[:] = clip(voltage, self.min_clip_val, self.max_clip_val)

        spiked_mask = np.zeros_like(voltage, dtype=bool)
        if self.current_activation_fn_id == SCTNNeuronType.ACT_BINARY:
            spiked_mask = voltage > self.threshold_pulse_val

        leak_now_mask = leakage_timer >= self.leakage_period_val
        if self.leakage_factor_val > 0:
            decay_multiplier = (1.0 - (1.0 / (2**int(self.leakage_factor_val))))
            voltage[leak_now_mask] *= decay_multiplier

        leakage_timer[leak_now_mask] = 0
        leakage_timer[~leak_now_mask] += 1

        output[:] = 0.0
        output[spiked_mask] = self.amplitude_val / dt

        if self.membrane_should_reset_val:
            voltage[spiked_mask] = self.reset_to_val

# --- Model Setup ---
with nengo.Network(seed=154) as model:
    zero_input_node = nengo.Node(output=-0.05)
    sctn_params = {
        "theta": 0.0,
        "leakage_period": 3,
        "leakage_factor": 2,
        "threshold_pulse": 0.0005,
        "reset_to": 0.0,
        "activation_function": "BINARY",
        "membrane_should_reset": True,
        "min_clip": -10.0,
        "max_clip": 110.0
    }
    our_sctn_ensemble = nengo.Ensemble(
        n_neurons=1,
        dimensions=1,
        neuron_type=SCTNNeuronType(**sctn_params),
        gain=nengo.dists.Choice([1.0]),
        bias=nengo.dists.Choice([0.0]),
        label="SCTN_Test_Ensemble"
    )
    nengo.Connection(zero_input_node, our_sctn_ensemble, synapse=None)
    probe_J_input = nengo.Probe(our_sctn_ensemble.neurons, 'input')
    probe_voltage = nengo.Probe(our_sctn_ensemble.neurons, 'voltage')
    probe_spikes = nengo.Probe(our_sctn_ensemble.neurons, 'output')
    probe_leakage_timer = nengo.Probe(our_sctn_ensemble.neurons, 'leakage_timer')

sim_dt = 0.001
simulation_time = 1
with nengo.Simulator(model, dt=sim_dt) as sim:
    sim.run(simulation_time)

# --- Plotting ---
time_vector = sim.trange()
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.suptitle("SCTNNeuronType Test: Zero Input & Leakage", fontsize=16)
axs[0].plot(time_vector, sim.data[probe_J_input], label="Input Current J to Neuron", color='black')
axs[0].set_title("Actual Input Current (J) to Neuron's Step Method")
axs[0].set_ylabel("Current")
axs[0].legend(loc='upper right')
axs[0].grid(True)
axs[0].set_ylim(-0.1, 0.1)
axs[1].plot(time_vector, sim.data[probe_voltage], label="Neuron Voltage", color='red')
axs[1].set_title("Neuron Membrane Voltage")
axs[1].set_ylabel("Voltage (V)")
axs[1].axhline(0, color='grey', linestyle=':', linewidth=0.8)
axs[1].legend(loc='upper right')
axs[1].grid(True)
axs[2].plot(time_vector, sim.data[probe_leakage_timer], label="Leakage Timer", color='green', linestyle='--')
axs[2].set_title("Neuron Leakage Timer State")
axs[2].set_ylabel("Timer Value")
axs[2].set_yticks(np.arange(0, sctn_params["leakage_period"] + 2))
axs[2].legend(loc='upper right')
axs[2].grid(True)
axs[3].plot(time_vector, sim.data[probe_spikes], label="Neuron Spikes", color='purple')
axs[3].set_title("Neuron Output Spikes")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel(f"Spike Output (0 or {int(1/sim_dt)})")
axs[3].legend(loc='upper right')
axs[3].grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Verification Print ---
print(f"--- SCTN Neuron Parameters ---")
for key, value in sctn_params.items():
    print(f"{key}: {value}")
print(f"\n--- Initial State & Early Steps (dt={sim_dt}s) ---")
print(f"Time (s) | Input J | Voltage | Leakage Timer | Spikes")
print("--------------------------------------------------------------")
for i in range(min(10, len(time_vector))):
    t = time_vector[i]
    j_in = sim.data[probe_J_input][i, 0]
    v = sim.data[probe_voltage][i, 0]
    timer_val = sim.data[probe_leakage_timer][i, 0]
    spike_val = sim.data[probe_spikes][i, 0]
    print(f"{t:8.4f} | {j_in:7.4f} | {v:7.4f} | {timer_val:13d} | {spike_val:6d}")
initial_voltage = sim.data[probe_voltage][0, 0]
if np.isclose(initial_voltage, 0.0, atol=1e-6):
    print("\nSUCCESS: Neuron voltage starts at (or very near) 0.0 as expected.")
else:
    print(f"\nNOTE: Neuron voltage at t={time_vector[0]}s is {initial_voltage:.4f}, not 0.0. Initial jump may still be present.")
