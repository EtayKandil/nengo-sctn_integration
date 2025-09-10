# nengo-sctn_integration
This repository explores a spiking continuous-time neuron (SCTN) integrated into Nengo (NEF). We implement a two-population harmonic oscillator and instrument the code with hardware offload hooks so that heavy compute steps can later be accelerated on FPGA (or emulated). The notebook demonstrates configuration, simulation, and basic analysis (time-series and—optionally—phase portrait).



# TL;DR

SCTN implemented as a custom NeuronType for Nengo

Two-ensemble harmonic oscillator (NEF) with recurrent transform

Clear hand-off points for future FPGA acceleration

Jupyter notebook for quick demo + plots

# Features

Native Nengo integration: SCTN as a drop-in NeuronType, working with Ensemble, Connection, and Probe.

Harmonic oscillator model: Classic 2-D oscillator built from two reciprocally coupled ensembles.

FPGA readiness: Abstractions (simulator/network/ensemble wrappers) to route heavy steps to hardware.

Reproducible demo: One notebook (oscillator_sctn.ipynb) that builds, runs, and visualizes the model.

# Installation

Recommended: Python 3.8+ in a virtual environment.

git clone https://github.com/EtayKandil/nengo-sctn_integration.git
cd nengo-sctn_integration

python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Minimal dependencies:
pip install nengo numpy matplotlib jupyter
# or, if present:
# pip install -r requirements.txt

# Quickstart (Notebook)
jupyter notebook oscillator_sctn.ipynb


In the notebook, run cells top-to-bottom:

Imports & Config

FPGA Hooks (Classes) – simulator/network/ensemble wrappers

Simulation Parameters – n_neurons, omega, T, etc.

Model Definition – two ensembles (z1, z2) + recurrent transform

Run & Probes – simulate and collect signals

Plots – time-series of 
𝑥
1
(
𝑡
)
,
𝑥
2
(
𝑡
)
x
1
	​

(t),x
2
	​

(t) (and optional phase portrait)




# FPGA Offload (Road to Hardware)

The repository includes hand-off points (e.g., neuron state update / integration) where you can later redirect computation to an FPGA accelerator. Current behavior is software/“emulation”; next steps usually include:

RTL/driver integration (e.g., AXI-Lite/PCIe as applicable)

End-to-end data path for spike/state updates

Benchmarking: wall-time speedup, energy per step, accuracy vs. software



(Adjust names to match your actual layout.)

Troubleshooting

ImportError: No module named 'nengo'
Ensure you installed into the active venv: pip install nengo and check which python / which pip.

Plots don’t appear
Run all cells, verify variable names used in plotting match the notebook’s probes/arrays.

Python version mismatch
Verify with python --version. On Windows, WSL/Ubuntu or a clean Python 3.8+ often simplifies setup.



