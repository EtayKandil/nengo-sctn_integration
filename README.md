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




