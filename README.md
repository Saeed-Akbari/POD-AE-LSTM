# POD-AE-LSTM

"Numerical assessments of a nonintrusive surrogate model based on recurrent neural networks and proper orthogonal decomposition: Rayleigh Bénard convection"

Authors: Saeed Akbari, Suraj Pawar, Omer San

Journal: International Journal of Computational Fluid Dynamics

# Dataset

All the files required to simulate Rayleigh-Bénard convection are in the FOM folder.
Inputs of the full order model is in a yaml file ("FOM/config/rbc_parameters.yaml"). After specifying desired inputs, "FOM/ns2d_ws_rbc.py" must be executed to simulate flow.
The FOM data must be collected to be used for building the nonintrusive reduced order model.
Details of the numerical schemes are in the paper.

# Building NIROM

Training and testing the NIROM including POD, AE, and LSTM can be done by running "ROM/PODAE.py" file. Inputs can be specified in "ROM/input.yaml". Variable "mode" in the input file manages training and testing modes for autoencoder and LSTM networks.
After training some models, the script "ROM/sd.py" gets inputs of the models from "ROM/DAinput/" files and visualizes mean value and $\pm 2$ times of standard deviation.
