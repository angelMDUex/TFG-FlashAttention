# TFG: FlashAttention Implementation (C++ / CUDA)

Bachelor's Thesis (Trabajo de Fin de Grado)  
Author: Ángel Moreno Dominguez (@angelMDUex)  
Institution: Universidad de Extremadura (UEx)

## Overview

This repository contains the source code for my Bachelor's Thesis (Trabajo de Fin de Grado), which centers on implementing a GPT-2 architecture from scratch to explore modern deep learning concepts and low-level hardware acceleration. 

A core focus of this project is the implementation and evaluation of the FlashAttention mechanism across different paradigms. This repository includes versions implemented in:
* **Standard PyTorch** (for baseline comparison)
* **OpenAI Triton** (for high-level GPU programming)
* **Custom CUDA** (for low-level hardware optimization)

The project was developed with two primary objectives:
1. To gain a comprehensive understanding of modern deep learning architectures and transformer mechanics.
2. To master low-level code optimizations, memory tiling, and SRAM management on GPU hardware.

The full written thesis document detailing the mathematical foundations and performance benchmarks is available in [Final.pdf](./Final.pdf).

---

## Environment Setup

This project uses `uv` for dependency and virtual environment management. It is a fast virtual environment manager, and quite simpe both to setup and to use.

To set up the environment locally:

```bash
# Clone the repository
git clone [https://github.com/angelMDUex/TFG-FlashAttention.git](https://github.com/angelMDUex/TFG-FlashAttention.git)
cd TFG-FlashAttention

# Create and activate the virtual environment
uv venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Run
`make test` : Tests the implementations. For debugging pourposes or simply to check if we have correctly installed GPU sdks
`make inference` : Actually run and profile all GPT2 versions, from small to XL.
`make clean`: Clean the intermediate compilation files and GPT2 data.
`make install`: Compile the cuda implementation and integrate it with Python via pybind11.
