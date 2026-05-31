# TFG: FlashAttention Implementation (C++ / CUDA)

Bachelor's Thesis (Trabajo de Fin de Grado)  
Author: Ángel Moreno Dominguez (@angelMDUex)  
Institution: Universidad de Extremadura (UEx)

## Overview

This project is an implementation of GPT2 to familiarize myself with modern deep learning concepts. Multiple versions of flash attention have been implemented too, in CUDA, OpenAi's Triton and standard pytorch implementations. The objectives of the project were to understand modern deep learning and understand low-level code optimizations in GPUs. 

The full written thesis document is available in [Final.pdf](./Final.pdf).

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
