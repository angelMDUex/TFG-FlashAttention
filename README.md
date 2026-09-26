# TFG - FlashAttention

Implementación y estudio de FlashAttention no causal en CUDA y Triton para GPUs NVIDIA Ampere, con comparación frente a Torch SDPA y cuDNN SDPA.

## Uso

```bash
uv run python src/py/cuda_optimizer.py
./profile.sh
./validez.sh
```
El repositorio incluye código fuente, resultados de Optuna, informes de Nsight Compute (.ncu-rep) y scripts de validación y perfilado.
