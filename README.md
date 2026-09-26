# TFG - FlashAttention en GPU

Implementación y optimización de **FlashAttention no causal** en **CUDA** y **Triton** para GPUs NVIDIA Ampere, con comparación frente a **Torch SDPA** y **cuDNN SDPA**.

![Sobrecoste temporal respecto a la implementación más rápida](flashattention_resultado_tfg.png)

El repositorio incluye el código fuente, resultados de Optuna, informes de Nsight Compute (`.ncu-rep`) y scripts de validación y perfilado.
