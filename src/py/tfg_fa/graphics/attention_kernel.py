# Coste del modulo de atencion en terminos de memoria: Naive vs flash attention
# Este modulo genera dos graficas para mostrar el coste de las trasferencias de memoria
# en una implementacion naive y otra imagen para flash attention.

# El objetivo es ver como crece el coste de la atencion y justificar por que
# evitar materializar las matrices es beneficioso.

# El coste de la atencion naive es:
# 1- Multiplicar Q y K
# A = Q * K^T,
# La matriz A tiene tamano seq_len * seq_len
# La matriz Q tiene tamano seq_len * head_dim
# La matriz K tiene tamano seq_len * head_dim
# El coste de esta operacion es la suma de los tamanos de la matriz

# 2- Identificar el maximo de cada row
# B, maximo = A
# B es de las mismas dimensiones de P,
# El coste es seq_len * seq_len + seq_len (el vector de maximos).

# 3- Restar maximo a elemntos de la funcion
# C = B - maximo
# C es de las mismas dimensiones de B
# El coste es 2 * seq_len * seq_len + seq_len (el vector de maximos)

# 4- Reescale por 1/sqrt(d)
# D = C * constant
# El coste es 2 * seq_len * seq_len (se asume que la constante se
# interpreta como un inmediato)

# 5- Calcular el softmax
# E = D
# El coste es 2 * seq_len * seq_len + seq_len


# 6- Multiplicar por V
# Resultado = E * V
# El coste es seq_len * seq_len + 2 * seq_len * head_dim


# El coste de la atencion, expresada como el flash attention:
# Resultado = (Q * K / sqrt(d))* V
# El coste es 4 * seq_len * head_dim

# COSTE TOTAL NAIVE:
#
#   9 * seq_len^2 + 4 * seq_len * d + 3 * seq_len
#
# Por tanto, para d constante, el termino dominante es:
#
#   O(seq_len^2)


# FLASHA TTENTION
#
# En el modelo idealizado, FlashAttention evita materializar las matrices
# intermedias de dimensiones seq_len * seq_len.
#
# Se leen:
#   Q: seq_len * d
#   K: seq_len * d
#   V: seq_len * d
# Se escribe:
#   O: seq_len * d
#
# Coste idealizado:
#
#   4 * seq_len * d
#
# Para d constante:
#
#   O(seq_len)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# Modelo de coste idealizado
# ============================================================

def naive_attention_cost(seq_len, head_dim):
    """
    Coste idealizado de movimiento de datos de la atencion naive.

    Formula:
        C_naive = 9*n^2 + 4*n*d + 3*n
    """
    n = seq_len
    d = head_dim

    return 9 * n * n + 4 * n * d + 3 * n


def flash_attention_cost(seq_len, head_dim):
    """
    Coste idealizado de movimiento de datos de FlashAttention.

    Formula:
        C_flash = 4*n*d
    """
    n = seq_len
    d = head_dim

    return 4 * n * d


# ============================================================
# Configuracion
# ============================================================

HEAD_DIM = 128

# Valores que se muestran en la grafica
SEQ_LENS_PLOT = np.array([
    8192,
    16384,
    32768,
    65536,
])

# Valores que se imprimen por consola
SEQ_LENS_ALL = np.array([
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
])

NVIDIA_GREEN = "#76B900"
NAVY_BLUE = "#0B1F5E"


# ============================================================
# Calculo para la grafica
# ============================================================

naive_plot = np.array([
    naive_attention_cost(seq_len, HEAD_DIM)
    for seq_len in SEQ_LENS_PLOT
])

flash_plot = np.array([
    flash_attention_cost(seq_len, HEAD_DIM)
    for seq_len in SEQ_LENS_PLOT
])

# Millones de elementos
naive_plot_m = naive_plot / 1e6
flash_plot_m = flash_plot / 1e6


# ============================================================
# Formateador del eje Y
# ============================================================

def format_axis_value(value, _):
    if value >= 1000:
        return f"{value / 1000:g} B"

    return f"{value:g} M"


# ============================================================
# Grafica
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6.5))

ax.plot(
    SEQ_LENS_PLOT,
    naive_plot_m,
    marker="o",
    markersize=8,
    linewidth=3,
    color=NAVY_BLUE,
    label="Atención naïve",
)

ax.plot(
    SEQ_LENS_PLOT,
    flash_plot_m,
    marker="o",
    markersize=8,
    linewidth=3,
    color=NVIDIA_GREEN,
    label="FlashAttention",
)


# ============================================================
# Eje X
# ============================================================

ax.set_xticks(SEQ_LENS_PLOT)

ax.set_xticklabels(
    [
        "8192",
        "16384",
        "32768",
        "65536",
    ],
    fontsize=10,
)

ax.set_xlabel(
    "Longitud de secuencia",
    fontsize=12,
)


# ============================================================
# Eje Y
# ============================================================

ax.yaxis.set_major_formatter(
    FuncFormatter(format_axis_value)
)

ax.set_ylabel(
    "Elementos movidos",
    fontsize=12,
)

ax.set_ylim(
    0,
    naive_plot_m[-1] * 1.08,
)


# ============================================================
# Titulo
# ============================================================

ax.set_title(
    "Comparativa del coste idealizado de movimiento de datos\n"
    r"Atención naïve vs FlashAttention ($\mathrm{head\_dim}=128$)",
    fontsize=14,
    pad=12,
)


# ============================================================
# Estilo
# ============================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.8,
    alpha=0.35,
)

ax.legend(
    fontsize=11,
    loc="upper left",
)

plt.tight_layout()


# ============================================================
# Guardar
# ============================================================

plt.savefig(
    "naive_vs_flash_attention_tfg.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "naive_vs_flash_attention_tfg.pdf",
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Resultados por consola
# ============================================================

for seq_len in SEQ_LENS_ALL:
    naive = naive_attention_cost(seq_len, HEAD_DIM) / 1e6
    flash = flash_attention_cost(seq_len, HEAD_DIM) / 1e6
    ratio = naive / flash

    print(
        f"seq_len = {seq_len:6d} | "
        f"naive = {naive:12.2f} M | "
        f"flash = {flash:10.2f} M | "
        f"ratio = {ratio:8.2f}x"
    )
