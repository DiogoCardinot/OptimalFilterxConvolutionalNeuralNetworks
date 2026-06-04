import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Dados fictícios
estiramento = np.linspace(1, 2, 50)

# Modelo Mooney-Rivlin (curva convexa)
tensao_mr = 0.5 * (estiramento**2 - 1/estiramento) + 0.1 * (estiramento - 1)

# Modelo Lopez-Pamies (ligeiramente diferente)
tensao_lp = 0.55 * (estiramento**2 - 1/estiramento) + 0.08 * (estiramento - 1) + 0.02 * (estiramento - 1)**3

# Modelo Anssari-Benam (com inflexão no final)
tensao_ab = 0.52 * (estiramento**2 - 1/estiramento) + 0.09 * (estiramento - 1) + 0.03 * np.sin(5 * (estiramento - 1))

# Criar figura principal
fig, ax = plt.subplots(figsize=(8, 6))

# Plotar as curvas
ax.plot(estiramento, tensao_mr, 'b-', label='Mooney-Rivlin', linewidth=2)
ax.plot(estiramento, tensao_lp, 'r--', label='Lopez-Pamies', linewidth=2)
ax.plot(estiramento, tensao_ab, 'g-.', label='Anssari-Benam', linewidth=2)

# Configurar eixos principais
ax.set_xlabel('Estiramento', fontsize=12)
ax.set_ylabel('Tensão de Compressão (MPa)', fontsize=12)
ax.set_xlim(1.0, 2.0)
ax.set_ylim(0.8, 2.1)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

# --- Área de zoom (inset) ---
# Definir a região a ser ampliada
x1, x2 = 1.7, 1.8
y1, y2 = 1.3, 1.5

# Criar eixos internos
axins = inset_axes(ax, width="18%", height="18%", loc='lower right', borderpad=0.3)

axins.set_xticks([])  # remove ticks do x
axins.set_yticks([])  # remove ticks do y
for spine in axins.spines.values():
    spine.set_linewidth(0.5)  # linhas finas
    spine.set_color('black')

# Plotar as mesmas curvas dentro do zoom
axins.plot(estiramento, tensao_mr, 'b-', linewidth=2)
axins.plot(estiramento, tensao_lp, 'r--', linewidth=2)
axins.plot(estiramento, tensao_ab, 'g-.', linewidth=2)

# Definir limites da área ampliada
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Opcional: marcar os ticks internos
# axins.tick_params(axis='both', labelsize=8)

# Desenhar retângulo na área principal e linhas conectando ao zoom
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="gray", linewidth=1.5)

# Título geral (opcional, você pode colocar como título do gráfico principal)
ax.set_title('Ajuste de dados dos três modelos', fontsize=14, pad=15)

plt.tight_layout()
plt.show()