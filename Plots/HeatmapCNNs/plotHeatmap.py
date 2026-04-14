import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

occupations = [10,50,80,100]
fontSize = 30

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)
base_path = os.path.dirname(os.path.dirname(path))
plt.rcParams['savefig.directory'] = os.path.dirname(path)


def load_data_for_occupation(occupation):
    """Carrega dados para uma ocupação específica"""
    data_dict = {}
    for i in range(1, 8):
        if i==1 or i==2 or i==4 or i==6 or i==7:
            results_ocupacao = 'results_occupation'
        else:
            results_ocupacao = 'results_ocupacao'

        cnn_path = os.path.join(base_path,f'RedeNeuralConvolucional', f"CNN_{i}", f"{results_ocupacao}_{occupation}.npz")
        cnn_data = np.load(cnn_path)
        data_dict[f'CNN_{i}'] = {
            'rms': cnn_data['rms'],
            'r2': cnn_data['r2'], 
            'mae': cnn_data['mae'],
            'medae': cnn_data['medae']
        }
    return data_dict

def create_heatmap_data(data_dict):
    """Cria matriz para o heatmap: 7 CNNs x 4 métricas"""
    cnn_names = [f'CNN-{i}' for i in range(1, 8)]
    metrics = ['rms', 'r2', 'mae', 'medae']
    
    heatmap_data = np.zeros((7, 4))
    
    for i, cnn in enumerate(cnn_names):
        for j, metric in enumerate(metrics):
            heatmap_data[i, j] = data_dict[cnn.replace('-', '_')][metric]
    
    return heatmap_data, cnn_names, metrics

def plot_heatmaps_individual_colorbar():
    """Plota os 4 heatmaps com colorbar individual para cada um"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = ['#FFFFFF', '#F0F0F0', '#D0D0D0', '#B0B0B0', '#909090', '#707070', '#505050', '#303030', '#101010', '#000000']
    cmap = LinearSegmentedColormap.from_list('custom_gray', colors, N=100)
    
    for idx, occupation in enumerate(occupations):
        data_dict = load_data_for_occupation(occupation)
        heatmap_data, cnn_names, metrics = create_heatmap_data(data_dict)
        
        n_cnns, n_metrics = heatmap_data.shape

        vmin, vmax = np.min(heatmap_data), np.max(heatmap_data)
        
        im = axes[idx].imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

        axes[idx].set_xticks(range(n_metrics))
        axes[idx].set_yticks(range(n_cnns))
        axes[idx].set_xticklabels(['RMS', 'R²', 'MAE', 'MedAE'], fontsize=fontSize)
        axes[idx].set_yticklabels(cnn_names, fontsize=fontSize)
        
        for i in range(n_cnns):
            for j in range(n_metrics):
                cell_value = heatmap_data[i, j]
                normalized_value = (cell_value - vmin) / (vmax - vmin)
                text_color = 'white' if normalized_value > 0.6 else 'black'
                
                axes[idx].text(j, i, f'{heatmap_data[i, j]:.4f}',
                             ha='center', va='center', 
                             color=text_color, fontsize=fontSize-6,
                             fontweight='bold')
        
        axes[idx].set_title(f'Occupancy {occupation}%', fontsize=fontSize, fontweight='bold', pad=20)
        
        # Adicionar grid
        axes[idx].set_xticks(np.arange(-0.5, n_metrics, 1), minor=True)
        axes[idx].set_yticks(np.arange(-0.5, n_cnns, 1), minor=True)
        axes[idx].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        cbar = plt.colorbar(im, ax=axes[idx], shrink=1.0, pad=0.02)
        cbar.ax.tick_params(labelsize=fontSize)
        cbar.set_label('Metric magnitude', fontsize=fontSize-2, rotation=270, labelpad=25)
    
    plt.tight_layout()
    plt.show()


plot_heatmaps_individual_colorbar() 
