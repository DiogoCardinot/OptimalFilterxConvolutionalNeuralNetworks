import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import KFold
import time

"""
CNN Architecture: LargerKernels_32-16  
- Input: (7, 1) - Reshape from 7 features
- Conv1D: 32 filters, kernel_size=5, ReLU, same padding  ← KERNEL 5
- BatchNormalization
- Conv1D: 16 filters, kernel_size=3, ReLU, same padding  ← KERNEL 3
- BatchNormalization
- Flatten
- Dense: 32 units, ReLU
- Dropout: 0.3
- Output: 1 unit (amplitude regression)

Training:
- Optimizer: Adam
- Loss: MSE
- Metrics: MAE
- Epochs: 15
- Batch size: 4096
- K-Folds: 100
"""

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

N_JANELAMENTO=7

base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks","ManipulacaoDados", "DadosConcatenados",f'janelamento_{N_JANELAMENTO}', "dataset_completo.npz")

dataset_output = os.path.join(path, f"CNN_3")

#----------------------------- Data loading -------------------------------------------
data = np.load(dataset_path)

MatrizAmostras = data['MatrizAmostras']
AmplitudeAssociada = data['AmplitudeReal']
Ocupacao = data['Ocupacao']

#--------------------------- PARAMS -----------------------------------------------------

N_JANELAMENTO = 7                   
K_FOLDS = 100                                        
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42
BATCH_SIZE = 4096  
EPOCHS =  15

# ----------------------------- CNN configuration ------------------------------------------
def criar_cnn_config_base():
    """Configuração base: 2 camadas convolucionais + batch norm + dropout"""
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((N_JANELAMENTO, 1), input_shape=(N_JANELAMENTO,)),
        # Primeira camada convolucional
        tf.keras.layers.Conv1D(
            filters=32,           
            kernel_size=5,       
            activation='relu',
            padding='same'       
        ),
        tf.keras.layers.BatchNormalization(),
        
        # Segunda camada convolucional  
        tf.keras.layers.Conv1D(
            filters=16,           
            kernel_size=3,       
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),


        # Camadas densas
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),  # 30% dropout
        tf.keras.layers.Dense(1)       # Output: amplitude
    ])
    
    # Compilar o modelo
    model.compile(
        optimizer='adam',
        loss='mse',               # Mean Squared Error
        metrics=['mae']           # Mean Absolute Error
    )
    
    return model


# ----------------------------- GPU Configuration -----------------------------------------

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Usando RTX 4070 com {gpus[0].name}")


# ----------------------------- K-fold configuration ---------------------------------------
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

results_ocupacoes = {}
stats_ocupacoes = {}
ocupacoes = np.unique(Ocupacao)

# ===================== OTIMIZAÇÃO: Criar mapeamento de índices locais UMA VEZ =====================
print("\n=== CRIANDO MAPEAMENTO DE ÍNDICES LOCAIS ===")
indices_locais_global = np.zeros(len(MatrizAmostras), dtype=np.int64)
ocupacao_limites = {}

for ocupacao in ocupacoes:
    mask_ocupacao = (Ocupacao == ocupacao)
    n_amostras = np.sum(mask_ocupacao)
    indices_locais_global[mask_ocupacao] = np.arange(n_amostras)
    
    ocupacao_limites[ocupacao] = {
        'inicio': np.where(mask_ocupacao)[0][0],
        'fim': np.where(mask_ocupacao)[0][-1] + 1,
        'n_amostras': n_amostras
    }
    print(f"Ocupação {ocupacao}: {n_amostras} amostras (índices locais 0 a {n_amostras-1})")

for ocupacao in ocupacoes:
    results_ocupacoes[ocupacao] = {
        'error': [],
        'real_amplitude': [],
        'estimated_amplitude': [],
        'indices_locais': [] 
    }
    stats_ocupacoes[ocupacao] = {
        'mean_error': 0.0,    
        'std_error': 0.0,    
        'rms': 0.0,
        'r2': 0.0,
        'corr_mean': 0.0,
        'time_mean': 0.0,
        'mae': 0.0, 
        'mape': 0.0, 
        'max_error': 0.0, 
        'medae': 0.0,
        'mean_folds': [],    
        'std_folds': [],
        'rms_folds': [],      
        'r2_folds': [],
        'corr_folds': [],
        'time_folds': [],
        'mae_folds': [], 
        'mape_folds': [], 
        'max_error_folds': [], 
        'medae_folds': []       
    }


for fold, (train_index, test_index) in enumerate(kf.split(MatrizAmostras)):
    print(f"Processando Fold {fold + 1}/{K_FOLDS}")
    # ---------------------------- Otimization -----------------------------------------------

    # stop if there is no improvement in patience epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,           
        restore_best_weights=True
    )
    start_time = time.time()
    X_treino, X_teste = MatrizAmostras[train_index], MatrizAmostras[test_index]
    y_treino, y_teste = AmplitudeAssociada[train_index], AmplitudeAssociada[test_index]
    occ_teste = Ocupacao[test_index]

    print(f"  Treino: {len(X_treino)} amostras")
    print(f"  Teste: {len(X_teste)} amostras")
    print(f"  Ocupacoes no teste: {np.unique(occ_teste)}")

    print(f"  Treinando CNN...")
    modelo = criar_cnn_config_base()

    with tf.device('/GPU:0'):
        historia = modelo.fit(
            X_treino, y_treino,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE, 
            callbacks=[early_stopping],
            validation_split=VALIDATION_SPLIT,
            verbose=1 
        )

    # Fazer predições reais
    y_pred = modelo.predict(X_teste, batch_size=BATCH_SIZE, verbose=0).flatten()

    #Error
    error = y_pred - y_teste

    end_time = time.time()  
    fold_time = end_time - start_time

    for ocupacao in ocupacoes:
        mask = (occ_teste == ocupacao)
        if np.sum(mask) > 0:
            indices_globais = test_index[mask]
            indices_locais = indices_locais_global[indices_globais]  # Mapeamento direto!
            results_ocupacoes[ocupacao]['error'].extend(error[mask].tolist())
            results_ocupacoes[ocupacao]['real_amplitude'].extend(y_teste[mask].tolist())
            results_ocupacoes[ocupacao]['estimated_amplitude'].extend(y_pred[mask].tolist())
            results_ocupacoes[ocupacao]['indices_locais'].extend(indices_locais.tolist())

            error_ocupacao = error[mask]
            y_teste_ocupacao = y_teste[mask]   
            y_pred_ocupacao = y_pred[mask]   
    
            stats_ocupacoes[ocupacao]['mean_folds'].append(np.mean(error_ocupacao))
            stats_ocupacoes[ocupacao]['std_folds'].append(np.std(error_ocupacao))
            
            stats_ocupacoes[ocupacao]['rms_folds'].append(np.sqrt(np.mean(error_ocupacao**2)))
            
            ss_res_ocup = np.sum((y_teste_ocupacao - y_pred_ocupacao)**2)
            ss_tot_ocup = np.sum((y_teste_ocupacao - np.mean(y_teste_ocupacao))**2)

            if len(y_teste_ocupacao) <= 1 or ss_tot_ocup < 1e-12:
                r2 = 0.0 
            else:
                r2 = 1 - (ss_res_ocup / ss_tot_ocup)

            r2 = np.clip(r2, -100, 1) 
            stats_ocupacoes[ocupacao]['r2_folds'].append(r2)
            
            if len(y_teste_ocupacao) > 1 and np.std(y_teste_ocupacao) > 1e-12 and np.std(y_pred_ocupacao) > 1e-12:
                corr = np.corrcoef(y_teste_ocupacao, y_pred_ocupacao)[0, 1]
            else:
                corr = 0.0
                
            stats_ocupacoes[ocupacao]['corr_folds'].append(corr)
            stats_ocupacoes[ocupacao]['time_folds'].append(fold_time)

            stats_ocupacoes[ocupacao]['mae_folds'].append(np.mean(np.abs(error_ocupacao)))
            mape = np.mean(np.abs(error_ocupacao / (y_teste_ocupacao + 1e-8))) * 100
            stats_ocupacoes[ocupacao]['mape_folds'].append(mape)

            stats_ocupacoes[ocupacao]['max_error_folds'].append(np.max(np.abs(error_ocupacao)))

            stats_ocupacoes[ocupacao]['medae_folds'].append(np.median(np.abs(error_ocupacao)))

            print(f"Ocupacao: {ocupacao} -> Media: {np.mean(error_ocupacao):.4f}, STD: {np.std(error_ocupacao):.4f}, R^2: {stats_ocupacoes[ocupacao]['r2_folds'][-1]:.4f}")

    
    print(f"  Fold {fold + 1} concluido")
    print("-" * 50)

# ===================== CALCULAR ESTATÍSTICAS FINAIS =====================
print("\n=== CALCULANDO ESTATÍSTICAS FINAIS ===")
for ocupacao in ocupacoes:
    if len(stats_ocupacoes[ocupacao]['mean_folds']) > 0:
        # Média das médias dos folds
        stats_ocupacoes[ocupacao]['mean_error'] = np.mean(stats_ocupacoes[ocupacao]['mean_folds'])
        # Std dos stds dos folds (mais robusto)
        stats_ocupacoes[ocupacao]['std_error'] = np.mean(stats_ocupacoes[ocupacao]['std_folds'])
        stats_ocupacoes[ocupacao]['rms'] = np.mean(stats_ocupacoes[ocupacao]['rms_folds'])
        stats_ocupacoes[ocupacao]['r2'] = np.mean(stats_ocupacoes[ocupacao]['r2_folds'])
        stats_ocupacoes[ocupacao]['corr_mean'] = np.mean(stats_ocupacoes[ocupacao]['corr_folds'])
        stats_ocupacoes[ocupacao]['time_mean'] = np.mean(stats_ocupacoes[ocupacao]['time_folds'])
        stats_ocupacoes[ocupacao]['mae'] = np.mean(stats_ocupacoes[ocupacao]['mae_folds'])
        stats_ocupacoes[ocupacao]['mape'] = np.mean(stats_ocupacoes[ocupacao]['mape_folds'])
        stats_ocupacoes[ocupacao]['max_error'] = np.mean(stats_ocupacoes[ocupacao]['max_error_folds'])
        stats_ocupacoes[ocupacao]['medae'] = np.mean(stats_ocupacoes[ocupacao]['medae_folds'])

print("\n=== RESUMO FINAL ===")
for ocupacao in ocupacoes:
    n_amostras = len(results_ocupacoes[ocupacao]['error'])
    print(f"Ocupação {ocupacao}: {n_amostras:,} amostras | MAE: {stats_ocupacoes[ocupacao]['mae']:.4f} | R²: {stats_ocupacoes[ocupacao]['r2']:.4f}")


# --------------------------------------- Save Data ---------------------------------------
os.makedirs(dataset_output, exist_ok=True)

# Salvar resultados por ocupação
for ocupacao in ocupacoes:
    if len(results_ocupacoes[ocupacao]['error']) > 0:
        indices_locais = np.array(results_ocupacoes[ocupacao]['indices_locais'], dtype=np.int64)
        error_array = np.array(results_ocupacoes[ocupacao]['error'], dtype=np.float64)
        real_amplitude_array = np.array(results_ocupacoes[ocupacao]['real_amplitude'], dtype=np.float64)
        estimated_amplitude_array = np.array(results_ocupacoes[ocupacao]['estimated_amplitude'], dtype=np.float32)
        
        # Verificação de integridade
        print(f"\nSalvando ocupação {ocupacao}:")
        print(f"  Índices locais - shape: {indices_locais.shape}, min: {indices_locais.min()}, max: {indices_locais.max()}")
        print(f"  Range esperado: 0 a {ocupacao_limites[ocupacao]['n_amostras']-1}")
        
        if indices_locais.max() < ocupacao_limites[ocupacao]['n_amostras']:
            print(f"  Índices locais dentro do range esperado!")
        else:
            print(f"   ATENÇÃO: Índices locais fora do range esperado!")

        np.savez(
            os.path.join(dataset_output, f"results_ocupacao_{ocupacao}.npz"),
            error=np.array(results_ocupacoes[ocupacao]['error']),
            real_amplitude=np.array(results_ocupacoes[ocupacao]['real_amplitude']),
            estimated_amplitude=np.array(results_ocupacoes[ocupacao]['estimated_amplitude']),
            indices=indices_locais,
            mean_error=stats_ocupacoes[ocupacao]['mean_error'], 
            std_error=stats_ocupacoes[ocupacao]['std_error'],
            rms =  stats_ocupacoes[ocupacao]['rms'] ,
            r2 = stats_ocupacoes[ocupacao]['r2'],
            corr_mean = stats_ocupacoes[ocupacao]['corr_mean'],
            time_mean = stats_ocupacoes[ocupacao]['time_mean'],
            mae=stats_ocupacoes[ocupacao]['mae'],
            mape=stats_ocupacoes[ocupacao]['mape'],
            max_error=stats_ocupacoes[ocupacao]['max_error'],
            medae=stats_ocupacoes[ocupacao]['medae']
        )

with open(os.path.join(dataset_output, "model_config.txt"), "w") as f:
    f.write("CNN Configuration:\n")
    f.write("- 2 convolutional layers (32, 16 filters)\n")
    f.write("- BatchNormalization after each conv layer\n")
    f.write("- Dense layer with 32 units + Dropout 0.3\n")
    f.write("- Output: 1 unit (amplitude)\n")
    f.write(f"- K-Folds: {K_FOLDS}\n")
    f.write(f"- Epochs: {EPOCHS}\n")
    f.write(f"- Batch size: {BATCH_SIZE}\n")

print(f"\n✅ Resultados salvos em: {dataset_output}")
print("=" * 50)