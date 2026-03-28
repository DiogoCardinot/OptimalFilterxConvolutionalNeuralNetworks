import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import KFold
import time

"""
CNN Architecture: DenserClassifier_32-16
- Input: (7, 1) - Reshape from 7 features
- Conv1D: 32 filters, kernel_size=3, ReLU, same padding
- BatchNormalization
- Conv1D: 16 filters, kernel_size=2, ReLU, same padding  
- BatchNormalization
- Flatten
- Dense: 64 units, ReLU  ← LARGER DENSE
- Dropout: 0.3
- Dense: 32 units, ReLU  ← EXTRA DENSE LAYER
- Dropout: 0.2
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

dataset_path = os.path.join(path, "AllSamples_Amplitudes", "dataset_completo.npz")

dataset_output = os.path.join(path, f"CNN_5")

#----------------------------- Data loading -------------------------------------------
data = np.load(dataset_path)

X_total = data['X']
Y_total = data['y']
occ_total = data['occ']

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
            kernel_size=3,       
            activation='relu',
            padding='same'       
        ),
        tf.keras.layers.BatchNormalization(),
        
        # Segunda camada convolucional  
        tf.keras.layers.Conv1D(
            filters=16,           
            kernel_size=2,       
            activation='relu',
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        
        # Camadas densas
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    
    model.compile(
        optimizer='adam',
        loss='mse',               
        metrics=['mae']          
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

results_occupations = {}
stats_occupations = {}
occupations = np.unique(occ_total)

for occupation in occupations:
    results_occupations[occupation] = {
        'error': [],
        'real_amplitude': [],
        'estimated_amplitude': [],
        'indices': []
    }
    stats_occupations[occupation] = {
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


for fold, (train_index, test_index) in enumerate(kf.split(X_total)):
    print(f"Processando Fold {fold + 1}/{K_FOLDS}")
    # ---------------------------- Otimization -----------------------------------------------

    # stop if there is no improvement in patience epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,           
        restore_best_weights=True
    )
    start_time = time.time()
    X_treino, X_teste = X_total[train_index], X_total[test_index]
    y_treino, y_teste = Y_total[train_index], Y_total[test_index]
    occ_teste = occ_total[test_index]

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

    for ocupacao in occupations:
        mask = (occ_teste == ocupacao)
        if np.sum(mask) > 0:
            indices_ocupacao = test_index[mask]
            results_occupations[ocupacao]['error'].extend(error[mask])
            results_occupations[ocupacao]['real_amplitude'].extend(y_teste[mask])
            results_occupations[ocupacao]['estimated_amplitude'].extend(y_pred[mask])

            error_ocupacao = error[mask]
            y_teste_ocupacao = y_teste[mask]   
            y_pred_ocupacao = y_pred[mask]   
    
            stats_occupations[ocupacao]['mean_folds'].append(np.mean(error_ocupacao))
            stats_occupations[ocupacao]['std_folds'].append(np.std(error_ocupacao))
            
            stats_occupations[ocupacao]['rms_folds'].append(np.sqrt(np.mean(error_ocupacao**2)))
            
            ss_res_ocup = np.sum((y_teste_ocupacao - y_pred_ocupacao)**2)
            ss_tot_ocup = np.sum((y_teste_ocupacao - np.mean(y_teste_ocupacao))**2)

            if len(y_teste_ocupacao) <= 1 or ss_tot_ocup < 1e-12:
                r2 = 0.0 
            else:
                r2 = 1 - (ss_res_ocup / ss_tot_ocup)

            r2 = np.clip(r2, -100, 1) 
            stats_occupations[ocupacao]['r2_folds'].append(r2)
            
            if len(y_teste_ocupacao) > 1 and np.std(y_teste_ocupacao) > 1e-12 and np.std(y_pred_ocupacao) > 1e-12:
                corr = np.corrcoef(y_teste_ocupacao, y_pred_ocupacao)[0, 1]
            else:
                corr = 0.0
                
            stats_occupations[ocupacao]['corr_folds'].append(corr)
            stats_occupations[ocupacao]['time_folds'].append(fold_time)

            stats_occupations[ocupacao]['mae_folds'].append(np.mean(np.abs(error_ocupacao)))
            mape = np.mean(np.abs(error_ocupacao / (y_teste_ocupacao + 1e-8))) * 100
            stats_occupations[ocupacao]['mape_folds'].append(mape)

            stats_occupations[ocupacao]['max_error_folds'].append(np.max(np.abs(error_ocupacao)))

            stats_occupations[ocupacao]['medae_folds'].append(np.median(np.abs(error_ocupacao)))

            print(f"Ocupacao: {ocupacao} -> Media: {np.mean(error_ocupacao):.4f}, STD: {np.std(error_ocupacao):.4f}, R^2: {stats_occupations[ocupacao]['r2_folds'][-1]:.4f}")

    
    print(f"  Fold {fold + 1} concluido")
    print("-" * 50)

for occupation in occupations:
    if len(stats_occupations[occupation]['mean_folds']) > 0:
        # Média das médias dos folds
        stats_occupations[occupation]['mean_error'] = np.mean(stats_occupations[occupation]['mean_folds'])
        # Std dos stds dos folds (mais robusto)
        stats_occupations[occupation]['std_error'] = np.mean(stats_occupations[occupation]['std_folds'])
        stats_occupations[occupation]['rms'] = np.mean(stats_occupations[occupation]['rms_folds'])
        stats_occupations[occupation]['r2'] = np.mean(stats_occupations[occupation]['r2_folds'])
        stats_occupations[occupation]['corr_mean'] = np.mean(stats_occupations[occupation]['corr_folds'])
        stats_occupations[occupation]['time_mean'] = np.mean(stats_occupations[occupation]['time_folds'])
        stats_occupations[occupation]['mae'] = np.mean(stats_occupations[occupation]['mae_folds'])
        stats_occupations[occupation]['mape'] = np.mean(stats_occupations[occupation]['mape_folds'])
        stats_occupations[occupation]['max_error'] = np.mean(stats_occupations[occupation]['max_error_folds'])
        stats_occupations[occupation]['medae'] = np.mean(stats_occupations[occupation]['medae_folds'])

print("\n=== RESUMO ===")
for ocupacao in occupations:
    n_amostras = len(results_occupations[ocupacao]['error'])
    print(f"Ocupacao {ocupacao}: {n_amostras} amostras")


# --------------------------------------- Save Data ---------------------------------------
os.makedirs(dataset_output, exist_ok=True)

# Salvar resultados por ocupação
for occupation in occupations:
    if len(results_occupations[occupation]['error']) > 0:
        np.savez(
            os.path.join(dataset_output, f"results_occupation_{occupation}.npz"),
            error=np.array(results_occupations[occupation]['error']),
            real_amplitude=np.array(results_occupations[occupation]['real_amplitude']),
            estimated_amplitude=np.array(results_occupations[occupation]['estimated_amplitude']),
            indices=np.array(results_occupations[occupation]['indices']),
            mean_error=stats_occupations[occupation]['mean_error'], 
            std_error=stats_occupations[occupation]['std_error'],
            rms =  stats_occupations[occupation]['rms'] ,
            r2 = stats_occupations[occupation]['r2'],
            corr_mean = stats_occupations[occupation]['corr_mean'],
            time_mean = stats_occupations[occupation]['time_mean'],
            mae=stats_occupations[occupation]['mae'],
            mape=stats_occupations[occupation]['mape'],
            max_error=stats_occupations[occupation]['max_error'],
            medae=stats_occupations[occupation]['medae']
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

print("Resultados salvos em:", dataset_output)