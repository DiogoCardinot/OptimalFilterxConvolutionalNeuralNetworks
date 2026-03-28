import numpy as np
import os
from sklearn.model_selection import KFold
from tqdm import tqdm
import time

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

#--------------------------- PARAMS -----------------------------------------------------
N_JANELAMENTO = 7                   
K_FOLDS = 100                        
RANDOM_STATE = 42


# ---------------------------- Ref And Derivate Pulse loading ---------------------------
def load_pulse_and_derivatives_samples(caminho_base):
    target_times = np.array([-75, -50, -25, 0, 25, 50, 75])

    def read_datfile(filename):
        times = []
        values = []
        with open(filename, 'r') as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) >= 2:
                    try:
                        time = float(cols[0])
                        value = float(cols[1])
                    except ValueError:
                        continue
                    times.append(time)
                    values.append(value)
        return np.array(times), np.array(values)

    times_pulse, pulse = read_datfile(os.path.join(caminho_base, "../PulsoRef","pulsehi_physics.dat"))
    times_dpulse, dpulse = read_datfile(os.path.join(caminho_base, "../PulsoRef", "dpulsehi_physics.dat"))
    
    indices = [np.argmin(np.abs(times_pulse - t)) for t in target_times]
    indices_first =  [np.argmin(np.abs(times_dpulse - t)) for t in target_times]

    pulse_7 = pulse[indices]
    dpulse_7 = dpulse[indices_first]

    return target_times, pulse_7, dpulse_7

target_times, pulse_7, dpulse_7 = load_pulse_and_derivatives_samples(path)


#----------------------------- Data loading -------------------------------------------
base_path = os.path.dirname(os.path.dirname(path))
dataset_path = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks","ManipulacaoDados", "DadosPorOcupacao", f"janelamento_{N_JANELAMENTO}")

# ------------------------------ RUN OF to ALL Occupations --------------------------

for ocupacao in ocupacoes:
    print(f"Executando para ocupacao: {ocupacao}")
    data = os.path.join(dataset_path, f'dataset_ocup_{ocupacao}.npz')

    if not os.path.exists(data):
        print(f"Arquivo nao encontrado: {data}")
        continue
    
    data_values = np.load(data)
    MatrizAmostras = data_values['matriz_amostras']
    AmplitudeReal = data_values['amplitude_real']

    print(f"Total de amostras: {len(MatrizAmostras)}")

    def CovMatrix(data):
        cov_matrix = np.cov(data, rowvar=False)
        return cov_matrix


    def MatrixA(cov_matrix, pulse, dpulse):
        n = len(pulse)
        A = np.zeros((n + 3, n + 3))
        
        A[:n, :n] = cov_matrix
        
        A[:n, n] = pulse      # coluna g
        A[:n, n+1] = dpulse   # coluna g'
        A[:n, n+2] = 1        # coluna pedestal
        
        A[n, :n] = pulse      # linha g^T
        A[n+1, :n] = dpulse   # linha g'^T
        A[n+2, :n] = 1        # linha 1^T
        
        return A

    def vectorB(n_janelamento):
        B = np.zeros((n_janelamento + 3, 1))
        B[n_janelamento] = 1
        return B


    def Solution(matrizA, matrizB):
        matrizAAmpliada = np.hstack((matrizA, matrizB))
        postoA = np.linalg.matrix_rank(matrizA)
        postoAAmpliada = np.linalg.matrix_rank(matrizAAmpliada)
        n = matrizA.shape[1] 
        if postoA != postoAAmpliada:
            return "Sistema não possui solução!"
        else:
            if postoA == n:
                solucao = np.linalg.solve(matrizA, matrizB)  
                return solucao
            if postoA < n:
                return "Sistema com múltiplas soluções"
            
    # ----------------------------- K-fold configuration ---------------------------------------
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results_occupations = {}
    stats_occupations = {}

    results_occupations = {
        ocupacao: {
            'error': [],
            'real_amplitude': [],
            'estimated_amplitude': [],
            'indices': []
        }
    }

    stats_occupations = {
        ocupacao: {
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
    }

    for fold, (train_index, test_index) in enumerate(tqdm(kf.split(MatrizAmostras), desc=f"Ocupação {ocupacao}",total=K_FOLDS)):
        print(f"Processando Fold {fold + 1}/{K_FOLDS} para ocupacao {ocupacao}")

        start_time = time.time()
        Training_Matrix, Test_Matrix = MatrizAmostras[train_index], MatrizAmostras[test_index]
        Training_Amplitudes, Test_Amplitudes = AmplitudeReal[train_index], AmplitudeReal[test_index]

        print(f"  Treino: {len(Training_Matrix)} amostras")
        print(f"  Teste: {len(Test_Matrix)} amostras")

        cov_matrix = CovMatrix(Training_Matrix)
        A = MatrixA(cov_matrix, pulse_7, dpulse_7)
        B = vectorB(N_JANELAMENTO)

        res = Solution(A, B)
        w_kfold = np.zeros((N_JANELAMENTO))
        if type(res) != str:  
            for i in range(N_JANELAMENTO):
                w_kfold[i] = res[i][0]
        
        else:
            print(res)

        estimated_amplitudes = Test_Matrix @ w_kfold
        error = estimated_amplitudes - Test_Amplitudes
        
        end_time = time.time() 
        fold_time = end_time - start_time

        #RMS
        rms_error = np.sqrt(np.mean(error**2))

        #R^2
        ss_res = np.sum((Test_Amplitudes - estimated_amplitudes)**2)
        ss_tot = np.sum((Test_Amplitudes - np.mean(Test_Amplitudes))**2)
        r2_fold = 1 - (ss_res / ss_tot)

        #Correlação
        corr_fold = np.corrcoef(Test_Amplitudes, estimated_amplitudes)[0, 1]

        results_occupations[ocupacao]['error'].extend(error)
        results_occupations[ocupacao]['real_amplitude'].extend(Test_Amplitudes)
        results_occupations[ocupacao]['estimated_amplitude'].extend(estimated_amplitudes)
        results_occupations[ocupacao]['indices'].extend(test_index)

        stats_occupations[ocupacao]['mean_folds'].append(np.mean(error))
        stats_occupations[ocupacao]['std_folds'].append(np.std(error))
        stats_occupations[ocupacao]['rms_folds'].append(rms_error)
        stats_occupations[ocupacao]['r2_folds'].append(r2_fold)
        stats_occupations[ocupacao]['corr_folds'].append(corr_fold)
        stats_occupations[ocupacao]['time_folds'].append(fold_time)

        stats_occupations[ocupacao]['mae_folds'].append(np.mean(np.abs(error)))
        mape = np.mean(np.abs(error / (Test_Amplitudes + 1e-8))) * 100
        stats_occupations[ocupacao]['mape_folds'].append(mape)

        stats_occupations[ocupacao]['max_error_folds'].append(np.max(np.abs(error)))

        stats_occupations[ocupacao]['medae_folds'].append(np.median(np.abs(error)))

        print(f"  Fold {fold + 1} concluido - Media: {np.mean(error):.4f}, Std: {np.std(error):.4f}, R^2: {r2_fold:.4f}")
        print("-" * 50)


    assert len(results_occupations[ocupacao]['real_amplitude']) == len(results_occupations[ocupacao]['estimated_amplitude'])
    print(f"Verificação: {len(results_occupations[ocupacao]['real_amplitude'])} amostras alinhadas")

    if len(stats_occupations[ocupacao]['mean_folds']) > 0:
        stats_occupations[ocupacao]['mean_error'] = np.mean(stats_occupations[ocupacao]['mean_folds'])
        stats_occupations[ocupacao]['std_error'] = np.mean(stats_occupations[ocupacao]['std_folds'])
        stats_occupations[ocupacao]['rms'] = np.mean(stats_occupations[ocupacao]['rms_folds'])
        stats_occupations[ocupacao]['r2'] = np.mean(stats_occupations[ocupacao]['r2_folds'])
        stats_occupations[ocupacao]['corr_mean'] = np.mean(stats_occupations[ocupacao]['corr_folds'])
        stats_occupations[ocupacao]['time_mean'] = np.mean(stats_occupations[ocupacao]['time_folds'])
        stats_occupations[ocupacao]['mae'] = np.mean(stats_occupations[ocupacao]['mae_folds'])
        stats_occupations[ocupacao]['mape'] = np.mean(stats_occupations[ocupacao]['mape_folds'])
        stats_occupations[ocupacao]['max_error'] = np.mean(stats_occupations[ocupacao]['max_error_folds'])
        stats_occupations[ocupacao]['medae'] = np.mean(stats_occupations[ocupacao]['medae_folds'])

    # --------------------------------------- Save Data ---------------------------------------
    dataset_output = os.path.join(path, f"AmplitudeEstimada_OF", f'janelamento_{N_JANELAMENTO}')
    os.makedirs(dataset_output, exist_ok=True)

    if len(results_occupations[ocupacao]['error']) > 0:
        np.savez_compressed(
            os.path.join(dataset_output, f"results_occupation_{ocupacao}.npz"),
            error=np.array(results_occupations[ocupacao]['error']),
            real_amplitude=np.array(results_occupations[ocupacao]['real_amplitude']),
            estimated_amplitude=np.array(results_occupations[ocupacao]['estimated_amplitude']),
            mean_error=stats_occupations[ocupacao]['mean_error'], 
            std_error=stats_occupations[ocupacao]['std_error'],
            rms =  stats_occupations[ocupacao]['rms'] ,
            r2 = stats_occupations[ocupacao]['r2'],
            corr_mean = stats_occupations[ocupacao]['corr_mean'],
            time_mean = stats_occupations[ocupacao]['time_mean'],
            mae=stats_occupations[ocupacao]['mae'],
            mape=stats_occupations[ocupacao]['mape'],
            max_error=stats_occupations[ocupacao]['max_error'],
            medae=stats_occupations[ocupacao]['medae'],
            indices=np.array(results_occupations[ocupacao]['indices']) 
        )

    config_file = os.path.join(dataset_output, "Configs")
    os.makedirs(config_file, exist_ok=True)
    with open(os.path.join(config_file, f"of_config_{ocupacao}.txt"), "w") as f:
        f.write("Optimal Filter Configuration:\n")
        f.write(f"- Janelamento: {N_JANELAMENTO}\n")
        f.write(f"- Ocupacao: {ocupacao}\n")
        f.write(f"- K-Folds: {K_FOLDS}\n")
        f.write(f"- Pulse shape: {pulse_7}\n")
        f.write(f"- Derivative pulse: {dpulse_7}\n")

    print("\n=== RESUMO ===")
    n_amostras = len(results_occupations[ocupacao]['error'])
    if n_amostras > 0:
        print(f"Ocupacao {ocupacao}: {n_amostras} amostras")

    print(f"Resultados salvos em: {dataset_output}")

    print(f"Terminou a run para ocupacao: {ocupacao}")
    print("-" * 50)
    print("\n\n\n")