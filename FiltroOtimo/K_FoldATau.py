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
    FaseReal = data_values['phase_real']

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
        B[n_janelamento+1] = 1
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
            'real_phase': [],
            'estimated_A_tau': [],
            'indices': []
        }
    }

    for fold, (train_index, test_index) in enumerate(tqdm(kf.split(MatrizAmostras), desc=f"Ocupação {ocupacao}",total=K_FOLDS)):
        print(f"Processando Fold {fold + 1}/{K_FOLDS} para ocupacao {ocupacao}")

        Training_Matrix, Test_Matrix = MatrizAmostras[train_index], MatrizAmostras[test_index]
        Training_Fase, Test_Fase = FaseReal[train_index], FaseReal[test_index]

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

        estimated_Atau = Test_Matrix @ w_kfold
    
        results_occupations[ocupacao]['real_phase'].extend(Test_Fase)
        results_occupations[ocupacao]['estimated_A_tau'].extend(estimated_Atau)
        results_occupations[ocupacao]['indices'].extend(test_index)

        if (fold + 1) % 20 == 0:  # Print a cada 20 folds
            print(f"  Fold {fold + 1}/{K_FOLDS} concluído")

    n_samples = len(results_occupations[ocupacao]['real_phase'])
    assert len(results_occupations[ocupacao]['estimated_A_tau']) == n_samples
    print(f"\n  Verificação: {n_samples} amostras alinhadas")

    # --------------------------------------- Save Data ---------------------------------------
    dataset_output = os.path.join(path, f"A_tau_OF", f'janelamento_{N_JANELAMENTO}')
    os.makedirs(dataset_output, exist_ok=True)

    if n_samples > 0:
        np.savez_compressed(
            os.path.join(dataset_output, f"results_occupation_{ocupacao}.npz"),
            real_phase=np.array(results_occupations[ocupacao]['real_phase']),
            estimated_A_tau=np.array(results_occupations[ocupacao]['estimated_A_tau']),
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
   
    print(f"Resultados salvos em: {dataset_output}")

    print(f"Terminou a run para ocupacao: {ocupacao}")
    print("-" * 50)
    print("\n\n\n")