import numpy as np
import os

n_janelamento = 7
EPS = 1e-8

root_path = os.path.abspath(__file__)
path = os.path.dirname(root_path)

of_data = os.path.join(path, "AmplitudeEstimada_OF", f'janelamento_{n_janelamento}')
a_tau_data = os.path.join(path,"A_tau_OF", f'janelamento_{n_janelamento}')

base_path = os.path.dirname(os.path.dirname(path))
CNN = 5
cnn_data = os.path.join(base_path, "OptimalFilterxConvolutionalNeuralNetworks","RedeNeuralConvolucional", f"CNN_{CNN}")

ocupacoes = [0,10,20,30,40,50,60,70,80,90,100]

results_summary = {
    'of': {},
    'cnn': {}
}

for ocupacao in ocupacoes:
    print(f"Ocupacao - {ocupacao}")
    of_amplitude_filepath = os.path.join(of_data, f'results_occupation_{ocupacao}.npz')
    cnn_amplitude_filepath = os.path.join(cnn_data, f'results_ocupacao_{ocupacao}.npz')
    a_tau_filepath = os.path.join(a_tau_data, f'results_occupation_{ocupacao}.npz')

    if not os.path.exists(a_tau_filepath):
        print(f"  ERRO: Arquivo A_tau não encontrado: {a_tau_filepath}")
        continue
    
    if not os.path.exists(of_amplitude_filepath):
        print(f"  ERRO: Arquivo OF amplitude não encontrado: {of_amplitude_filepath}")
        continue
    
    if not os.path.exists(cnn_amplitude_filepath):
        print(f"  AVISO: Arquivo CNN amplitude não encontrado: {cnn_amplitude_filepath}")
        print(f"  Continuando apenas com OF...")


    a_tau_data_file = np.load(a_tau_filepath)
    of_amp_data_file = np.load(of_amplitude_filepath)
    cnn_amp_data_file = np.load(cnn_amplitude_filepath)
    

    a_tau_estimated = a_tau_data_file['estimated_A_tau']
    a_tau_indices = a_tau_data_file['indices']
    real_phase = a_tau_data_file['real_phase']
    
    of_estimated_amplitude = of_amp_data_file['estimated_amplitude']
    real_amplitude = of_amp_data_file['real_amplitude']
    of_indices = of_amp_data_file['indices']

    cnn_estimated_amplitude = cnn_amp_data_file['estimated_amplitude']
    cnn_indices = cnn_amp_data_file['indices']

    
    a_tau_dict = {idx: val for idx, val in zip(a_tau_indices, a_tau_estimated)}
    real_phase_dic = {idx: val for idx, val in zip(a_tau_indices, real_phase)}
    of_amp_dict = {idx: val for idx, val in zip(of_indices, of_estimated_amplitude)}
    cnn_amp_dict = {idx: val for idx, val in zip(cnn_indices, cnn_estimated_amplitude)}
    real_amplitude_dict = {idx: val for idx, val in zip(of_indices, real_amplitude)}

    common_indices = sorted(set(a_tau_dict.keys()) & set(of_amp_dict.keys()) & set(cnn_amp_dict.keys()))
    print(f'Indices comuns: \n{len(common_indices)}')

    if len(common_indices)==0:
        print(f'ERRO: Nenhum indice igual!')
        continue
    
    a_tau_aligned = np.array([a_tau_dict[idx] for idx in common_indices])
    real_phase_aligned = np.array([real_phase_dic[idx] for idx in common_indices])
    of_amp_aligned = np.array([of_amp_dict[idx] for idx in common_indices])
    cnn_amp_aligned = np.array([cnn_amp_dict[idx] for idx in common_indices])
    real_amplitude_aligned = np.array([real_amplitude_dict[idx] for idx in common_indices])

    total_indices_iguais = len(common_indices)
    of_estimated_phase = np.zeros(total_indices_iguais)
    cnn_estimated_phase = np.zeros(total_indices_iguais)
    real_amplitude_estimated_phase = np.zeros(total_indices_iguais)

    for i in range(total_indices_iguais):
        if of_amp_aligned[i]==0:
            of_estimated_phase[i] = a_tau_aligned[i]/EPS
        else:
            of_estimated_phase[i] = a_tau_aligned[i]/of_amp_aligned[i]

        if cnn_amp_aligned[i]==0:
            cnn_estimated_phase[i] = a_tau_aligned[i]/EPS
        else:
            cnn_estimated_phase[i] = a_tau_aligned[i]/cnn_amp_aligned[i]

        if real_amplitude_aligned[i]==0:
            real_amplitude_estimated_phase[i] = a_tau_aligned[i]/EPS
        else:
            real_amplitude_estimated_phase[i] = a_tau_aligned[i]/real_amplitude_aligned[i]


    of_phase_error = of_estimated_phase - real_phase_aligned
    cnn_phase_error = cnn_estimated_phase - real_phase_aligned
    real_amplitude_phase_error = real_amplitude_estimated_phase - real_phase_aligned

    rms_of = np.sqrt(np.mean(of_phase_error**2))
    mae_of = np.mean(np.abs(of_phase_error))
    medae_of = np.median(np.abs(of_phase_error))

    ss_res_of = np.sum((real_phase_aligned - of_estimated_phase)**2)
    ss_tot_of = np.sum((real_phase_aligned - np.mean(real_phase_aligned))**2)
    r2_of = 1 - (ss_res_of / ss_tot_of) if ss_tot_of > 0 else 0
    
    corr_of = np.corrcoef(real_phase_aligned, of_estimated_phase)[0, 1] if len(real_phase_aligned) > 1 else 0


    rms_cnn = np.sqrt(np.mean(cnn_phase_error**2))
    mae_cnn = np.mean(np.abs(cnn_phase_error))
    medae_cnn = np.median(np.abs(cnn_phase_error))

    ss_res_cnn = np.sum((real_phase_aligned - cnn_estimated_phase)**2)
    ss_tot_cnn = np.sum((real_phase_aligned - np.mean(real_phase_aligned))**2)
    r2_cnn = 1 - (ss_res_cnn / ss_tot_cnn) if ss_tot_cnn > 0 else 0
    
    corr_cnn = np.corrcoef(real_phase_aligned, cnn_estimated_phase)[0, 1] if len(real_phase_aligned) > 1 else 0

    rms_real_amplitude = np.sqrt(np.mean(real_amplitude_phase_error**2))
    mae_real_amplitude = np.mean(np.abs(real_amplitude_phase_error))
    medae_real_amplitude = np.median(np.abs(real_amplitude_phase_error))

    ss_res_real_amplitude = np.sum((real_phase_aligned - real_amplitude_estimated_phase)**2)
    ss_tot_real_amplitude = np.sum((real_phase_aligned - np.mean(real_phase_aligned))**2)
    r2_real_amplitude = 1 - (ss_res_real_amplitude / ss_tot_real_amplitude) if ss_tot_real_amplitude > 0 else 0
    
    corr_real_amplitude = np.corrcoef(real_phase_aligned, real_amplitude_estimated_phase)[0, 1] if len(real_phase_aligned) > 1 else 0

    output_path = os.path.join(path, "FaseEstimada_OF", f'janelamento_{n_janelamento}')
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path,f"phase_of_occupation_{ocupacao}.npz")
    
    # np.savez_compressed(
    #     output_file,
    #     estimated_phase=of_estimated_phase,
    #     real_phase=real_phase_aligned,
    #     error=of_phase_error,
    #     estimated_A_tau=a_tau_aligned,
    #     estimated_amplitude=of_amp_aligned,
    #     indices=common_indices,
    #     rms=rms_of,
    #     mae=mae_of,
    #     medae=medae_of,
    #     r2=r2_of,
    #     correlation=corr_of,
    #     n_samples=len(common_indices)
    # )
    # CNN
    output_path_cnn = os.path.join(path, "FaseEstimada_CNN", f'janelamento_{n_janelamento}', f'CNN_{CNN}')
    os.makedirs(output_path_cnn, exist_ok=True)
    output_file_cnn = os.path.join(output_path_cnn,f"phase_cnn_occupation_{ocupacao}.npz")
    
    np.savez_compressed(
        output_file_cnn,
        estimated_phase=cnn_estimated_phase,
        real_phase=real_phase_aligned,
        error=cnn_phase_error,
        estimated_A_tau=a_tau_aligned,
        estimated_amplitude=cnn_amp_aligned,
        indices=common_indices,
        rms=rms_cnn,
        mae=mae_cnn,
        medae=medae_cnn,
        r2=r2_cnn,
        correlation=corr_cnn,
        n_samples=len(common_indices)
    )

    # output_path_real_amplitude = os.path.join(path, "FaseEstimada_RealAmplitude", f'janelamento_{n_janelamento}')
    # os.makedirs(output_path_real_amplitude, exist_ok=True)
    # output_file_real_amplitude = os.path.join(output_path_real_amplitude, f'phase_real_amplitude_occupation_{ocupacao}.npz')

    # np.savez_compressed(
    #     output_file_real_amplitude,
    #     estimated_phase=real_amplitude_estimated_phase,
    #     real_phase=real_phase_aligned,
    #     error=real_amplitude_phase_error,
    #     estimated_A_tau=a_tau_aligned,
    #     estimated_amplitude=real_amplitude_aligned,
    #     indices=common_indices,
    #     rms=rms_real_amplitude,
    #     mae=mae_real_amplitude,
    #     medae=medae_real_amplitude,
    #     r2=r2_real_amplitude,
    #     correlation=corr_real_amplitude,
    #     n_samples=len(common_indices)

    # )