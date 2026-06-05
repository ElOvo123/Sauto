import subprocess
import json
import time
import os
import optuna
from optuna.samplers import CmaEsSampler

print("=== INÍCIO DA OTIMIZAÇÃO FASTSLAM (1 HORA / 10x SPEED) ===")
print("Base de Dados SQLite Ativada | Auto-limpeza do ROS Ativada")
print("Tentativas: 45 (~60 minutos de execução)")
print("-------------------------------------------------------------------------")

def objective(trial):
    
    # Limites agressivos para abranger qualquer derrapagem
    a0 = trial.suggest_float("alpha_1_rot_rot", 0.10, 1.50)
    a1 = trial.suggest_float("alpha_2_rot_trans", 0.01, 0.40)
    a2 = trial.suggest_float("alpha_3_trans_trans", 0.10, 2.00)
    a3 = trial.suggest_float("alpha_4_trans_rot", 0.01, 0.40)
    
    r_dist = trial.suggest_float("r_noise_dist", 0.05, 1.00)
    r_ang = trial.suggest_float("r_noise_ang", 0.01, 0.50) 
    
    print(f"\n[Tent. {trial.number}/45] Alphas: [{a0:.2f}, {a1:.2f}, {a2:.2f}, {a3:.2f}] | R_Dist: {r_dist:.2f} | R_Ang: {r_ang:.2f}")
    
    params = {
        "alphas": [a0, a1, a2, a3],
        "r_noise_dist": r_dist,
        "r_noise_ang": r_ang
    }
    with open("parametros_atuais.json", "w") as f:
        json.dump(params, f)

    if os.path.exists("resultado_rmse.txt"):
        os.remove("resultado_rmse.txt")

    # Iniciar FastSLAM
    slam_process = subprocess.Popen(["python3", "fastslam_ros.py"])
    time.sleep(8) # Tempo crucial para o nó não perder o arranque do bag
    
    # Iniciar Rosbag a 10x
    bag_process = subprocess.Popen(["ros2", "bag", "play", "rosbag2_2026_05_13-12_08_16/", "-r", "10"])

    # Timeout de 120s (2 minutos). A 10x, o bag de 10 min demora 1 minuto.
    timeout = 120 
    start_wait = time.time()
    
    while slam_process.poll() is None:
        if os.path.exists("resultado_rmse.txt"):
            break
        if time.time() - start_wait > timeout:
            print("  -> Timeout! O robô perdeu-se ou o nó encravou.")
            break
        time.sleep(1.0)

    # Matar processos
    slam_process.kill()
    bag_process.kill()
    subprocess.run(["pkill", "-f", "ros2 bag play"], stderr=subprocess.DEVNULL) 
    
    # LIMPEZA DO DISCO (Previne o OSError: No space left on device)
    os.system("rm -rf ~/.ros/log/*")

    # Ler ATE
    try:
        with open("resultado_rmse.txt", "r") as f:
            rmse = float(f.read())
            
        print(f"  -> ATE Obtido (SVD): {rmse:.4f}")
        
        if rmse > 500.0 or rmse == 0.0:
            return 999.0 
            
        return rmse
        
    except Exception:
        print("  -> Erro crítico na execução.")
        return 999.0 

# =========================================================
# ESTUDO CMA-ES COM GRAVAÇÃO NO DISCO
# =========================================================

db_path = "sqlite:///fastslam_estudo_1hora.db"

study = optuna.create_study(
    study_name="calibracao_1_hora",
    storage=db_path,          
    load_if_exists=True,      
    direction="minimize", 
    sampler=CmaEsSampler()
)

try:
    study.optimize(objective, n_trials=45)
except KeyboardInterrupt:
    print("\n[!] Otimização interrompida manualmente antes do fim.")

print("\n=================================================")
print("OTIMIZAÇÃO DE 1 HORA CONCLUÍDA")
print(f"O Menor ATE Absoluto alcançado foi: {study.best_value:.4f}")
best = study.best_params
print(f"Alphas = [{best['alpha_1_rot_rot']:.3f}, {best['alpha_2_rot_trans']:.3f}, {best['alpha_3_trans_trans']:.3f}, {best['alpha_4_trans_rot']:.3f}]")
print(f"R_Noise_Dist = {best['r_noise_dist']:.4f}")
print(f"R_Noise_Ang = {best['r_noise_ang']:.4f}")
print("=================================================")