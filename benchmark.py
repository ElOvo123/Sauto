import subprocess
import json
import time
import os
import matplotlib.pyplot as plt

# A lista de partículas que queres testar
particles_list = [10, 50, 100, 200, 300, 500]
resultados = []

# Podes colocar aqui os "Melhores Alphas" que achares mais realistas
melhores_parametros = {
    "alphas": [0.5, 0.1, 0.8, 0.15],
    "r_noise_dist": 0.20,
    "r_noise_ang": 0.10
}

print("=== INÍCIO DO BENCHMARK DO FASTSLAM ===")
print(f"Testes agendados para: {particles_list} partículas")
print("--------------------------------------------------")

for n in particles_list:
    print(f"\n▶ A iniciar teste com {n} partículas...")
    
    # 1. Preparar o JSON para o ROS ler
    params = melhores_parametros.copy()
    params["num_particles"] = n
    with open("parametros_atuais.json", "w") as f:
        json.dump(params, f)
        
    # Limpar ficheiros da corrida anterior
    if os.path.exists("resultado_rmse.txt"): os.remove("resultado_rmse.txt")
    if os.path.exists("resultado_time.txt"): os.remove("resultado_time.txt")
    
    # 2. Iniciar FastSLAM
    slam_process = subprocess.Popen(["python3", "fastslam_ros.py"])
    time.sleep(6) # Tempo para o ROS arrancar
    
    # 3. Iniciar Rosbag a 10x
    bag_process = subprocess.Popen(["ros2", "bag", "play", "rosbag2_2026_05_13-12_08_16/", "-r", "10"])
    
    timeout = 150 # Timeout de 2.5 minutos
    start_wait = time.time()
    
    # 4. Esperar que o nó termine
    while slam_process.poll() is None:
        if os.path.exists("resultado_rmse.txt") and os.path.exists("resultado_time.txt"):
            break
        if time.time() - start_wait > timeout:
            print("  [!] Timeout atingido.")
            break
        time.sleep(1.0)
        
    # 5. Matar processos e limpar a RAM
    slam_process.kill()
    bag_process.kill()
    subprocess.run(["pkill", "-f", "ros2 bag play"], stderr=subprocess.DEVNULL)
    os.system("rm -rf ~/.ros/log/*")
    
    # 6. Ler resultados
    try:
        with open("resultado_rmse.txt", "r") as f:
            rmse = float(f.read())
        with open("resultado_time.txt", "r") as f:
            avg_time = float(f.read())
            
        print(f"  ✔ Concluído! Erro: {rmse:.3f}m | Tempo por Step: {avg_time:.2f} ms")
        resultados.append((n, avg_time, rmse))
    except Exception as e:
        print(f"  ✖ Erro a ler resultados: {e}")

# ==========================================
# GERAR O GRÁFICO FINAL
# ==========================================
if resultados:
    ns = [r[0] for r in resultados]
    tempos = [r[1] for r in resultados]
    erros = [r[2] for r in resultados]
    
    # Desenhar o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(ns, tempos, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # Estilização
    plt.title('Impacto do Número de Partículas no Tempo de Processamento', fontsize=14)
    plt.xlabel('Número de Partículas (N)', fontsize=12)
    plt.ylabel('Tempo Médio por Ciclo (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar os valores do Erro (RMSE) junto aos pontos do gráfico
    for i, txt in enumerate(erros):
        if txt != 999.0:
            plt.annotate(f"Erro: {txt:.2f}m", (ns[i], tempos[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Guardar a imagem
    plt.savefig('grafico_benchmark.png', bbox_inches='tight')
    print("\n==================================================")
    print("BENCHMARK CONCLUÍDO! O gráfico foi guardado como 'grafico_benchmark.png'")