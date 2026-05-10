import math
import numpy as np
from particle_filter import ParticleFilter
from ekf import ExtendedKalmanFilter

# --- FUNÇÕES ESTÁTICAS (EKF) ---
def landmark_motion_model(state_estimate, control_input=None):
    return state_estimate

def landmark_motion_jacobian(state_estimate, control_input=None):
    return np.eye(2)

def make_measurement_model(robot_state, camera_offset=0.05):
    rx, ry, rtheta = robot_state

    cam_x = rx + camera_offset * math.cos(rtheta)
    cam_y = ry + camera_offset * math.sin(rtheta)

    def model(lm_state):
        lx, ly = lm_state
        dx = lx - cam_x
        dy = ly - cam_y

        r = math.hypot(dx, dy)
        b = math.atan2(dy, dx) - rtheta
        b = (b + math.pi) % (2 * math.pi) - math.pi

        return np.array([r, b])

    return model

def make_measurement_jacobian(robot_state, camera_offset=0.05):
    rx, ry, rtheta = robot_state

    cam_x = rx + camera_offset * math.cos(rtheta)
    cam_y = ry + camera_offset * math.sin(rtheta)

    def jacobian(lm_state):
        lx, ly = lm_state
        dx = lx - cam_x
        dy = ly - cam_y

        q = dx**2 + dy**2
        if q < 1e-6:
            q = 1e-6

        sq = math.sqrt(q)

        return np.array([
            [dx / sq, dy / sq],
            [-dy / q, dx / q]
        ])

    return jacobian

class FastSLAM1(ParticleFilter):
    def __init__(self, initial_pose, num_particles=300):
        def dummy_init(n, d, rng):
            return np.tile(initial_pose, (n, 1))
            
        super().__init__(
            num_particles=num_particles,
            state_dim=3,
            init_fn=dummy_init,
            motion_fn=None,
            measurement_likelihood_fn=None,
            resample_threshold_ratio=0.5
        )
        
        self.prev_odom = np.array(initial_pose, dtype=float)
        
        # Parâmetros
        self.alphas = [0.03, 0.01, 0.05, 0.01]
        self.R_noise = np.array([[0.0025, 0.0], [0.0, 0.0004]]) 

        # NOVOS PARÂMETROS PARA RESOLVER O ERRO TEMPORAL
        # O SLAM só corre se o robô andar 5 cm ou rodar ~3 graus (0.05 radianos)
        self.min_trans_update = 0.00  
        self.min_rot_update = 0.00   
        self.is_initialized = False

    def step(self, current_odom, measurements, dt):
        # Calcular a diferença desde a ÚLTIMA VEZ que o SLAM executou um ciclo
        dx = current_odom[0] - self.prev_odom[0]
        dy = current_odom[1] - self.prev_odom[1]
        trans = math.hypot(dx, dy)
        
        rot_total = (current_odom[2] - self.prev_odom[2] + math.pi) % (2 * math.pi) - math.pi
        
        # --- A BARREIRA ESPACIAL ---
        # Se não andou o suficiente nem rodou o suficiente, devolve as poses antigas e NÃO faz nada!
        if trans < self.min_trans_update and abs(rot_total) < self.min_rot_update:
            
            # Exceção: Se for o primeiríssimo frame, queremos mapear o que está à volta antes de arrancar
            if not self.is_initialized and len(measurements) > 0:
                self._update_maps(measurements)
                self.is_initialized = True
                
            best_p = max(self.particles, key=lambda p: p.weight)
            est_pose = best_p.state.tolist()
            est_map = {m_id: [ekf.state_estimate[0], ekf.state_estimate[1]] for m_id, ekf in best_p.landmarks.items()}
            particles_poses = [[p.state[0], p.state[1], p.state[2], p.weight] for p in self.particles]
            return particles_poses, est_pose, est_map

        # --- SE CHEGOU AQUI, O ROBÔ MOVEU-SE O SUFICIENTE ---
        dir_angle = math.atan2(dy, dx)
        diff_angle = (dir_angle - self.prev_odom[2] + math.pi) % (2 * math.pi) - math.pi
        
        if abs(diff_angle) > math.pi / 2.0:
            trans = -trans
            dir_angle = (dir_angle + math.pi) % (2 * math.pi) - math.pi
            
        rot1 = dir_angle - self.prev_odom[2]
        rot1 = (rot1 + math.pi) % (2 * math.pi) - math.pi 
        rot2 = current_odom[2] - self.prev_odom[2] - rot1
        rot2 = (rot2 + math.pi) % (2 * math.pi) - math.pi
        
        # Atualiza o marco de referência da odometria APENAS quando o ciclo vai correr
        self.prev_odom = np.array(current_odom, dtype=float)

        # 1. FASE PREDICT: Salto Cego
        self._predict(rot1, trans, rot2)
            
        # 2. FASE UPDATE: Avaliar
        if len(measurements) > 0:
            self._update_maps(measurements)
            self.is_initialized = True
            
        if self.effective_sample_size() < self.resample_threshold:
            self.systematic_resample()

        # Extrair a melhor estimativa
        best_p = max(self.particles, key=lambda p: p.weight)
        est_pose = best_p.state.tolist()
        
        est_map = {m_id: [ekf.state_estimate[0], ekf.state_estimate[1]] for m_id, ekf in best_p.landmarks.items()}
        particles_poses = [[p.state[0], p.state[1], p.state[2], p.weight] for p in self.particles]

        return particles_poses, est_pose, est_map

    def _predict(self, rot1, trans, rot2):
        # Variâncias do movimento (exatamente como no FS2, mas aplicadas cegamente)
        v_trans = (self.alphas[2] * abs(trans) + self.alphas[3] * (abs(rot1) + abs(rot2)))**2 + 1e-6
        v_rot1 = (self.alphas[0] * abs(rot1) + self.alphas[1] * abs(trans))**2 + 1e-6
        v_rot2 = (self.alphas[0] * abs(rot2) + self.alphas[1] * abs(trans))**2 + 1e-6

        sd_trans = math.sqrt(v_trans)
        sd_rot1 = math.sqrt(v_rot1)
        sd_rot2 = math.sqrt(v_rot2)

        for p in self.particles:
            # Injectar ruído Gaussiano puro
            noisy_trans = trans + self.rng.normal(0, sd_trans)
            noisy_rot1 = rot1 + self.rng.normal(0, sd_rot1)
            noisy_rot2 = rot2 + self.rng.normal(0, sd_rot2)

            # Mover a partícula
            p.state[0] += noisy_trans * math.cos(p.state[2] + noisy_rot1)
            p.state[1] += noisy_trans * math.sin(p.state[2] + noisy_rot1)
            p.state[2] += noisy_rot1 + noisy_rot2
            p.state[2] = (p.state[2] + math.pi) % (2 * math.pi) - math.pi

    def _update_maps(self, measurements):
        for p in self.particles:
            current_meas_model = make_measurement_model(p.state)
            current_meas_jacob = make_measurement_jacobian(p.state)

            for meas in measurements:
                m_id = meas[0]
                z = np.array([meas[1], meas[2]]) # [range, bearing]

                if m_id not in p.landmarks:
                    # INICIALIZAR NOVO MARCO (com o camera_offset exato)
                    cam_x = p.state[0] + 0.05 * math.cos(p.state[2])
                    cam_y = p.state[1] + 0.05 * math.sin(p.state[2])

                    lx = cam_x + z[0] * math.cos(p.state[2] + z[1])
                    ly = cam_y + z[0] * math.sin(p.state[2] + z[1])

                    p.landmarks[m_id] = ExtendedKalmanFilter(
                        initial_state=np.array([lx, ly]),
                        initial_covariance=np.eye(2) * 1.0,
                        process_noise_covariance=np.zeros((2, 2)),
                        measurement_noise_covariance=self.R_noise,
                        motion_model=landmark_motion_model,
                        measurement_model=current_meas_model,
                        motion_jacobian=landmark_motion_jacobian,
                        measurement_jacobian=current_meas_jacob
                    )
                    # No FS1, a primeira observação não altera o peso
                else:
                    ekf = p.landmarks[m_id]

                    # Atualizar modelos com a pose que adivinhámos no predict
                    ekf.measurement_model = current_meas_model
                    ekf.measurement_jacobian = current_meas_jacob

                    # PREVISÃO vs REALIDADE (Calcular Likelihood e cortar o Peso)
                    z_pred = current_meas_model(ekf.state_estimate)
                    H = current_meas_jacob(ekf.state_estimate)

                    v = z - z_pred
                    v[1] = (v[1] + math.pi) % (2 * math.pi) - math.pi

                    Q = H @ ekf.covariance_estimate @ H.T + self.R_noise

                    try:
                        det_Q = np.linalg.det(Q)
                        inv_Q = np.linalg.inv(Q)
                        likelihood = np.exp(-0.5 * v.T @ inv_Q @ v) / np.sqrt((2 * np.pi)**2 * det_Q)
                        p.weight *= float(likelihood)
                    except np.linalg.LinAlgError:
                        p.weight *= 1e-300

                    # Truque do EKF para evitar saltos nos ângulos
                    inov = z - z_pred
                    inov[1] = (inov[1] + math.pi) % (2 * math.pi) - math.pi
                    z_adj = np.array([z[0], z_pred[1] + inov[1]])

                    # FINALMENTE: Atualizar o mapa do marco
                    ekf.update(measurement=z_adj)

        # Normalizar pesos no fim da ronda
        weights = self.get_weights_array()
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            for p in self.particles:
                p.weight /= weight_sum
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles