import math
import numpy as np
from particle_filter import ParticleFilter
from ekf import ExtendedKalmanFilter

# --- FUNÇÕES ESTÁTICAS (EKF) ---
def landmark_motion_model(state_estimate, control_input=None):
    return state_estimate

def landmark_motion_jacobian(state_estimate, control_input=None):
    return np.eye(2)

def make_measurement_model(robot_state):
    rx, ry, rtheta = robot_state
    def model(lm_state):
        lx, ly = lm_state
        dx = lx - rx
        dy = ly - ry
        r = math.hypot(dx, dy)
        b = math.atan2(dy, dx) - rtheta
        b = (b + math.pi) % (2 * math.pi) - math.pi
        return np.array([r, b])
    return model

def make_measurement_jacobian(robot_state):
    rx, ry, rtheta = robot_state
    def jacobian(lm_state):
        lx, ly = lm_state
        dx = lx - rx
        dy = ly - ry
        q = dx**2 + dy**2
        if q < 1e-6: q = 1e-6
        sq = math.sqrt(q)
        return np.array([
            [dx / sq, dy / sq],
            [-dy / q, dx / q]
        ])
    return jacobian


class FastSLAM(ParticleFilter):
    def __init__(self, initial_pose, num_particles=100):
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
        
        # Alphas mais relaxados (O FastSLAM 2.0 lida muito melhor com o ruído)
        self.alphas = [0.03, 0.01, 0.05, 0.01]
        self.R_noise = np.array([[0.0025, 0.0], [0.0, 0.0004]]) 

    def step(self, current_odom, measurements, dt):
        dx = current_odom[0] - self.prev_odom[0]
        dy = current_odom[1] - self.prev_odom[1]
        trans = math.hypot(dx, dy)
        
        dir_angle = math.atan2(dy, dx)
        diff_angle = (dir_angle - self.prev_odom[2] + math.pi) % (2 * math.pi) - math.pi
        
        if abs(diff_angle) > math.pi / 2.0:
            trans = -trans
            dir_angle = (dir_angle + math.pi) % (2 * math.pi) - math.pi
            
        if abs(trans) > 0.0001:
            rot1 = dir_angle - self.prev_odom[2]
        else:
            rot1 = 0.0
            
        rot1 = (rot1 + math.pi) % (2 * math.pi) - math.pi 
        rot2 = current_odom[2] - self.prev_odom[2] - rot1
        rot2 = (rot2 + math.pi) % (2 * math.pi) - math.pi
        
        self.prev_odom = np.array(current_odom, dtype=float)

        if trans > 0.0001 or abs(rot1) > 0.0001 or abs(rot2) > 0.0001 or len(measurements) > 0:
            # O coração do FastSLAM 2.0: Ciclo fundido
            self._fastslam2_cycle(rot1, trans, rot2, measurements)
            
            if self.effective_sample_size() < self.resample_threshold:
                self.systematic_resample()

        best_p = max(self.particles, key=lambda p: p.weight)
        est_pose = best_p.state.tolist()
        
        est_map = {m_id: [ekf.state_estimate[0], ekf.state_estimate[1]] for m_id, ekf in best_p.landmarks.items()}
        particles_poses = [[p.state[0], p.state[1], p.state[2]] for p in self.particles]

        return particles_poses, est_pose, est_map

    def _fastslam2_cycle(self, rot1, trans, rot2, measurements):
        for p in self.particles:
            # 1. Calcular a Prior da Odometria (Onde o robô estaria sem câmara)
            x_prior = np.copy(p.state)
            x_prior[0] += trans * math.cos(p.state[2] + rot1)
            x_prior[1] += trans * math.sin(p.state[2] + rot1)
            x_prior[2] += rot1 + rot2
            x_prior[2] = (x_prior[2] + math.pi) % (2 * math.pi) - math.pi

            # Calcular Covariância do Movimento (Rt)
            v_trans = (self.alphas[2] * abs(trans) + self.alphas[3] * (abs(rot1) + abs(rot2)))**2 + 1e-6
            v_rot = (self.alphas[0] * (abs(rot1) + abs(rot2)) + self.alphas[1] * abs(trans))**2 + 1e-6
            Rt = np.diag([v_trans, v_trans, v_rot])
            inv_Rt = np.linalg.inv(Rt)

            # Procurar um marco conhecido para gerar a Distribuição de Proposta
            known_meas = [m for m in measurements if m[0] in p.landmarks]

            if len(known_meas) > 0:
                # O MAGIA DO FASTSLAM 2.0 COMEÇA AQUI
                meas = known_meas[0] # Usar o marco mais forte para ancorar a partícula
                m_id, z_r, z_b = meas[0], meas[1], meas[2]
                z = np.array([z_r, z_b])
                ekf = p.landmarks[m_id]

                meas_model = make_measurement_model(x_prior)
                meas_jacob = make_measurement_jacobian(x_prior)

                z_pred = meas_model(ekf.state_estimate)
                Hm = meas_jacob(ekf.state_estimate)

                # Construção limpa do Jacobiano do Robô (Hx) usando o Jacobiano do Marco (Hm)
                Hx = np.array([
                    [-Hm[0,0], -Hm[0,1], 0.0],
                    [-Hm[1,0], -Hm[1,1], -1.0]
                ])

                # Equações da Distribuição de Proposta
                Qj = Hm @ ekf.covariance_estimate @ Hm.T + self.R_noise
                inv_Qj = np.linalg.inv(Qj)

                inv_Sigma_x = Hx.T @ inv_Qj @ Hx + inv_Rt
                Sigma_x = np.linalg.inv(inv_Sigma_x)

                v = z - z_pred
                v[1] = (v[1] + math.pi) % (2 * math.pi) - math.pi

                mu_x = x_prior + Sigma_x @ Hx.T @ inv_Qj @ v

                # Saltar para a nova pose proposta pela câmara + odometria combinadas!
                p.state = self.rng.multivariate_normal(mu_x, Sigma_x)
                p.state[2] = (p.state[2] + math.pi) % (2 * math.pi) - math.pi

                # Cálculo exato do peso FastSLAM 2.0
                Q = Hx @ Rt @ Hx.T + Qj
                inv_Q = np.linalg.inv(Q)
                det_Q = np.linalg.det(Q)
                likelihood = np.exp(-0.5 * v.T @ inv_Q @ v) / np.sqrt((2 * np.pi)**2 * det_Q)
                p.weight *= (float(likelihood) + 1e-10)

            else:
                # Cego (Nenhum marco conhecido). Faz um salto normal só com odometria
                p.state = self.rng.multivariate_normal(x_prior, Rt)
                p.state[2] = (p.state[2] + math.pi) % (2 * math.pi) - math.pi

            # 2. Atualizar TODOS os marcos (EKFs) usando a pose recém-descoberta
            final_model = make_measurement_model(p.state)
            final_jacob = make_measurement_jacobian(p.state)

            for meas in measurements:
                m_id = meas[0]
                z = np.array([meas[1], meas[2]])

                if m_id not in p.landmarks:
                    lx = p.state[0] + z[0] * math.cos(p.state[2] + z[1])
                    ly = p.state[1] + z[0] * math.sin(p.state[2] + z[1])
                    p.landmarks[m_id] = ExtendedKalmanFilter(
                        initial_state=np.array([lx, ly]),
                        initial_covariance=np.eye(2) * 1.0,
                        process_noise_covariance=np.zeros((2,2)),
                        measurement_noise_covariance=self.R_noise,
                        motion_model=landmark_motion_model,
                        measurement_model=final_model,
                        motion_jacobian=landmark_motion_jacobian,
                        measurement_jacobian=final_jacob
                    )
                else:
                    ekf = p.landmarks[m_id]
                    ekf.measurement_model = final_model
                    ekf.measurement_jacobian = final_jacob
                    
                    # Truque do EKF para evitar saltos nos ângulos de -180/180
                    z_p = final_model(ekf.state_estimate)
                    inov = z - z_p
                    inov[1] = (inov[1] + math.pi) % (2 * math.pi) - math.pi
                    z_adj = np.array([z[0], z_p[1] + inov[1]])
                    
                    ekf.update(measurement=z_adj)

        # Normalizar Pesos
        weights = self.get_weights_array()
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            for p in self.particles:
                p.weight /= weight_sum
        else:
            for p in self.particles:
                p.weight = 1.0 / self.num_particles