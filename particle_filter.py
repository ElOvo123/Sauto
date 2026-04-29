import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Particle:
    def __init__(self, state, weight):
        self.state = np.array(state, dtype=float)
        self.weight = float(weight)

    def copy(self):
        return Particle(self.state.copy(), self.weight)


class ParticleFilter:
    def __init__(
        self,
        num_particles,
        state_dim,
        init_fn,
        motion_fn,
        measurement_likelihood_fn,
        resample_threshold_ratio=0.5,
        seed=None,
        debug=False
    ):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.init_fn = init_fn
        self.motion_fn = motion_fn
        self.measurement_likelihood_fn = measurement_likelihood_fn
        self.resample_threshold = resample_threshold_ratio * num_particles
        self.debug = debug

        self.rng = np.random.default_rng(seed)

        initial_states = self.init_fn(self.num_particles, self.state_dim, self.rng)

        self.particles = [
            Particle(state=initial_states[i], weight=1.0 / self.num_particles)
            for i in range(self.num_particles)
        ]

        self.history_estimates = []
        self.history_neff = []
        self.history_resampled = []
        self.history_particles = []

    def get_states_array(self):
        return np.array([p.state for p in self.particles])

    def get_weights_array(self):
        return np.array([p.weight for p in self.particles])

    def set_states_and_weights(self, states, weights):
        for i, particle in enumerate(self.particles):
            particle.state = states[i].copy()
            particle.weight = float(weights[i])

    def predict(self, k, u=None):
        states = self.get_states_array()
        predicted_states = self.motion_fn(states, k, u, self.rng)

        for i, particle in enumerate(self.particles):
            particle.state = predicted_states[i].copy()

    def update(self, z):
        states = self.get_states_array()
        likelihoods = self.measurement_likelihood_fn(z, states)
        likelihoods = np.maximum(likelihoods, 1e-300)

        weights = self.get_weights_array()
        weights *= likelihoods

        weight_sum = np.sum(weights)

        if weight_sum <= 0 or not np.isfinite(weight_sum):
            print("[WARNING] Weight collapse. Resetting weights.")
            weights[:] = 1.0 / self.num_particles
        else:
            weights /= weight_sum

        for i, particle in enumerate(self.particles):
            particle.weight = float(weights[i])

    def estimate(self):
        states = self.get_states_array()
        weights = self.get_weights_array()
        return np.average(states, axis=0, weights=weights)

    def effective_sample_size(self):
        weights = self.get_weights_array()
        return 1.0 / np.sum(weights ** 2)

    def systematic_resample(self):
        weights = self.get_weights_array()
        states = self.get_states_array()

        positions = (self.rng.random() + np.arange(self.num_particles)) / self.num_particles

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0

        indexes = np.searchsorted(cumulative_sum, positions)

        new_states = states[indexes]
        new_weights = np.ones(self.num_particles) / self.num_particles

        self.set_states_and_weights(new_states, new_weights)

    def step(self, z, k, u=None):
        self.predict(k, u)
        self.update(z)

        estimate = self.estimate()
        neff = self.effective_sample_size()

        self.history_particles.append(self.get_states_array().copy())

        resampled = False
        if neff < self.resample_threshold:
            self.systematic_resample()
            resampled = True

        self.history_estimates.append(estimate.copy())
        self.history_neff.append(neff)
        self.history_resampled.append(resampled)

        if self.debug:
            print(
                f"k={k:03d} | z={z:.3f} | est={estimate[0]:.3f} "
                f"| N_eff={neff:.1f} | resampled={resampled}"
            )

        return estimate


def nonlinear_state_transition(x_prev, k):
    return (
        0.5 * x_prev
        + 25.0 * x_prev / (1.0 + x_prev ** 2)
        + 8.0 * np.cos(1.2 * k)
    )


def generate_data(T=100, process_var=10.0, measurement_var=1.0, seed=1):
    rng = np.random.default_rng(seed)

    x_true = np.zeros(T)
    z_meas = np.zeros(T)

    x_true[0] = rng.normal(0, np.sqrt(process_var))
    z_meas[0] = x_true[0] ** 2 / 20.0 + rng.normal(0, np.sqrt(measurement_var))

    for k in range(1, T):
        x_true[k] = nonlinear_state_transition(x_true[k - 1], k) + rng.normal(0, np.sqrt(process_var))
        z_meas[k] = x_true[k] ** 2 / 20.0 + rng.normal(0, np.sqrt(measurement_var))

    return x_true, z_meas


def init_particles(num_particles, state_dim, rng):
    return rng.uniform(-25, 25, size=(num_particles, state_dim))


def motion_model(particles, k, u, rng):
    process_var = 10.0
    noise = rng.normal(0, np.sqrt(process_var), size=particles.shape)

    x = particles[:, 0]
    predicted = nonlinear_state_transition(x, k) + noise[:, 0]

    return predicted.reshape(-1, 1)


def measurement_likelihood(z, particles):
    measurement_var = 1.0
    x = particles[:, 0]

    expected_z = x ** 2 / 20.0
    error = z - expected_z

    likelihood = np.exp(-0.5 * error ** 2 / measurement_var)
    likelihood /= np.sqrt(2.0 * np.pi * measurement_var)

    return likelihood


def run_test(debug=True):
    T = 100
    num_particles = 500

    x_true, z_meas = generate_data(T=T, seed=10)

    pf = ParticleFilter(
        num_particles=num_particles,
        state_dim=1,
        init_fn=init_particles,
        motion_fn=motion_model,
        measurement_likelihood_fn=measurement_likelihood,
        resample_threshold_ratio=0.5,
        seed=42,
        debug=debug
    )

    estimates = np.zeros(T)

    for k in range(T):
        estimate = pf.step(z_meas[k], k)
        estimates[k] = estimate[0]

    rmse = np.sqrt(np.mean((x_true - estimates) ** 2))
    print(f"\nRMSE = {rmse:.4f}")

    particle_history = np.array(pf.history_particles)[:, :, 0]

    fig, axs = plt.subplots(4, 1, figsize=(11, 14), sharex=True)

    axs[0].plot(x_true, label="True state")
    axs[0].plot(estimates, label="PF estimate")
    axs[0].set_ylabel("State x")
    axs[0].set_title("Particle Filter Tracking")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(z_meas, label="Measurements")
    axs[1].set_ylabel("Measurement z")
    axs[1].set_title("Noisy Measurements")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(pf.history_neff)
    axs[2].axhline(pf.resample_threshold, linestyle="--", label="Resample threshold")
    axs[2].set_ylabel("N_eff")
    axs[2].set_title("Degeneracy Debug")
    axs[2].legend()
    axs[2].grid(True)

    for k in range(T):
        axs[3].scatter(
            np.full(num_particles, k),
            particle_history[k],
            s=2,
            alpha=0.15
        )

    axs[3].plot(x_true, linewidth=2, label="True state")
    axs[3].plot(estimates, linewidth=2, label="PF estimate")
    axs[3].set_ylabel("Particles")
    axs[3].set_xlabel("Time step k")
    axs[3].set_title("Particle Cloud Over Time")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

    return pf, x_true, z_meas, estimates

def animate_particle_filter(pf, x_true, z_meas, estimates):
    particle_history = np.array(pf.history_particles)[:, :, 0]

    T = len(x_true)
    num_particles = particle_history.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, T)
    ax.set_ylim(-30, 30)
    ax.set_xlabel("Time step k")
    ax.set_ylabel("State x")
    ax.set_title("Animated Particle Filter")

    ax.grid(True)

    true_line, = ax.plot([], [], label="True state", linewidth=2)
    estimate_line, = ax.plot([], [], label="PF estimate", linewidth=2)
    particle_scatter = ax.scatter([], [], s=8, alpha=0.25, label="Particles")

    current_true, = ax.plot([], [], "o", markersize=8, label="Current true state")
    current_est, = ax.plot([], [], "o", markersize=8, label="Current estimate")

    text_box = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax.legend(loc="upper right")

    def init():
        true_line.set_data([], [])
        estimate_line.set_data([], [])
        particle_scatter.set_offsets(np.empty((0, 2)))
        current_true.set_data([], [])
        current_est.set_data([], [])
        text_box.set_text("")
        return true_line, estimate_line, particle_scatter, current_true, current_est, text_box

    def update(frame):
        k_values = np.arange(frame + 1)

        true_line.set_data(k_values, x_true[:frame + 1])
        estimate_line.set_data(k_values, estimates[:frame + 1])

        x_particles = np.full(num_particles, frame)
        y_particles = particle_history[frame]

        particle_scatter.set_offsets(np.column_stack((x_particles, y_particles)))

        current_true.set_data([frame], [x_true[frame]])
        current_est.set_data([frame], [estimates[frame]])

        resampled_text = "YES" if pf.history_resampled[frame] else "NO"

        text_box.set_text(
            f"k = {frame}\n"
            f"z = {z_meas[frame]:.2f}\n"
            f"True x = {x_true[frame]:.2f}\n"
            f"PF estimate = {estimates[frame]:.2f}\n"
            f"N_eff = {pf.history_neff[frame]:.1f}\n"
            f"Resampled = {resampled_text}"
        )

        return true_line, estimate_line, particle_scatter, current_true, current_est, text_box

    ani = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=120,
        blit=True,
        repeat=True
    )

    plt.show()
    return ani


if __name__ == "__main__":
    pf, x_true, z_meas, estimates = run_test(debug=False)
    # animate_particle_filter(pf, x_true, z_meas, estimates)