"""
extended_kalman_filter.py

Generic Extended Kalman Filter implementation.
Variable names are written clearly and avoid single-letter names.
"""

from typing import Callable, Optional, Tuple
import numpy as np


Array = np.ndarray


class ExtendedKalmanFilter:
    def __init__(
        self,
        initial_state: Array,
        initial_covariance: Array,
        process_noise_covariance: Array,
        measurement_noise_covariance: Array,
        motion_model: Callable[[Array, Optional[Array]], Array],
        measurement_model: Callable[[Array], Array],
        motion_jacobian: Callable[[Array, Optional[Array]], Array],
        measurement_jacobian: Callable[[Array], Array],
    ):
        self.state_estimate = np.asarray(initial_state, dtype=float)
        self.covariance_estimate = np.asarray(initial_covariance, dtype=float)

        self.process_noise_covariance = np.asarray(process_noise_covariance, dtype=float)
        self.measurement_noise_covariance = np.asarray(
            measurement_noise_covariance,
            dtype=float,
        )

        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.motion_jacobian = motion_jacobian
        self.measurement_jacobian = measurement_jacobian

        self.state_dimension = self.state_estimate.shape[0]
        self.identity_matrix = np.eye(self.state_dimension)

    def predict(self, control_input: Optional[Array] = None) -> Tuple[Array, Array]:
        motion_model_jacobian = self.motion_jacobian(self.state_estimate, control_input,)

        predicted_state = self.motion_model(self.state_estimate, control_input,)

        predicted_covariance = (motion_model_jacobian @ self.covariance_estimate @ motion_model_jacobian.T + self.process_noise_covariance)

        self.state_estimate = predicted_state
        self.covariance_estimate = predicted_covariance

        return self.state_estimate, self.covariance_estimate

    def update(self, measurement: Array) -> Tuple[Array, Array]:
        measurement = np.asarray(measurement, dtype=float)

        measurement_model_jacobian = self.measurement_jacobian(self.state_estimate,)

        predicted_measurement = self.measurement_model(self.state_estimate,)

        innovation = measurement - predicted_measurement

        innovation_covariance = (measurement_model_jacobian @ self.covariance_estimate @ measurement_model_jacobian.T + self.measurement_noise_covariance)

        kalman_gain = (self.covariance_estimate @ measurement_model_jacobian.T @ np.linalg.inv(innovation_covariance))

        updated_state = self.state_estimate + kalman_gain @ innovation

        updated_covariance = (self.identity_matrix - kalman_gain @ measurement_model_jacobian) @ self.covariance_estimate

        self.state_estimate = updated_state
        self.covariance_estimate = updated_covariance

        return self.state_estimate, self.covariance_estimate

    def step(self, measurement: Array, control_input: Optional[Array] = None,) -> Tuple[Array, Array]:

        self.predict(control_input)
        self.update(measurement)

        return self.state_estimate, self.covariance_estimate

def motion_model(state_estimate, control_input):
    time_step = 0.1

    position = state_estimate[0]
    velocity = state_estimate[1]

    next_position = position + velocity * time_step
    next_velocity = velocity

    return np.array([next_position, next_velocity])


def motion_jacobian(state_estimate, control_input):
    time_step = 0.1

    return np.array([
        [1.0, time_step],
        [0.0, 1.0],
    ])


def measurement_model(state_estimate):
    position = state_estimate[0]
    return np.array([position])


def measurement_jacobian(state_estimate):
    return np.array([
        [1.0, 0.0],
    ])


def run_test():
    initial_state = np.array([0.0, 0.0])

    initial_covariance = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    process_noise_covariance = np.array([
        [0.01, 0.0],
        [0.0, 0.01],
    ])

    measurement_noise_covariance = np.array([
        [0.05],
    ])

    ekf = ExtendedKalmanFilter(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise_covariance=process_noise_covariance,
        measurement_noise_covariance=measurement_noise_covariance,
        motion_model=motion_model,
        measurement_model=measurement_model,
        motion_jacobian=motion_jacobian,
        measurement_jacobian=measurement_jacobian,
    )

    true_position = 0.0
    true_velocity = 1.0
    time_step = 0.1

    measurements = [
        0.10,
        0.21,
        0.29,
        0.42,
        0.51,
        0.60,
        0.72,
        0.81,
        0.91,
        1.02,
    ]

    print("Testing Extended Kalman Filter\n")

    for step_index, measured_position in enumerate(measurements, start=1):
        true_position = true_position + true_velocity * time_step

        measurement = np.array([measured_position])

        state_estimate, covariance_estimate = ekf.step(
            measurement=measurement,
            control_input=None,
        )

        estimated_position = state_estimate[0]
        estimated_velocity = state_estimate[1]

        print(f"Step {step_index}")
        print(f"True position:      {true_position:.3f}")
        print(f"Measured position:  {measured_position:.3f}")
        print(f"Estimated position: {estimated_position:.3f}")
        print(f"Estimated velocity: {estimated_velocity:.3f}")
        print("Covariance:")
        print(covariance_estimate)
        print("-" * 40)


if __name__ == "__main__":
    run_test()