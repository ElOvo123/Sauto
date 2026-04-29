Particle Filter:

Initialize N particles from initial distribution
Set all weights = 1/N

For each measurement z_k:

    # Prediction
    For each particle i:
        x_i = motion_model(x_i, u_k) + process_noise

    # Update
    For each particle i:
        weight_i = likelihood(z_k | x_i)

    Normalize weights

    Estimate state:
        x_est = weighted_mean(particles, weights)

    Compute effective sample size:
        N_eff = 1 / sum(weights^2)

    If N_eff < resample_threshold:
        Resample particles according to weights
        Reset weights = 1/N


Extended Kalman Filter:

Initialize state estimate x_est from initial guess
Initialize covariance P
Set process noise covariance Q
Set measurement noise covariance R

For each time step k with control input u_k and measurement z_k:

    # Prediction
    Compute motion Jacobian:
        F_k = ∂f/∂x evaluated at x_est

    Predict state:
        x_pred = f(x_est, u_k)

    Predict covariance:
        P_pred = F_k P F_kᵀ + Q

    # Measurement prediction
    Compute measurement Jacobian:
        H_k = ∂h/∂x evaluated at x_pred

    Predict measurement:
        z_pred = h(x_pred)

    # Innovation
    innovation = z_k - z_pred

    Innovation covariance:
        S = H_k P_pred H_kᵀ + R

    # Kalman gain
    K = P_pred H_kᵀ S⁻¹

    # Update
    x_est = x_pred + K innovation

    P = (I - K H_k) P_pred

    # Optional
    Save x_est and P


Feature Extraction:


Class ArucoFeatureExtractor:

    Initialize:
        Load predefined ArUco dictionary
        Create detector parameters

    Function extract(frame):

        Initialize empty list: detected_features

        If frame is None:
            Return detected_features

        Convert frame to grayscale

        Detect markers:
            corners, ids = detectMarkers(gray_image)

        If no markers detected:
            Return detected_features

        For each detected marker:

            Extract marker corners

            Compute center:
                center = mean(corners)

            Create feature:
                feature = {
                    id: marker_id
                    type: "aruco"
                    corners: marker_corners
                    center_px: center
                }

            Add feature to detected_features

        Return detected_features

Input image
    ↓
Grayscale conversion
    ↓
Aruco detection
    ↓
Loop markers
    ↓
Compute center + store data
    ↓
Return feature list

Feature Marking:

Initialize empty feature map
Set next_feature_id = 0
Set association_threshold = d_thresh

For each measurement set z_k:

    For each detected feature z_i in z_k:

        # Find best match
        best_distance = infinity
        best_feature = None

        For each existing feature f_j in map:

            d = distance(z_i, f_j)

            If d < best_distance:
                best_distance = d
                best_feature = f_j

        # Association decision
        If best_distance < association_threshold:

            Assign z_i to best_feature

            Update feature:
                f_j.position = z_i
                f_j.seen_count += 1

        Else:

            Create new feature:
                id = next_feature_id
                position = z_i
                seen_count = 1

            Add feature to map

            next_feature_id += 1


FastSLAM 1.0: