#O objetivo deste ficheiro é fazer a ponte entre os algoritmos de fastslam e de deteção dos arukos com os dados do rosbag e posteriormente do robo


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from collections import defaultdict
import cv2
import numpy as np
import math
import time
import struct

from feature_extraction import ArucoFeatureExtractor
from fastslam1 import FastSLAM1



#Funções auxiliares para passar os dados de ros que trabalha em 3d, para os dados do slam que trabalha em 2d, e vice versa

#Passa de euler para quaternião
def euler_to_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

#Passa de quaternião para euler
def quaternion_to_euler(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q



#Aqui, criamos um nó que irá subscrever aos tópicos do rosbag, irá chamar os algoritmos de extração de features e de fastslam, e publica tópicos ao rosbag.
class FastSlam_ROS(Node):

    #Inicialização do nó
    def __init__(self):
        super().__init__('fastslam_ros')

        self.bridge = CvBridge()
        self.extractor = ArucoFeatureExtractor()

        self.latest_odom = None
        self.slam = None
        self.last_time = None

        # Best final trajectory
        self.best_weight_path = []

        # Store full trajectory of every particle
        self.best_weight_landmarks = {}

        # AMCL path + error
        self.amcl_path = Path()
        self.amcl_path.header.frame_id = "map"
        self.amcl_path_points = []

        # Lap detection
        self.lap_started = False
        self.lap_finished = False
        self.start_landmark_id = None
        self.start_landmark_lost = False
        self.start_pose = None
        self.start_landmark_forward_distance = 0.0
        self.odom_only_path = []

        # Minimum trajectory size before allowing lap closure
        self.min_lap_points = 30

        # Timing
        self.step_times = defaultdict(list)
        self.print_timing_every = 30
        self.frame_counter = 0

        # For error analysis
        self.latest_amcl = None
        self.latest_amcl_stamp = None

        self.prev_odom_for_calib = None
        self.prev_amcl_for_calib = None

        self.motion_errors = []
        self.measurement_errors = []

        #Subscrições:

        #Subscrever o tópico da imagem comprimida
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.imagem, 10)
        #Subscrever o tópico da odometria
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odometria, 10)
        #Subscrever o tópico da posição estimada pelo amcl
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        #Subscrever o tópico da posição estimada pelo amcl para calcular o erro de alinhamento entre o fastslam e o amcl
        self.error_pub = self.create_publisher(Float32, '/fastslam/alignment_error', 10)
        #Subscrever o tópico da odometria para publicar a trajetória do odom apenas
        self.odom_only_path_pub = self.create_publisher(PointCloud2,'/fastslam/odom_only_path',10)


        #Publicaçẽs:

        #Publicar o tópico da camera com a deteção dos arukos
        self.image_pub = self.create_publisher(Image, '/camera/aruco_debug', 10)
        #Publicar o tópico da nuvem de particulas
        self.particles_pub = self.create_publisher(PointCloud2, '/fastslam/particles', 10) #o formato de point cloud permite um maior numero de funcionalidades no foxglove
        #Publicar o tópico das landmarks
        self.map_pub = self.create_publisher(PointCloud2, '/fastslam/map_markers', 10)
        #Publicar o tópico com a posição do robo
        self.pose_pub = self.create_publisher(PoseStamped, '/fastslam/robot_pose', 10)
        #Publicar o tópico com a melhor trajetória
        self.best_weight_path_pub = self.create_publisher(PointCloud2, '/fastslam/best_weight_path', 10)
        #Publicar o tópico com a trajetória do amcl
        self.amcl_path_pub = self.create_publisher(PointCloud2,'/amcl/path',10)
        # Publicar o tópico com as true landmarks
        self.true_landmarks_pub = self.create_publisher(PointCloud2, '/fastslam/true_landmarks', 10)

        #O logger é semelhante a um print, mas para além disso cria um tópico ros com os loggs
        self.get_logger().info("FastSLAM ROS Node iniciado!")

    #Função chamada sempre que se recebe uma mensagem no tópico da odometria
    def odometria (self, msg):

        #Atualização da última posição com base nos dados de odometria recebidos
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = euler_to_quaternion(msg.pose.pose.orientation)

        self.latest_odom = [x, y, theta]
        self.odom_only_path.append([float(x), float(y), 0.12])

        #O fastslam só é iniciado após receber a primeira mensagem de odometria (o slam precisa de uma posição inicial)
        if self.slam is None:
            self.slam = FastSLAM1(initial_pose=self.latest_odom, num_particles=100, seed=42)
            self.last_time = None
            self.get_logger().info("FastSLAM inicializado com a odometria inicial!")

    def imagem(self, msg):

        if self.slam is None or self.latest_odom is None:
            return

        # Use bag timestamp instead of wall clock
        current_time = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        self.last_time = current_time

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        features = self.extractor.extract(frame, robot_pose=None)

        measurements = []
        for f in features:
            lx = f["landmark_x"]
            ly = f["landmark_y"]

            r = math.hypot(lx, ly)
            b = -math.atan2(lx, ly)

            measurements.append([f["aruco_id"], r, b])
        
        odom_progress_from_start = 0.0
        if self.start_pose is not None:
            dx = self.latest_odom[0] - self.start_pose[0]
            dy = self.latest_odom[1] - self.start_pose[1]
            start_theta = self.start_pose[2]
            odom_progress_from_start = (dx * math.cos(start_theta) + dy * math.sin(start_theta))

        t0 = time.perf_counter()

        particles, est_pose, est_map = self.slam.step(self.latest_odom, measurements, dt)
        self.collect_motion_calibration_sample()
        self.collect_measurement_calibration_samples(measurements)

        t1 = time.perf_counter()

        self.get_logger().info(f"FastSLAM step took {(t1 - t0)*1000:.2f} ms")

        visible_ids = [m[0] for m in measurements]

        # Start lap with first visible landmark
        if not self.lap_started and visible_ids:
            self.start_landmark_id = visible_ids[0]
            self.lap_started = True
            self.start_landmark_lost = False
            self.start_pose = list(self.latest_odom)

            start_measurement = next((m for m in measurements if m[0] == self.start_landmark_id),None)

            if start_measurement is not None:
                self.start_landmark_forward_distance = max(0.0, start_measurement[1] * math.cos(start_measurement[2]))
            else:
                self.start_landmark_forward_distance = 0.0

            self.get_logger().info(f"Lap started with landmark {self.start_landmark_id}")

        elif self.lap_started and not self.lap_finished:

            best_live_particle = max(self.slam.particles, key=lambda p: p.weight)
            path_len = len(self.slam.path_nodes)
            # First wait until we lose the starting landmark
            if (self.start_landmark_id not in visible_ids and path_len > self.min_lap_points 
                and odom_progress_from_start > self.start_landmark_forward_distance):
                self.start_landmark_lost = True

            # Then finish lap when we see it again
            if (self.start_landmark_lost and self.start_landmark_id in visible_ids):
                self.lap_finished = True

                # Melhor partícula no fim da lap, antes do resampling
                best_p = self.slam.best_particle_before_resample

                if best_p is None:
                    best_p = max(self.slam.particles, key=lambda p: p.weight)

                best_particle = {
                    "weight": float(best_p.weight),
                    "landmarks": best_p.landmarks,
                    "node_id": best_p.node_id
                }
                
                self.get_logger().info(f"Lap finished. Selected saved particle weight: {best_particle['weight']}")


                self.best_weight_path = self.slam.reconstruct_path_from_node(best_particle["node_id"])

                self.best_weight_landmarks = {m_id: [float(ekf.state_estimate[0]), float(ekf.state_estimate[1])] for m_id, ekf in best_particle["landmarks"].items()}

                # Compute optimal alignment once the map is finalized
                self.get_logger().info(f"Lap finished. Best particle weight: {best_particle['weight']:.6f}. Computing optimal alignment...")

            
                self.publish_particles(particles, msg.header)
                self.publish_map(self.best_weight_landmarks, msg.header)
                self.publish_best_weight_path(msg.header)
                self.publish_odom_only_path(msg.header)
                
                self.estimate_motion_parameters()
                self.estimate_measurement_noise()

                return

        # Pose keeps publishing live
        self.publish_pose(est_pose, msg.header)

        # Camera debug keeps publishing live
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)

    #Cria o formato de point cloud para publicar topicos como a nuvem de particulas e as landmarks
    def create_point_cloud(self, points, header, r, g, b, frame_id="map"):
        """Converte uma lista de [x, y, z] numa mensagem PointCloud2 com cor RGBA"""
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = header.stamp
        msg.height = 1
        msg.width = len(points)
        
        # Definir os eixos X, Y, Z e a COR (RGBA)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16 # 3 floats (12 bytes) + 4 bytes de cor = 16 bytes
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        
        buffer = bytearray(msg.row_step)
        a = 255 # Opacidade máxima (não transparente)
        
        for i, p in enumerate(points):
            # O empacotamento padrão do ROS para a cor lê a ordem Azul, Verde, Vermelho, Alfa (BGRA)
            struct.pack_into('<fffBBBB', buffer, i * 16, p[0], p[1], p[2], b, g, r, a)
            
        msg.data = bytes(buffer)
        return msg
    
    
    #Publicar a nuvem de particulas
    def publish_particles(self, particles, header):

        if not particles:
            return

        points = []

        for p in particles:
            # ROTATE FASTSLAM PARTICLES ONLY
            points.append([float(p[0]), float(p[1]), 0.05])

        cloud_msg = self.create_point_cloud(points, header, 255, 0, 0)
        self.particles_pub.publish(cloud_msg)

    #Publicar a trajetória do odom apenas
    def publish_odom_only_path(self, header):
        if not self.odom_only_path:
            return

        points = []

        for p in self.odom_only_path:
            x, y = float(p[0]), float(p[1])
            points.append([x, y, float(p[2])])

        cloud_msg = self.create_point_cloud(points, header, 255, 165, 0)
        self.odom_only_path_pub.publish(cloud_msg)

    #Publicar as landmarks do mapa
    def publish_map(self, est_map, header):

        if not est_map:
            return

        points = []

        for _, coords in est_map.items():
            # ROTATE FASTSLAM LANDMARKS ONLY
            points.append([float(coords[0]), float(coords[1]), 0.1])

        cloud_msg = self.create_point_cloud(points, header, 0, 255, 0)
        self.map_pub.publish(cloud_msg)

    #Publicar a posição estimada
    def publish_pose(self, est_pose, header):
        msg = PoseStamped()
        msg.header.frame_id = "odom"
        msg.header.stamp = header.stamp
        msg.pose.position.x = float(est_pose[0])
        msg.pose.position.y = float(est_pose[1])
        msg.pose.orientation = quaternion_to_euler(est_pose[2])
        self.pose_pub.publish(msg)

    #Publicar a melhor trajetória
    def publish_best_weight_path(self, header):

        if not self.best_weight_path:
            return

        aligned_path = []

        for p in self.best_weight_path:
            # ROTATE FASTSLAM BEST PATH ONLY
            aligned_path.append([float(p[0]), float(p[1]), float(p[2])])

        cloud_msg = self.create_point_cloud(aligned_path, header, 0, 0, 255)
        self.best_weight_path_pub.publish(cloud_msg)

    #Função chamada sempre que se recebe uma mensagem no tópico da posição estimada pelo amcl
    def amcl_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        theta = euler_to_quaternion(msg.pose.pose.orientation)

        self.latest_amcl = [x, y, theta]
        self.latest_amcl_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self.amcl_path_points:
            self.get_logger().info(f"--- FIRST AMCL POSE (Map Frame): X={x:.3f}, Y={y:.3f} ---")
            self.publish_true_landmarks(msg.header)

        self.amcl_path_points.append([float(x), float(y)])

        self.publish_amcl_path(msg.header)

    #Publicar a trajetória do amcl, alinhada com a trajetória do fastslam para comparação no foxglove
    def publish_amcl_path(self, header):

        if not self.amcl_path_points:
            return

        points = []

        for p in self.amcl_path_points:
            x, y = float(p[0]), float(p[1])
            points.append([x, y, 0.14])

        cloud_msg = self.create_point_cloud(
            points,
            header,
            255,   # R
            0,     # G
            255,    # B
            frame_id="map"
        )

        #print (points)

        self.amcl_path_pub.publish(cloud_msg)

    def publish_true_landmarks(self, header):
        
        # Offsets calculated after rotating 90-degrees clockwise
        offset_x = -1.918  
        offset_y = 0.563

        self.landmarks = {
             # Left corridor
            19: [0.07, 4.39],
            0: [0.07, 7.39],
            17: [1.67, 8.79],
            18: [0.07, 10.34],
            16: [0.07, 13.34],
            5: [0.17, 15.74],
            # Top corridor
            15: [2.68, 14.38],
            14: [6.14, 15.68],
            13: [8.55, 14.35],
            11: [13.29, 15.68],
            12: [15.74, 15.00],
            # Right corridor
            10: [15.66, 9.76],   
            7: [14.08, 6.86], 
            9: [14.44, 5.81], 
            8: [15.66, 2.46],
            6: [15.6, 0.01],
            # bottom corridor 
            2: [9.67, 0.07],   
            1: [8.20, 1.6],
            3: [3.71, 0.05], 
            4: [0.02, 0.75], 
        }

        points = []
        for lm_id, coords in self.landmarks.items():
            phys_x = coords[0]
            phys_y = coords[1]

            # Apply 90-degree rotation AND the translation offset
            map_x = phys_y + offset_x
            map_y = -phys_x + offset_y
            
            # Z=0.2 elevates them slightly so they don't clip into the floor
            points.append([map_x, map_y, 0.2])

        # Publish in the "map" frame, colored Yellow (R=255, G=255, B=0)
        cloud_msg = self.create_point_cloud(
            points, 
            header, 
            255, 255, 0, 
            frame_id="map"
        )
        
        # You will need to create this publisher in your __init__:
        # self.true_landmarks_pub = self.create_publisher(PointCloud2, '/fastslam/true_landmarks', 10)
        self.true_landmarks_pub.publish(cloud_msg)

    def compute_optimal_alignment(self):
        """
        Uses SVD to find optimal rotation and translation between
        estimated landmarks and ground truth landmarks in the map frame.
        """
        # 1. Pair up the landmarks (only those that appear in both)
        est_pts = []
        true_pts = []
        
        # Hardcoded ground truth used in publish_true_landmarks
        true_landmarks_data = {
            19: [0.07, 4.39], 0: [0.07, 7.39], 17: [1.67, 8.79], 18: [0.07, 10.34],
            16: [0.07, 13.34], 5: [0.17, 15.74], 15: [2.68, 14.38], 14: [6.14, 15.68],
            13: [8.55, 14.35], 11: [13.29, 15.68], 12: [15.74, 15.00], 10: [15.66, 9.76],
            7: [14.08, 6.86], 9: [14.44, 5.81], 8: [15.66, 2.46], 6: [15.6, 0.01],
            2: [9.67, 0.07], 1: [8.20, 1.6], 3: [3.71, 0.05], 4: [0.02, 0.75]
        }

        # Offsets used to bring physical coordinates into the map frame
        offset_x = -1.918  
        offset_y = 0.563

        for l_id, est_coords in self.best_weight_landmarks.items():
            if l_id in true_landmarks_data:
                phys_x = true_landmarks_data[l_id][0]
                phys_y = true_landmarks_data[l_id][1]

                # CRITICAL FIX: Standardize physical landmarks to the Map Frame 
                # using your exact transformation formula
                map_x = phys_y + offset_x
                map_y = -phys_x + offset_y

                mx, my = self.manual_align_point(float(est_coords[0]), float(est_coords[1]))
                est_pts.append([mx, my])
                true_pts.append([map_x, map_y])

        if len(est_pts) < 3: # Need at least 3 points for a robust alignment
            self.get_logger().warn("Not enough matching landmarks to compute optimal alignment.")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 1.0

        # Convert to numpy
        A = np.array(est_pts)
        B = np.array(true_pts)

        # Center data
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # Covariance matrix and SVD
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Ensure right-handed coordinate system (handles reflection edge cases)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Translation vector: t = centroid_B - R * centroid_A
        t = centroid_B - np.dot(R, centroid_A)
        
        angle_rad = math.atan2(R[1,0], R[0,0])
        
        self.get_logger().info(f"--- SVD Optimization Complete ---")
        self.get_logger().info(f"Fine-tuning Rotation: {math.degrees(angle_rad):.3f} degrees")
        self.get_logger().info(f"Fine-tuning Translation: X={t[0]:.3f}m, Y={t[1]:.3f}m")
        
        # Return parameters matching the expected return structure
        return centroid_A[0], centroid_A[1], angle_rad, t[0], t[1], 1.0

    def relative_motion(self, prev_pose, curr_pose):
        dx = curr_pose[0] - prev_pose[0]
        dy = curr_pose[1] - prev_pose[1]
        prev_theta = prev_pose[2]

        local_dx = math.cos(prev_theta) * dx + math.sin(prev_theta) * dy
        local_dy = -math.sin(prev_theta) * dx + math.cos(prev_theta) * dy

        trans = local_dx

        rot = curr_pose[2] - prev_pose[2]
        rot = math.atan2(math.sin(rot), math.cos(rot))

        return trans, rot


    def collect_motion_calibration_sample(self):
        if self.latest_odom is None or self.latest_amcl is None:
            return

        if self.prev_odom_for_calib is None:
            self.prev_odom_for_calib = list(self.latest_odom)
            self.prev_amcl_for_calib = list(self.latest_amcl)
            return

        odom_trans, odom_rot = self.relative_motion(
            self.prev_odom_for_calib,
            self.latest_odom
        )

        gt_trans, gt_rot = self.relative_motion(
            self.prev_amcl_for_calib,
            self.latest_amcl
        )

        e_trans = gt_trans - odom_trans
        e_rot = gt_rot - odom_rot
        e_rot = math.atan2(math.sin(e_rot), math.cos(e_rot))

        self.motion_errors.append({
            "odom_trans": odom_trans,
            "odom_rot": odom_rot,
            "e_trans": e_trans,
            "e_rot": e_rot
        })

        self.prev_odom_for_calib = list(self.latest_odom)
        self.prev_amcl_for_calib = list(self.latest_amcl)


    def estimate_motion_parameters(self):
        if len(self.motion_errors) < 30:
            self.get_logger().warn("Not enough motion samples to estimate alphas.")
            return

        samples = []

        for s in self.motion_errors:
            trans = abs(s["odom_trans"])
            rot = abs(s["odom_rot"])

            e_trans = s["e_trans"]
            e_rot = s["e_rot"]

            if trans < 1e-4 and rot < 1e-4:
                continue

            samples.append({
                "trans": trans,
                "rot": rot,
                "e_trans": e_trans,
                "e_rot": e_rot
            })

        if len(samples) < 30:
            self.get_logger().warn("Not enough valid samples to estimate alphas.")
            return

        # -------------------------
        # 1) Remove systematic bias
        # -------------------------
        mean_e_trans = np.mean([s["e_trans"] for s in samples])
        mean_e_rot = np.mean([s["e_rot"] for s in samples])

        for s in samples:
            s["res_trans"] = s["e_trans"] - mean_e_trans
            s["res_rot"] = s["e_rot"] - mean_e_rot

        # -------------------------
        # 2) Bin samples by motion size
        #    and compute variance per bin
        # -------------------------
        rot_bins = []
        trans_bins = []

        # Rotation-noise bins
        for s in samples:
            motion_mag = s["rot"] + s["trans"]

            rot_bins.append([
                s["rot"],
                s["trans"],
                s["res_rot"] ** 2,
                motion_mag
            ])

            trans_bins.append([
                s["trans"],
                s["rot"],
                s["res_trans"] ** 2,
                motion_mag
            ])

        rot_bins = np.array(rot_bins)
        trans_bins = np.array(trans_bins)

        # -------------------------
        # 3) Least squares:
        #
        # Var(rot_error)   ~= (a0*rot + a1*trans)^2
        # Var(trans_error) ~= (a2*trans + a3*rot)^2
        #
        # Therefore:
        # sqrt(variance) ~= a0*rot + a1*trans
        # -------------------------

        A_rot = []
        y_rot = []

        A_trans = []
        y_trans = []

        for row in rot_bins:
            rot = row[0]
            trans = row[1]
            err2 = row[2]

            A_rot.append([rot, trans])
            y_rot.append(math.sqrt(max(err2, 1e-12)))

        for row in trans_bins:
            trans = row[0]
            rot = row[1]
            err2 = row[2]

            A_trans.append([trans, rot])
            y_trans.append(math.sqrt(max(err2, 1e-12)))

        A_rot = np.array(A_rot)
        y_rot = np.array(y_rot)

        A_trans = np.array(A_trans)
        y_trans = np.array(y_trans)

        rot_params, _, _, _ = np.linalg.lstsq(A_rot, y_rot, rcond=None)
        trans_params, _, _, _ = np.linalg.lstsq(A_trans, y_trans, rcond=None)

        alpha0 = max(float(rot_params[0]), 1e-6)
        alpha1 = max(float(rot_params[1]), 1e-6)

        alpha2 = max(float(trans_params[0]), 1e-6)
        alpha3 = max(float(trans_params[1]), 1e-6)

        # -------------------------
        # 4) Clamp to sane FastSLAM values
        # -------------------------
        alpha0 = min(alpha0, 1.2)
        alpha1 = min(alpha1, 0.05)
        alpha2 = min(alpha2, 0.10)
        alpha3 = min(alpha3, 0.05)

        self.get_logger().info(
            f"Motion bias removed: "
            f"mean_e_trans={mean_e_trans:.6f}, "
            f"mean_e_rot={math.degrees(mean_e_rot):.3f} deg"
        )

        self.get_logger().info(
            f"Estimated FastSLAM alphas = "
            f"[{alpha0:.6f}, {alpha1:.6f}, {alpha2:.6f}, {alpha3:.6f}]"
        )

    def collect_measurement_calibration_samples(self, measurements):

        if self.latest_amcl is None:
            return

        if not hasattr(self, "landmarks"):
            return

        rx, ry, rtheta = self.latest_amcl

        cam_x = rx + 0.05 * math.cos(rtheta)
        cam_y = ry + 0.05 * math.sin(rtheta)

        for m in measurements:

            lm_id = m[0]

            if lm_id not in self.landmarks:
                continue

            measured_r = m[1]
            measured_b = m[2]

            lm_x = self.landmarks[lm_id][1] - 1.918
            lm_y = -self.landmarks[lm_id][0] + 0.563

            dx = lm_x - cam_x
            dy = lm_y - cam_y

            expected_r = math.hypot(dx, dy)

            expected_b = math.atan2(dy, dx) - rtheta
            expected_b = math.atan2(
                math.sin(expected_b),
                math.cos(expected_b)
            )

            e_r = measured_r - expected_r

            e_b = measured_b - expected_b
            e_b = math.atan2(
                math.sin(e_b),
                math.cos(e_b)
            )

            self.measurement_errors.append(
                [e_r, e_b]
            )


    def estimate_measurement_noise(self):

        if len(self.measurement_errors) < 30:
            self.get_logger().warn(
                "Not enough measurement samples."
            )
            return

        E = np.array(
            self.measurement_errors
        )

        mean = np.mean(
            E,
            axis=0
        )

        residuals = E - mean

        R = np.cov(
            residuals.T
        )

        range_noise = min(
            max(
                float(R[0,0]),
                0.01
            ),
            0.5
        )

        bearing_noise = min(
            max(
                float(R[1,1]),
                0.001
            ),
            0.2
        )

        self.get_logger().info(
            f"Estimated R_noise = "
            f"[[{range_noise:.6f},0],"
            f"[0,{bearing_noise:.6f}]]"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FastSlam_ROS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()