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

        self.initial_odom = None   
        self.initial_amcl = None
        self.sync_odom = None

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
            self.initial_odom = list(self.latest_odom)
            self.slam = FastSLAM1(initial_pose=self.latest_odom, num_particles=300, seed=42)
            self.last_time = None
            self.get_logger().info("FastSLAM inicializado com a odometria inicial!")

    def imagem(self, msg):

        if self.slam is None or self.latest_odom is None:
            return

        if not hasattr(self, 'ecra_limpo'):
            empty_cloud = self.create_point_cloud([], msg.header, 0, 0, 0)
            self.particles_pub.publish(empty_cloud)
            self.map_pub.publish(empty_cloud)
            self.best_weight_path_pub.publish(empty_cloud)
            self.ecra_limpo = True

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
            measurements.append([f["aruco_id"], f["range"], f["bearing"]])
        
        odom_progress_from_start = 0.0
        if self.start_pose is not None:
            dx = self.latest_odom[0] - self.start_pose[0]
            dy = self.latest_odom[1] - self.start_pose[1]
            start_theta = self.start_pose[2]
            odom_progress_from_start = (dx * math.cos(start_theta) + dy * math.sin(start_theta))

        t0 = time.perf_counter()

        particles, est_pose, est_map = self.slam.step(self.latest_odom, measurements, dt)

        t1 = time.perf_counter()

        #self.get_logger().info(f"FastSLAM step took {(t1 - t0)*1000:.2f} ms")

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
                
                self.compute_and_write_svd_rmse()

                return

        # Pose keeps publishing live
        self.publish_pose(est_pose, msg.header)

        self.publish_particles(particles, msg.header)
        self.publish_map(est_map, msg.header)

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
            points.append([float(p[0]), float(p[1]), float(p[2])])

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

        if self.latest_odom is None:
            self.get_logger().warn("Mensagem AMCL ignorada: Odometria ainda não foi recebida.")
            return


        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        theta = euler_to_quaternion(msg.pose.pose.orientation)

        self.latest_amcl = [x, y, theta]
        self.latest_amcl_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if not self.amcl_path_points:
            self.initial_amcl = list(self.latest_amcl)
            self.sync_odom = list(self.latest_odom)
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
    

    def compute_and_write_svd_rmse(self):
        if len(self.best_weight_path) == 0 or len(self.amcl_path_points) == 0:
            with open("resultado_rmse.txt", "w") as f:
                f.write("999.0")
            return 999.0

        slam_pts = []
        amcl_pts = []

        for p_slam in self.best_weight_path:
            sx, sy = float(p_slam[0]), float(p_slam[1])

            min_dist_sq = float("inf")
            best_amcl = None

            for p_amcl in self.amcl_path_points:
                gx, gy = float(p_amcl[0]), float(p_amcl[1])

                dist_sq = (sx - gx)**2 + (sy - gy)**2

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    best_amcl = [gx, gy]

            if best_amcl is not None:
                slam_pts.append([sx, sy])
                amcl_pts.append(best_amcl)

        if len(slam_pts) < 3:
            with open("resultado_rmse.txt", "w") as f:
                f.write("999.0")
            return 999.0

        A = np.array(slam_pts)
        B = np.array(amcl_pts)

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A

        sum_sq_errors = 0.0

        for i in range(len(A)):
            aligned_pt = R @ A[i] + t
            sum_sq_errors += np.sum((aligned_pt - B[i])**2)

        ate_rmse = math.sqrt(sum_sq_errors / len(A))

        self.get_logger().info(f"ATE SVD RMSE: {ate_rmse:.4f} m")

        with open("resultado_rmse.txt", "w") as f:
            f.write(str(ate_rmse))

        return ate_rmse

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