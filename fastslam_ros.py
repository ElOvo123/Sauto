#O objetivo deste ficheiro é fazer a ponte entre os algoritmos de fastslam e de deteção dos arukos com os dados do rosbag e posteriormente do robo

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from collections import defaultdict
import cv2
import numpy as np
import math
import time
import struct
import json
import os
import sys

from feature_extraction import ArucoFeatureExtractor
from fastslam1 import FastSLAM1

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

class FastSlam_ROS(Node):

    def __init__(self):
        super().__init__('fastslam_ros')

        self.bridge = CvBridge()
        self.extractor = ArucoFeatureExtractor()

        self.initial_odom = None   
        self.initial_amcl = None
        self.sync_odom = None

        self.latest_odom = None
        self.slam = None

        self.step_times_list = []
        self.num_particles = 300 # Valor por defeito
        
        # Lê o número de partículas se o script do benchmark o enviar
        if os.path.exists("parametros_atuais.json"):
            try:
                with open("parametros_atuais.json", "r") as f:
                    params = json.load(f)
                    if "num_particles" in params:
                        self.num_particles = params["num_particles"]
            except Exception:
                pass

        self.last_time = None

        self.best_weight_path = []
        self.best_weight_landmarks = {}

        self.amcl_path = Path()
        self.amcl_path.header.frame_id = "map"
        self.amcl_path_points = []

        # ==========================================
        # HISTÓRICOS PARA O OPTUNA (CÁLCULO DO ERRO)
        # ==========================================
        self.amcl_history = []  
        self.slam_history = []  

        # Lap detection
        self.lap_started = False
        self.lap_finished = False
        self.start_landmark_id = None
        self.start_landmark_lost = False
        self.start_pose = None
        self.start_landmark_forward_distance = 0.0
        self.odom_only_path = []

        self.min_lap_points = 30

        self.frames_apos_loop_closure = 0

        self.latest_amcl = None
        self.latest_amcl_stamp = None

        #Subscrições e Publicações
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.imagem, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odometria, 10)
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        self.error_pub = self.create_publisher(Float32, '/fastslam/alignment_error', 10)
        self.odom_only_path_pub = self.create_publisher(PointCloud2,'/fastslam/odom_only_path',10)

        self.image_pub = self.create_publisher(Image, '/camera/aruco_debug', 10)
        self.particles_pub = self.create_publisher(PointCloud2, '/fastslam/particles', 10) 
        self.map_pub = self.create_publisher(PointCloud2, '/fastslam/map_markers', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fastslam/robot_pose', 10)
        self.best_weight_path_pub = self.create_publisher(PointCloud2, '/fastslam/best_weight_path', 10)
        self.amcl_path_pub = self.create_publisher(PointCloud2,'/amcl/path',10)
        self.true_landmarks_pub = self.create_publisher(PointCloud2, '/fastslam/true_landmarks', 10)

        self.get_logger().info("FastSLAM ROS Node iniciado!")

    def odometria (self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = euler_to_quaternion(msg.pose.pose.orientation)

        self.latest_odom = [x, y, theta]
        self.odom_only_path.append([float(x), float(y), 0.12])

        if self.slam is None:
            self.initial_odom = list(self.latest_odom)
            self.slam = FastSLAM1(initial_pose=self.latest_odom, num_particles=self.num_particles, seed=42)
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

        # ==========================================
        # GRAVAR HISTÓRICO DO SLAM PARA OPTUNA
        # ==========================================
        ax, ay = self.align_point(est_pose[0], est_pose[1])
        self.slam_history.append((current_time, ax, ay))

        t1 = time.perf_counter()

        self.step_times_list.append(t1 - t0)

        visible_ids = [m[0] for m in measurements]

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

            path_len = len(self.slam.path_nodes)
            
            # 1. Detetar se o perdemos de vista há muito tempo
            if (self.start_landmark_id not in visible_ids and path_len > self.min_lap_points 
                and odom_progress_from_start > self.start_landmark_forward_distance):
                self.start_landmark_lost = True

            # 2. O MOMENTO DO LOOP CLOSURE
            if (self.start_landmark_lost and self.start_landmark_id in visible_ids):
                
                # Em vez de acabar já, vamos deixar o FastSLAM "mastigar" esta informação!
                self.frames_apos_loop_closure += 1
                
                if self.frames_apos_loop_closure == 1:
                    self.get_logger().info("LOOP CLOSURE DETETADO! A estabilizar partículas...")

                # Só acaba verdadeiramente após 15 frames a olhar para o ArUco inicial
                if self.frames_apos_loop_closure > 15:
                    self.lap_finished = True
                    
                    self.get_logger().info("Multiverso colapsado. A extrair a melhor trajetória!")

                    # (A partir daqui é exatamente o teu código que já tinhas)
                    best_p = self.slam.best_particle_before_resample
                    if best_p is None:
                        best_p = max(self.slam.particles, key=lambda p: p.weight)

                    best_particle = {
                        "weight": float(best_p.weight),
                        "landmarks": best_p.landmarks,
                        "node_id": best_p.node_id
                    }
                    
                    self.best_weight_path = self.slam.reconstruct_path_from_node(best_particle["node_id"])
                    self.best_weight_landmarks = {m_id: [float(ekf.state_estimate[0]), float(ekf.state_estimate[1])] for m_id, ekf in best_particle["landmarks"].items()}

                    self.publish_particles(particles, msg.header)
                    self.publish_map(self.best_weight_landmarks, msg.header)
                    self.publish_best_weight_path(msg.header)
                    self.publish_odom_only_path(msg.header)
                    
                # ==================================================
                    # AVALIAÇÃO: DISTÂNCIA DE CHAMFER DA MELHOR PARTÍCULA
                    # ==================================================
                    self.get_logger().info("A avaliar a Trajetória Final da Melhor Partícula (Chamfer)...")
                    
                    # 1. Alinhar a trajetória reconstruída da melhor partícula com o mapa
                    final_slam_path = []
                    for p in self.best_weight_path:
                        ax, ay = self.align_point(float(p[0]), float(p[1]))
                        final_slam_path.append((ax, ay))

                    # 2. Funções de comprimento
                    def calc_length_amcl(hist):
                        dist = 0.0
                        for i in range(1, len(hist)):
                            dx = hist[i][1] - hist[i-1][1]
                            dy = hist[i][2] - hist[i-1][2]
                            dist += math.hypot(dx, dy)
                        return dist

                    def calc_length_slam(path):
                        dist = 0.0
                        for i in range(1, len(path)):
                            dx = path[i][0] - path[i-1][0]
                            dy = path[i][1] - path[i-1][1]
                            dist += math.hypot(dx, dy)
                        return dist
                    
                    amcl_len = calc_length_amcl(self.amcl_history)
                    slam_len = calc_length_slam(final_slam_path)
                    
                    # 3. Filtro de Sanidade: O caminho da partícula vencedora andou tudo?
                    # (Agora sim este filtro vai funcionar na perfeição!)
                    if slam_len < 0.7 * amcl_len or slam_len > 1.3 * amcl_len:
                        self.get_logger().warn(f"PENALIZAÇÃO: Trajetória falsa. SLAM: {slam_len:.2f}m | AMCL: {amcl_len:.2f}m")
                        rmse = 999.0
                    else:
                        # 4. Chamfer puro usando APENAS o caminho reconstruído da melhor partícula
                        erro_slam_para_amcl = []
                        for s_x, s_y in final_slam_path:
                            if not self.amcl_history: break
                            min_dist_sq = min((s_x - a_x)**2 + (s_y - a_y)**2 for _, a_x, a_y in self.amcl_history)
                            erro_slam_para_amcl.append(min_dist_sq)
                            
                        erro_amcl_para_slam = []
                        for _, a_x, a_y in self.amcl_history:
                            if not final_slam_path: break
                            min_dist_sq = min((a_x - s_x)**2 + (a_y - s_y)**2 for s_x, s_y in final_slam_path)
                            erro_amcl_para_slam.append(min_dist_sq)
                            
                        todos_os_erros_sq = erro_slam_para_amcl + erro_amcl_para_slam
                        if len(todos_os_erros_sq) > 0:
                            rmse = math.sqrt(sum(todos_os_erros_sq) / len(todos_os_erros_sq))
                        else:
                            rmse = 999.0
                            
                    self.get_logger().info(f"FIM DA VOLTA! Erro Chamfer (Melhor Partícula): {rmse:.4f} metros")

                    with open("resultado_rmse.txt", "w") as f:
                        f.write(str(rmse))

                    if len(self.step_times_list) > 0:
                        avg_time_ms = (sum(self.step_times_list) / len(self.step_times_list)) * 1000
                    else:
                        avg_time_ms = 0.0
                        
                    with open("resultado_time.txt", "w") as f:
                        f.write(str(avg_time_ms))
                    
                    import sys
                    sys.exit(0)

        # Pose keeps publishing live
        self.publish_pose(est_pose, msg.header)
        self.publish_particles(particles, msg.header)
        self.publish_map(est_map, msg.header)

        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)

    def create_point_cloud(self, points, header, r, g, b, frame_id="map"):
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = header.stamp
        msg.height = 1
        msg.width = len(points)
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgba', offset=12, datatype=PointField.UINT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16 
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        
        buffer = bytearray(msg.row_step)
        a = 255 
        
        for i, p in enumerate(points):
            struct.pack_into('<fffBBBB', buffer, i * 16, p[0], p[1], p[2], b, g, r, a)
            
        msg.data = bytes(buffer)
        return msg

    def align_point(self, x, y):
        if self.sync_odom is None or self.initial_amcl is None:
            return float(x), float(y)

        dx = x - self.sync_odom[0]
        dy = y - self.sync_odom[1]
        delta_theta = self.initial_amcl[2] - self.sync_odom[2]

        rot_x = dx * math.cos(delta_theta) - dy * math.sin(delta_theta)
        rot_y = dx * math.sin(delta_theta) + dy * math.cos(delta_theta)

        x_aligned = rot_x + self.initial_amcl[0]
        y_aligned = rot_y + self.initial_amcl[1]

        return float(x_aligned), float(y_aligned)
    
    def publish_particles(self, particles, header):
        if not particles: return
        points = []
        for p in particles:
            x, y = self.align_point(float(p[0]), float(p[1]))
            points.append([x, y, 0.05])
        cloud_msg = self.create_point_cloud(points, header, 255, 0, 0)
        self.particles_pub.publish(cloud_msg)

    def publish_odom_only_path(self, header):
        if not self.odom_only_path: return
        points = []
        for p in self.odom_only_path:
            x, y = self.align_point(float(p[0]), float(p[1]))
            points.append([x, y, float(p[2])])
        cloud_msg = self.create_point_cloud(points, header, 255, 165, 0)
        self.odom_only_path_pub.publish(cloud_msg)

    def publish_map(self, est_map, header):
        if not est_map: return
        points = []
        for _, coords in est_map.items():
            x, y = self.align_point(float(coords[0]), float(coords[1]))
            points.append([x, y, 0.1])
        cloud_msg = self.create_point_cloud(points, header, 0, 255, 0)
        self.map_pub.publish(cloud_msg)

    def publish_pose(self, est_pose, header):
        msg = PoseStamped()
        msg.header.frame_id = "odom"
        msg.header.stamp = header.stamp
        msg.pose.position.x = float(est_pose[0])
        msg.pose.position.y = float(est_pose[1])
        msg.pose.orientation = quaternion_to_euler(est_pose[2])
        self.pose_pub.publish(msg)

    def publish_best_weight_path(self, header):
        if not self.best_weight_path: return
        aligned_path = []
        for p in self.best_weight_path:
            x, y = self.align_point(float(p[0]), float(p[1]))
            aligned_path.append([x, y, float(p[2])])
        cloud_msg = self.create_point_cloud(aligned_path, header, 0, 0, 255)
        self.best_weight_path_pub.publish(cloud_msg)

    def amcl_callback(self, msg):
        if self.latest_odom is None:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = euler_to_quaternion(msg.pose.pose.orientation)

        self.latest_amcl = [x, y, theta]
        self.latest_amcl_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # ==========================================
        # GRAVAR HISTÓRICO DO AMCL PARA OPTUNA
        # ==========================================
        self.amcl_history.append((self.latest_amcl_stamp, float(x), float(y)))

        if not self.amcl_path_points:
            self.initial_amcl = list(self.latest_amcl)
            self.sync_odom = list(self.latest_odom)
            self.publish_true_landmarks(msg.header)

        self.amcl_path_points.append([float(x), float(y)])
        self.publish_amcl_path(msg.header)

    def publish_amcl_path(self, header):
        if not self.amcl_path_points: return
        points = []
        for p in self.amcl_path_points:
            x, y = float(p[0]), float(p[1])
            points.append([x, y, 0.14])
        cloud_msg = self.create_point_cloud(points, header, 255, 0, 255, frame_id="map")
        self.amcl_path_pub.publish(cloud_msg)

    def publish_true_landmarks(self, header):
        offset_x = -1.918  
        offset_y = 0.563
        self.landmarks = {
            19: [0.07, 4.39], 0: [0.07, 7.39], 17: [1.67, 8.79], 18: [0.07, 10.34],
            16: [0.07, 13.34], 5: [0.17, 15.74], 15: [2.68, 14.38], 14: [6.14, 15.68],
            13: [8.55, 14.35], 11: [13.29, 15.68], 12: [15.74, 15.00], 10: [15.66, 9.76],
            7: [14.08, 6.86], 9: [14.44, 5.81], 8: [15.66, 2.46], 6: [15.6, 0.01],
            2: [9.67, 0.07], 1: [8.20, 1.6], 3: [3.71, 0.05], 4: [0.02, 0.75], 
        }
        points = []
        for lm_id, coords in self.landmarks.items():
            phys_x = coords[0]
            phys_y = coords[1]
            map_x = phys_y + offset_x
            map_y = -phys_x + offset_y
            points.append([map_x, map_y, 0.2])

        cloud_msg = self.create_point_cloud(points, header, 255, 255, 0, frame_id="map")
        self.true_landmarks_pub.publish(cloud_msg)

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