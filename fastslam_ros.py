#O objetivo deste ficheiro é fazer a ponte entre os algoritmos de fastslam e de deteção dos arukos com os dados do rosbag e posteriormente do robo


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Float32
from cv_bridge import CvBridge
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
            self.slam = FastSLAM1(initial_pose=self.latest_odom, num_particles=300)
            self.last_time = time.time()
            self.get_logger().info("FastSLAM inicializado com a odometria inicial!")

    def imagem(self, msg):

        if self.slam is None or self.latest_odom is None:
            return

        current_time = time.time()
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
            b = math.atan2(lx, ly)

            measurements.append([f["aruco_id"], r, b])
        
        odom_progress_from_start = 0.0
        if self.start_pose is not None:
            dx = self.latest_odom[0] - self.start_pose[0]
            dy = self.latest_odom[1] - self.start_pose[1]
            start_theta = self.start_pose[2]
            odom_progress_from_start = (dx * math.cos(start_theta) + dy * math.sin(start_theta))

        particles, est_pose, est_map = self.slam.step(self.latest_odom, measurements, dt)

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
            path_len = len(best_live_particle.path)

            # First wait until we lose the starting landmark
            if (self.start_landmark_id not in visible_ids and path_len > self.min_lap_points 
                and odom_progress_from_start > self.start_landmark_forward_distance):
                self.start_landmark_lost = True

            # Then finish lap when we see it again
            if (self.start_landmark_lost and self.start_landmark_id in visible_ids):
                self.lap_finished = True

                if self.slam.best_particle_ever is not None:
                    best_particle = self.slam.best_particle_ever
                else:
                    best_particle = max(self.slam.particles, key=lambda p: p.weight)

                self.get_logger().info(
                    f"Lap finished. Selected saved particle weight: {best_particle.weight}"
                )

                self.best_weight_path = list(best_particle.path)

                self.best_weight_landmarks = {m_id: [float(ekf.state_estimate[0]), float(ekf.state_estimate[1])] for m_id, ekf in best_particle.landmarks.items()}

                self.get_logger().info(f"Lap finished. Best particle weight: {best_particle.weight}")

                self.publish_particles(particles, msg.header)
                self.publish_map(self.best_weight_landmarks, msg.header)
                self.publish_best_weight_path(msg.header)
                self.compute_and_publish_error()
                self.publish_odom_only_path(msg.header)
                self.publish_amcl_path(msg.header)

                return

        # Pose keeps publishing live
        self.publish_pose(est_pose, msg.header)

        # Camera debug keeps publishing live
        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)

    #Cria o formato de point cloud para publicar topicos como a nuvem de particulas e as landmarks
    def create_point_cloud(self, points, header, r, g, b):
        """Converte uma lista de [x, y, z] numa mensagem PointCloud2 com cor RGBA"""
        msg = PointCloud2()
        msg.header.frame_id = "odom"
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
    
    #Função para obter os parâmetros de alinhar
    def get_alignment_params(self):

        angle_deg = 0
        angle_rad = math.radians(angle_deg)

        tx = 0
        ty = 0

        scale = 1.0

        if self.best_weight_path:
            cx = sum(p[0] for p in self.best_weight_path) / len(self.best_weight_path)
            cy = sum(p[1] for p in self.best_weight_path) / len(self.best_weight_path)
        else:
            cx = 0.0
            cy = 0.0

        return cx, cy, angle_rad, tx, ty, scale


    #Alinhamento dos pontos para o foxglove 
    def align_point(self, x, y):

        cx, cy, angle_rad, tx, ty, scale = self.get_alignment_params()

        dx = x - cx
        dy = y - cy

        # Scale relative to center
        dx *= scale
        dy *= scale

        # Rotate + translate
        x_aligned = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + tx
        y_aligned = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + ty

        return x_aligned, y_aligned

    #Publicar a nuvem de particulas
    def publish_particles(self, particles, header):

        if not particles:
            return

        points = []

        for p in particles:
            x, y = self.align_point(float(p[0]), float(p[1]))
            points.append([x, y, 0.05])

        cloud_msg = self.create_point_cloud(points,header,255,0,0)
        self.particles_pub.publish(cloud_msg)

    #Publicar a trajetória do odom apenas
    def publish_odom_only_path(self, header):
        if not self.odom_only_path:
            return

        points = []

        for p in self.odom_only_path:
            x, y = self.align_point(float(p[0]), float(p[1]))
            points.append([x, y, float(p[2])])

        cloud_msg = self.create_point_cloud(points, header, 255, 165, 0)
        self.odom_only_path_pub.publish(cloud_msg)

    #Publicar as landmarks do mapa
    def publish_map(self, est_map, header):

        if not est_map:
            return

        points = []

        for _, coords in est_map.items():
            x, y = self.align_point(float(coords[0]),float(coords[1]))
            points.append([x,y,0.1])

        cloud_msg = self.create_point_cloud(points,header,0,255,0)
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
            x, y = self.align_point(float(p[0]),float(p[1]))
            aligned_path.append([x,y,float(p[2])])

        cloud_msg = self.create_point_cloud(aligned_path,header,0,0,255)
        self.best_weight_path_pub.publish(cloud_msg)

    #Função chamada sempre que se recebe uma mensagem no tópico da posição estimada pelo amcl
    def amcl_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.amcl_path_points.append([float(x), float(y)])

    #Publicar a trajetória do amcl, alinhada com a trajetória do fastslam para comparação no foxglove
    def publish_amcl_path(self, header):

        if not self.amcl_path_points:
            return

        points = []

        for p in self.amcl_path_points:
            x, y = self.align_point(float(p[0]), float(p[1]))
            points.append([x, y, 0.14])

        cloud_msg = self.create_point_cloud(
            points,
            header,
            255,   # R
            0,     # G
            255    # B
        )

        self.amcl_path_pub.publish(cloud_msg)

    #Publicar a trajetória do amcl
    def compute_and_publish_error(self):

        if len(self.best_weight_path) < 2 or len(self.amcl_path_points) < 2:
            return

        n = min(len(self.best_weight_path), len(self.amcl_path_points))

        errors = []

        for i in range(n):

            sx, sy, _ = self.best_weight_path[i]
            gx, gy = self.amcl_path_points[i]

            sx_rot, sy_rot = self.align_point(sx, sy)

            error = math.sqrt((sx_rot - gx) ** 2 + (sy_rot - gy) ** 2)
            errors.append(error)

        rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))

        msg = Float32()
        msg.data = float(rmse)

        self.error_pub.publish(msg)

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