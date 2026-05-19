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

        self.best_weight_path = []

        self.amcl_path = Path()
        self.amcl_path.header.frame_id = "map"
        self.amcl_path_points = []

        #Subscrições:

        #Subscrever o tópico da imagem comprimida
        self.image_sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.imagem, 10)
        #Subscrever o tópico da odometria
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odometria, 10)
        #Subscrever o tópico da posição estimada pelo amcl
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_callback, 10)
        #Subscrever o tópico da posição estimada pelo amcl para calcular o erro de alinhamento entre o fastslam e o amcl
        self.error_pub = self.create_publisher(Float32, '/fastslam/alignment_error', 10)



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
        self.amcl_path_pub = self.create_publisher(Path, '/amcl/path', 10)

        #O logger é semelhante a um print, mas para além disso cria um tópico ros com os loggs
        self.get_logger().info("FastSLAM ROS Node iniciado!")

    #Função chamada sempre que se recebe uma mensagem no tópico da odometria
    def odometria (self, msg):

        #Atualização da última posição com base nos dados de odometria recebidos
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = euler_to_quaternion(msg.pose.pose.orientation)

        self.latest_odom = [x, y, theta]

        #O fastslam só é iniciado após receber a primeira mensagem de odometria (o slam precisa de uma posição inicial)
        if self.slam is None:
            self.slam = FastSLAM1(initial_pose=self.latest_odom, num_particles=100)
            self.last_time = time.time()
            self.get_logger().info("FastSLAM inicializado com a odometria inicial!")

    #Função chamada sempre que se recebe uma mensagem no tópico da imagem comprimida
    def imagem (self, msg):

        #Não é necessário analisar as imagens se o slam não estiver a correr
        if self.slam is None or self.latest_odom is None:
            return 

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        #Transforma a stream de bytes numa lista e o cv2 monta a imagem colorida
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        #função do feature extraction
        features = self.extractor.extract(frame, robot_pose=None)

        #Calcular o range e bearing de cada feature
        measurements = []
        for f in features:
            lx = f["landmark_x"]
            lz = f["landmark_y"]
            r = math.hypot(lx, lz)
            b = math.atan2(-lx, lz) 
            print(f"Feature {f['aruco_id']}: range={r:.2f}, bearing={b:.2f}")
            
            measurements.append([f["aruco_id"], r, b])


        #Enviamos os dados para o slam, que retorna as coordenadas das particulas e as melhores estimativas da posição do robo e do mapa
        particles, est_pose, est_map = self.slam.step(self.latest_odom, measurements, dt)

        #Guardar a melhor trajetória para publicar depois
        best_particle = max(self.slam.particles, key=lambda p: p.weight)
        self.best_weight_path.append([float(best_particle.state[0]), float(best_particle.state[1]), 0.15])

        # Publicar resultados convertidos
        self.publish_particles(particles, msg.header)
        self.publish_map(est_map, msg.header)
        self.publish_pose(est_pose, msg.header)
        self.publish_best_weight_path(msg.header)
        self.compute_and_publish_error()

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

    #Publicar a nuvem de particulas
    def publish_particles(self, particles, header):

        if not particles:
            return

        angle_deg = -35
        angle_rad = math.radians(angle_deg)

        # Same center used for path + landmarks
        if self.best_weight_path:
            cx = sum(p[0] for p in self.best_weight_path) / len(self.best_weight_path)
            cy = sum(p[1] for p in self.best_weight_path) / len(self.best_weight_path)
        else:
            cx = 0.0
            cy = 0.0

        points = []

        for p in particles:

            x = float(p[0])
            y = float(p[1])

            dx = x - cx
            dy = y - cy

            x_rot = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y_rot = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

            points.append([
                x_rot,
                y_rot,
                0.05
            ])

        cloud_msg = self.create_point_cloud(
            points,
            header,
            255,
            0,
            0
        )

        self.particles_pub.publish(cloud_msg)

    def publish_map(self, est_map, header):
        if not est_map:
            return

        angle_deg = -35
        angle_rad = math.radians(angle_deg)

        # Use same center as rotated trajectory
        if self.best_weight_path:
            cx = sum(p[0] for p in self.best_weight_path) / len(self.best_weight_path)
            cy = sum(p[1] for p in self.best_weight_path) / len(self.best_weight_path)
        else:
            cx = 0.0
            cy = 0.0

        points = []

        for m_id, coords in est_map.items():
            x = float(coords[0])
            y = float(coords[1])

            dx = x - cx
            dy = y - cy

            x_rot = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y_rot = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

            points.append([x_rot, y_rot, 0.1])

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

        angle_deg = -35
        angle_rad = math.radians(angle_deg)

        cx = sum(p[0] for p in self.best_weight_path) / len(self.best_weight_path)
        cy = sum(p[1] for p in self.best_weight_path) / len(self.best_weight_path)

        rotated_path = []

        for p in self.best_weight_path:

            dx = p[0] - cx
            dy = p[1] - cy

            x_rot = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y_rot = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

            rotated_path.append([
                float(x_rot),
                float(y_rot),
                float(p[2])
            ])

        cloud_msg = self.create_point_cloud(
            rotated_path,
            header,
            0,
            0,
            255
        )

        self.best_weight_path_pub.publish(cloud_msg)

    #Função chamada sempre que se recebe uma mensagem no tópico da posição estimada pelo amcl
    def amcl_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        self.amcl_path_points.append([float(x), float(y)])

    def rotate_point_about_center(self, x, y, cx, cy, angle_rad):
        dx = x - cx
        dy = y - cy

        xr = cx + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        yr = cy + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

        return xr, yr


    def compute_and_publish_error(self):
        if len(self.best_weight_path) < 2 or len(self.amcl_path_points) < 2:
            return

        angle_deg = -35
        angle_rad = math.radians(angle_deg)

        cx = sum(p[0] for p in self.best_weight_path) / len(self.best_weight_path)
        cy = sum(p[1] for p in self.best_weight_path) / len(self.best_weight_path)

        n = min(len(self.best_weight_path), len(self.amcl_path_points))

        errors = []

        for i in range(n):
            sx, sy, _ = self.best_weight_path[i]
            gx, gy = self.amcl_path_points[i]

            sx_rot, sy_rot = self.rotate_point_about_center(
                sx,
                sy,
                cx,
                cy,
                angle_rad
            )

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