"""
ROS2 RViz Visualizer for RadarPillar (Astyx Dataset)
功能：
1. 直接读取 result.pkl 预测结果
2. 读取 Astyx 数据集的 GT 标签和雷达点云
3. 同时发布 GT 框、预测框、雷达点云到 RViz
"""
import os
import sys
import math
import json
import pickle
import rclpy
from rclpy.node import Node
from pathlib import Path
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np

# ===================== 【关键】复用原脚本的路径定义 =====================
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / 'data' / 'astyx'

# ===================== 导入原脚本的 BEV 工具类（用于颜色映射） =====================
sys.path.insert(0, str(ROOT / 'tools'))
try:
    from visualize_bev import CLASS_COLORS_GT, CLASS_COLORS_PRED
except ImportError:
    # 如果导入失败，使用默认颜色
    CLASS_COLORS_GT = {'Car': '#2ecc71', 'Pedestrian': '#3498db', 'Cyclist': '#e74c3c'}
    CLASS_COLORS_PRED = {'Car': '#27ae60', 'Pedestrian': '#2980b9', 'Cyclist': '#c0392b'}


class AstyxRvizVisualizer(Node):
    def __init__(self):
        super().__init__('astyx_rviz_visualizer')

        # --- 可配置参数 ---
        self.declare_parameter('pkl_path', str(
            ROOT / 'output' / 'astyx_models' / 'astyx_radarpillar' /
            'default' / 'eval' / 'epoch_160' / 'val' / 'default' / 'result.pkl'
        ))
        self.declare_parameter('sample_ids', ['000273', '000217', '000079'])
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('score_thresh', 0.3)
        self.declare_parameter('publish_rate', 2.0)  # 每秒发布几帧

        # --- 坐标轴投影 旋转参数（角度） ---
        self.declare_parameter('axis_roll_deg', 0.0)   # 绕 X 轴滚转（度）
        self.declare_parameter('axis_pitch_deg', 0.0)  # 绕 Y 轴俯仰（度）
        self.declare_parameter('axis_yaw_deg', 0.0)    # 绕 Z 轴偏航（度）
        self.declare_parameter('axis_length', 1.5)     # 轴箭头长度（米）
        self.declare_parameter('axis_width', 0.05)     # 轴箭头线宽（米）
        self.declare_parameter('show_axes', True)      # 是否显示坐标轴

        # --- 全局坐标系参数 ---
        self.declare_parameter('global_axis_length', 3.0)  # 全局坐标系轴长度（米）
        self.declare_parameter('global_axis_width', 0.1)   # 全局坐标系轴线宽（米）
        self.declare_parameter('show_global_axes', True)   # 是否显示全局坐标系

        # --- 读取参数 ---
        self.pkl_path = Path(self.get_parameter('pkl_path').value)
        self.sample_ids = self.get_parameter('sample_ids').value
        self.frame_id = self.get_parameter('frame_id').value
        self.score_thresh = self.get_parameter('score_thresh').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # 读取角度参数并转换为弧度
        self.axis_roll = math.radians(self.get_parameter('axis_roll_deg').value)
        self.axis_pitch = math.radians(self.get_parameter('axis_pitch_deg').value)
        self.axis_yaw = math.radians(self.get_parameter('axis_yaw_deg').value)
        self.axis_length = self.get_parameter('axis_length').value
        self.axis_width = self.get_parameter('axis_width').value
        self.show_axes = self.get_parameter('show_axes').value

        # 全局坐标系参数
        self.global_axis_length = self.get_parameter('global_axis_length').value
        self.global_axis_width = self.get_parameter('global_axis_width').value
        self.show_global_axes = self.get_parameter('show_global_axes').value

        # --- 加载数据 ---
        self.results = self._load_pkl(self.pkl_path)
        self.current_idx = 0

        # --- ROS2 发布器 ---
        self.gt_marker_pub = self.create_publisher(MarkerArray, '/astyx/gt_boxes', 10)
        self.pred_marker_pub = self.create_publisher(MarkerArray, '/astyx/pred_boxes', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/astyx/radar_points', 10)
        self.global_axes_pub = self.create_publisher(MarkerArray, '/astyx/global_axes', 10)

        # --- 定时器 ---
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.get_logger().info(f"✅ Astyx RViz Visualizer 已启动！")
        self.get_logger().info(f"📦 PKL路径: {self.pkl_path}")
        self.get_logger().info(f"🎯 样本列表: {self.sample_ids}")

    def _load_pkl(self, pkl_path):
        """参考原脚本的 PKL 加载逻辑"""
        if not pkl_path.exists():
            self.get_logger().error(f"❌ PKL 文件不存在: {pkl_path}")
            return []
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    def _hex_to_rgb(self, hex_color):
        """将十六进制颜色转为 RGB (0-1)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _create_box_marker(self, header, obj, is_gt, marker_id):
        """创建 3D 检测框 Marker (LINE_LIST)"""
        marker = Marker()
        marker.header = header
        marker.ns = "gt_boxes" if is_gt else "pred_boxes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # 线宽

        # 颜色
        color_map = CLASS_COLORS_GT if is_gt else CLASS_COLORS_PRED
        r, g, b = self._hex_to_rgb(color_map.get(obj['class'], '#ffffff'))
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 0.9

        # 获取框参数
        x, y, z = obj['x'], obj['y'], obj['z']
        l, w, h = obj['l'], obj['w'], obj['h']
        heading = obj['heading']

        # 计算 8 个顶点
        cos_r = math.cos(heading)
        sin_r = math.sin(heading)
        corners = [
            [-l/2, -w/2, 0], [ l/2, -w/2, 0], [ l/2,  w/2, 0], [-l/2,  w/2, 0],
            [-l/2, -w/2, h], [ l/2, -w/2, h], [ l/2,  w/2, h], [-l/2,  w/2, h],
        ]
        # 旋转 + 平移
        from geometry_msgs.msg import Point
        world_corners = []
        for c in corners:
            px = c[0] * cos_r - c[1] * sin_r + x
            py = c[0] * sin_r + c[1] * cos_r + y
            pz = c[2] + z
            world_corners.append(Point(x=px, y=py, z=pz))

        # 12 条边
        edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
        for e in edges:
            marker.points.append(world_corners[e[0]])
            marker.points.append(world_corners[e[1]])

        return marker

    def _create_axis_markers(self, header, obj, is_gt, base_id):
        """
        为检测框中心生成 XYZ 坐标轴箭头 Marker。
        支持完整的 Roll-Pitch-Yaw 旋转控制（使用角度参数）。

        旋转顺序：ZYX (Yaw-Pitch-Roll)
          1. 先应用框自身的 heading（绕Z轴）
          2. 再叠加用户设定的 yaw（绕Z）、pitch（绕Y）、roll（绕X）

        参数（通过 ROS2 参数设置，单位：度）：
          axis_roll_deg  — 绕 X 轴滚转
          axis_pitch_deg — 绕 Y 轴俯仰
          axis_yaw_deg   — 绕 Z 轴偏航
        """
        from geometry_msgs.msg import Point

        cx, cy, cz = obj['x'], obj['y'], obj['z'] + obj['h'] / 2.0
        heading = obj['heading']

        # 总旋转角度（弧度）
        yaw_total = heading + self.axis_yaw
        pitch = self.axis_pitch
        roll = self.axis_roll

        # 构造 ZYX 欧拉角旋转矩阵
        cy, sy = math.cos(yaw_total), math.sin(yaw_total)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        # 完整旋转矩阵（3x3）
        r11 = cy * cp
        r12 = cy * sp * sr - sy * cr
        r13 = cy * sp * cr + sy * sr
        r21 = sy * cp
        r22 = sy * sp * sr + cy * cr
        r23 = sy * sp * cr - cy * sr
        r31 = -sp
        r32 = cp * sr
        r33 = cp * cr

        def rotate(dx, dy, dz):
            """应用旋转矩阵"""
            rx = r11 * dx + r12 * dy + r13 * dz
            ry = r21 * dx + r22 * dy + r23 * dz
            rz = r31 * dx + r32 * dy + r33 * dz
            return rx, ry, rz

        # 三轴方向向量（单位向量 × length）
        axes = [
            (self.axis_length, 0.0, 0.0, 1.0, 0.0, 0.0),  # X 轴 → 红
            (0.0, self.axis_length, 0.0, 0.0, 1.0, 0.0),  # Y 轴 → 绿
            (0.0, 0.0, self.axis_length, 0.0, 0.0, 1.0),  # Z 轴 → 蓝
        ]

        ns = ("gt_axes" if is_gt else "pred_axes")
        markers = []
        for idx, (dx, dy, dz, r, g, b) in enumerate(axes):
            m = Marker()
            m.header = header
            m.ns = ns
            m.id = base_id * 3 + idx
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = self.axis_width        # 轴线宽
            m.scale.y = self.axis_width * 2.0  # 箭头头宽
            m.scale.z = self.axis_width * 2.0  # 箭头头长
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.9

            ex, ey, ez = rotate(dx, dy, dz)
            m.points.append(Point(x=cx, y=cy, z=cz))
            m.points.append(Point(x=cx + ex, y=cy + ey, z=cz + ez))
            markers.append(m)

        return markers

    def _create_global_axes(self, header):
        """
        创建全局坐标系的 XYZ 轴 Marker（位于原点）。
        X 轴 → 红色，Y 轴 → 绿色，Z 轴 → 蓝色
        """
        from geometry_msgs.msg import Point

        markers = []
        axes = [
            (self.global_axis_length, 0.0, 0.0, 1.0, 0.0, 0.0, 'X'),  # X 轴 → 红
            (0.0, self.global_axis_length, 0.0, 0.0, 1.0, 0.0, 'Y'),  # Y 轴 → 绿
            (0.0, 0.0, self.global_axis_length, 0.0, 0.0, 1.0, 'Z'),  # Z 轴 → 蓝
        ]

        for idx, (dx, dy, dz, r, g, b, name) in enumerate(axes):
            m = Marker()
            m.header = header
            m.ns = "global_axes"
            m.id = idx
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = self.global_axis_width        # 轴线宽
            m.scale.y = self.global_axis_width * 2.0  # 箭头头宽
            m.scale.z = self.global_axis_width * 2.0  # 箭头头长
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0

            # 从原点指向轴方向
            m.points.append(Point(x=0.0, y=0.0, z=0.0))
            m.points.append(Point(x=dx, y=dy, z=dz))
            markers.append(m)

        return markers

    def _create_pointcloud_msg(self, header, points):
        """创建雷达点云 PointCloud2 消息"""
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_dense = False
        msg.is_bigendian = False

        # 字段定义: x, y, z, rcs
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width

        # 填充数据 (x, y, z, rcs)
        data = np.zeros((points.shape[0], 4), dtype=np.float32)
        data[:, 0] = points[:, 0]  # x
        data[:, 1] = points[:, 1]  # y
        data[:, 2] = points[:, 2] if points.shape[1] > 2 else 0.0  # z
        data[:, 3] = points[:, 3]  # rcs (intensity)
        msg.data = data.tobytes()

        return msg

    def timer_callback(self):
        """定时器回调：循环发布数据"""
        if len(self.results) == 0:
            return

        # 获取当前样本 ID
        sample_id = self.sample_ids[self.current_idx]
        self.get_logger().info(f"---------------------------")
        self.get_logger().info(f"🎬 正在可视化样本: {sample_id}")

        # 创建 Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        # ===================== 0. 发布全局坐标系 =====================
        if self.show_global_axes:
            global_axes_array = MarkerArray()
            global_axes_array.markers = self._create_global_axes(header)
            self.global_axes_pub.publish(global_axes_array)

        # ===================== 1. 加载雷达点云 (参考原脚本) =====================
        pc_file = DATA_ROOT / 'training' / 'radar_6455' / f'{sample_id}.txt'
        if pc_file.exists():
            points = np.loadtxt(str(pc_file), dtype=np.float32, skiprows=2)
            # 发布点云
            pc_msg = self._create_pointcloud_msg(header, points)
            self.pc_pub.publish(pc_msg)
            self.get_logger().info(f"✅ 雷达点云已发布: {points.shape[0]} 个点")
        else:
            self.get_logger().warn(f"⚠️ 点云文件未找到: {pc_file}")

        # ===================== 2. 加载 GT 标签 (参考原脚本) =====================
        gt_file = DATA_ROOT / 'training' / 'groundtruth_obj3d' / f'{sample_id}.json'
        gt_marker_array = MarkerArray()
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                for i, obj in enumerate(json_data['objects']):
                    classname = obj['classname']
                    w_q, x_q, y_q, z_q = obj['orientation_quat']
                    heading = math.atan2(2.0 * (w_q * z_q + x_q * y_q), 1.0 - 2.0 * (y_q * y_q + z_q * z_q))
                    
                    parsed_obj = {
                        'class': classname,
                        'x': obj['center3d'][0],
                        'y': obj['center3d'][1],
                        'z': obj['center3d'][2],
                        'l': obj['dimension3d'][0],
                        'w': obj['dimension3d'][1],
                        'h': obj['dimension3d'][2],
                        'heading': heading
                    }
                    marker = self._create_box_marker(header, parsed_obj, is_gt=True, marker_id=i)
                    gt_marker_array.markers.append(marker)
                    if self.show_axes:
                        gt_marker_array.markers.extend(
                            self._create_axis_markers(header, parsed_obj, is_gt=True, base_id=i)
                        )
            self.gt_marker_pub.publish(gt_marker_array)
            self.get_logger().info(f"✅ GT 框已发布: {len(gt_marker_array.markers)} 个")
        else:
            self.get_logger().warn(f"⚠️ GT 文件未找到: {gt_file}")

        # ===================== 3. 加载预测框 (参考原脚本) =====================
        pred_marker_array = MarkerArray()
        found_frame = False
        for det in self.results:
            if det['frame_id'] == sample_id:
                found_frame = True
                n = len(det['name'])
                marker_id = 0
                for i in range(n):
                    score = det['score'][i]
                    if score < self.score_thresh:
                        continue
                    
                    # 解析预测框
                    name = det['name'][i]
                    dims = det['dimensions'][i]  # h, w, l
                    loc = det['location'][i]    # x, y, z
                    ry = det['rotation_y'][i]
                    
                    # 转换为统一格式 (注意 Astyx 坐标系)
                    parsed_obj = {
                        'class': name,
                        'x': loc[0],
                        'y': loc[1],
                        'z': loc[2],
                        'l': dims[2],  # length
                        'w': dims[1],  # width
                        'h': dims[0],  # height
                        'heading': ry
                    }
                    marker = self._create_box_marker(header, parsed_obj, is_gt=False, marker_id=marker_id)
                    pred_marker_array.markers.append(marker)
                    if self.show_axes:
                        pred_marker_array.markers.extend(
                            self._create_axis_markers(header, parsed_obj, is_gt=False, base_id=marker_id)
                        )
                    marker_id += 1
                break
        
        if found_frame:
            self.pred_marker_pub.publish(pred_marker_array)
            self.get_logger().info(f"✅ 预测框已发布: {len(pred_marker_array.markers)} 个 (阈值: {self.score_thresh})")
        else:
            self.get_logger().warn(f"⚠️ 预测框未找到样本: {sample_id}")

        # 更新索引，循环播放
        self.current_idx = (self.current_idx + 1) % len(self.sample_ids)


def main():
    rclpy.init()
    node = AstyxRvizVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()