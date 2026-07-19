#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import glob
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

from sensor_msgs_py import point_cloud2


def yaw_to_quaternion(yaw: float):
    """仅绕 Z 轴旋转."""
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def make_header(frame_id: str, stamp):
    h = Header()
    h.frame_id = frame_id
    h.stamp = stamp
    return h


def create_pointcloud2_xyz(points_xyz: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    points_xyz: [N,3] float32/float64
    """
    if points_xyz is None or len(points_xyz) == 0:
        points_xyz = np.zeros((0, 3), dtype=np.float32)

    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    header = make_header(frame_id, stamp)
    return point_cloud2.create_cloud(header, fields, points_xyz.tolist())


def create_pointcloud2_xyzi(points_xyzi: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """
    points_xyzi: [N,4] -> x,y,z,intensity
    """
    if points_xyzi is None or len(points_xyzi) == 0:
        points_xyzi = np.zeros((0, 4), dtype=np.float32)

    points_xyzi = np.asarray(points_xyzi, dtype=np.float32)

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    header = make_header(frame_id, stamp)
    return point_cloud2.create_cloud(header, fields, points_xyzi.tolist())


def rotation_z(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def box_corners_3d(box: np.ndarray) -> np.ndarray:
    """
    box: [x, y, z, dx, dy, dz, yaw]
    返回 8 个角点 [8,3]
    约定 z 为中心点高度中心
    """
    x, y, z, dx, dy, dz, yaw = box[:7]

    # 局部坐标系角点
    local = np.array([
        [ dx/2,  dy/2, -dz/2],
        [ dx/2, -dy/2, -dz/2],
        [-dx/2, -dy/2, -dz/2],
        [-dx/2,  dy/2, -dz/2],
        [ dx/2,  dy/2,  dz/2],
        [ dx/2, -dy/2,  dz/2],
        [-dx/2, -dy/2,  dz/2],
        [-dx/2,  dy/2,  dz/2],
    ], dtype=np.float32)

    R = rotation_z(yaw)
    corners = (R @ local.T).T
    corners += np.array([x, y, z], dtype=np.float32)
    return corners


def create_box_marker(box: np.ndarray,
                      marker_id: int,
                      frame_id: str,
                      stamp,
                      ns: str = "pred_boxes",
                      color=(1.0, 0.0, 0.0, 1.0),
                      line_width: float = 0.08,
                      lifetime_sec: float = 0.2,
                      text: Optional[str] = None) -> List[Marker]:
    """
    用 LINE_LIST 画 3D 框，可选 TEXT 显示 score/name
    """
    corners = box_corners_3d(box)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),   # top
        (0, 4), (1, 5), (2, 6), (3, 7)    # vertical
    ]

    marker = Marker()
    marker.header = make_header(frame_id, stamp)
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = line_width
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    marker.lifetime = Duration(sec=int(lifetime_sec),
                               nanosec=int((lifetime_sec - int(lifetime_sec)) * 1e9))

    for i, j in edges:
        p1 = Point()
        p1.x, p1.y, p1.z = corners[i].tolist()
        p2 = Point()
        p2.x, p2.y, p2.z = corners[j].tolist()
        marker.points.append(p1)
        marker.points.append(p2)

    markers = [marker]

    if text is not None:
        txt = Marker()
        txt.header = make_header(frame_id, stamp)
        txt.ns = ns + "_text"
        txt.id = marker_id
        txt.type = Marker.TEXT_VIEW_FACING
        txt.action = Marker.ADD
        txt.pose.position.x = float(box[0])
        txt.pose.position.y = float(box[1])
        txt.pose.position.z = float(box[2] + box[5] / 2.0 + 0.5)
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.6
        txt.color.r = 1.0
        txt.color.g = 1.0
        txt.color.b = 0.0
        txt.color.a = 1.0
        txt.text = text
        txt.lifetime = marker.lifetime
        markers.append(txt)

    return markers


def apply_transform_points(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    points_xyz: [N,3]
    T: [4,4]
    """
    if points_xyz is None or len(points_xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([points_xyz[:, :3], ones], axis=1)
    out = (T @ pts_h.T).T
    return out[:, :3]


def transform_box(box: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    将 box 从源坐标系变到目标坐标系。
    简化假设：只考虑绕 z 的旋转 + 平移。
    若 T 存在复杂俯仰/横滚，此方法需改为更严格版本。
    """
    out = box.copy()
    center = np.array([[box[0], box[1], box[2]]], dtype=np.float32)
    center_t = apply_transform_points(center, T)[0]
    out[0:3] = center_t

    # 从 T 提取 z 轴偏航角
    yaw_offset = math.atan2(T[1, 0], T[0, 0])
    out[6] = box[6] + yaw_offset
    return out


class AstyxRvizPublisher(Node):
    def __init__(self):
        super().__init__('astyx_rviz_publisher')

        # ===== 参数 =====
        self.declare_parameter('result_pkl',
            r'E:\Work\FT\RadarPillar\output\astyx_models\astyx_radarpillar\default\eval\epoch_160\val\default\result.pkl')
        self.declare_parameter('data_root',
            r'E:\Work\FT\RadarPillar\data\astyx')
        self.declare_parameter('fixed_frame', 'lidar')
        self.declare_parameter('rate', 2.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('use_radar_to_lidar_tf', True)

        self.result_pkl = self.get_parameter('result_pkl').value
        self.data_root = self.get_parameter('data_root').value
        self.fixed_frame = self.get_parameter('fixed_frame').value
        self.rate = float(self.get_parameter('rate').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.use_radar_to_lidar_tf = bool(self.get_parameter('use_radar_to_lidar_tf').value)

        # ===== 发布器 =====
        self.pub_radar = self.create_publisher(PointCloud2, '/astyx/radar_points', 10)
        self.pub_lidar = self.create_publisher(PointCloud2, '/astyx/lidar_points', 10)
        self.pub_boxes = self.create_publisher(MarkerArray, '/astyx/pred_boxes', 10)

        # ===== 载入检测结果 =====
        self.det_results = self.load_result_pkl(self.result_pkl)
        self.get_logger().info(f'Loaded {len(self.det_results)} frames from result.pkl')

        self.index = 0
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

    def load_result_pkl(self, path: str) -> List[dict]:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise RuntimeError(f'result.pkl 顶层不是 list，而是: {type(data)}')
        return data

    def on_timer(self):
        if len(self.det_results) == 0:
            return

        if self.index >= len(self.det_results):
            if self.loop:
                self.index = 0
            else:
                self.get_logger().info('Playback finished.')
                return

        det = self.det_results[self.index]
        stamp = self.get_clock().now().to_msg()

        frame_id = self.extract_frame_id(det)
        self.get_logger().info(f'Publishing frame: {frame_id}')

        # 1) 读取预测框
        boxes, scores, labels, names = self.extract_boxes(det)

        # 2) 读取对应帧点云
        radar_points, lidar_points = self.load_frame_points(frame_id)

        # 3) 读取标定
        T_radar_to_lidar = self.load_calibration_for_frame(frame_id)

        # 4) 坐标统一到 fixed_frame
        # 默认 fixed_frame = lidar
        if self.fixed_frame == 'lidar':
            # lidar 原样
            lidar_xyz = lidar_points[:, :3] if lidar_points is not None and len(lidar_points) > 0 else np.zeros((0, 3), dtype=np.float32)

            # radar 转到 lidar
            if radar_points is not None and len(radar_points) > 0:
                radar_xyz = radar_points[:, :3]
                if self.use_radar_to_lidar_tf and T_radar_to_lidar is not None:
                    radar_xyz = apply_transform_points(radar_xyz, T_radar_to_lidar)
                radar_pub = radar_xyz
            else:
                radar_pub = np.zeros((0, 3), dtype=np.float32)

            # 假设 boxes 已经在 lidar 坐标系
            # 如果你确认 boxes 在 radar 坐标系，则在这里 transform_box
            boxes_pub = []
            for b in boxes:
                boxes_pub.append(b.copy())
            boxes_pub = np.asarray(boxes_pub, dtype=np.float32) if len(boxes_pub) > 0 else np.zeros((0, 7), dtype=np.float32)

        else:
            self.get_logger().warn(f'当前脚本仅完整实现 fixed_frame=lidar，当前={self.fixed_frame}')
            lidar_xyz = lidar_points[:, :3] if lidar_points is not None and len(lidar_points) > 0 else np.zeros((0, 3), dtype=np.float32)
            radar_pub = radar_points[:, :3] if radar_points is not None and len(radar_points) > 0 else np.zeros((0, 3), dtype=np.float32)
            boxes_pub = boxes

        # 5) 发布点云
        msg_radar = create_pointcloud2_xyz(radar_pub, self.fixed_frame, stamp)
        msg_lidar = create_pointcloud2_xyz(lidar_xyz, self.fixed_frame, stamp)
        self.pub_radar.publish(msg_radar)
        self.pub_lidar.publish(msg_lidar)

        # 6) 发布框
        marr = MarkerArray()

        # 先 DELETEALL，避免残影
        delete_all = Marker()
        delete_all.header = make_header(self.fixed_frame, stamp)
        delete_all.action = Marker.DELETEALL
        marr.markers.append(delete_all)

        for i, box in enumerate(boxes_pub):
            score = scores[i] if i < len(scores) else None
            name = names[i] if i < len(names) else f'obj_{i}'

            txt = name
            if score is not None:
                txt += f' {score:.2f}'

            color = (1.0, 0.0, 0.0, 1.0)  # 默认红色
            if name.lower().startswith('car'):
                color = (0.0, 1.0, 0.0, 1.0)
            elif name.lower().startswith('ped'):
                color = (1.0, 1.0, 0.0, 1.0)

            markers = create_box_marker(
                box=box,
                marker_id=i,
                frame_id=self.fixed_frame,
                stamp=stamp,
                ns='pred_boxes',
                color=color,
                line_width=0.06,
                lifetime_sec=1.0 / self.rate + 0.1,
                text=txt
            )
            marr.markers.extend(markers)

        self.pub_boxes.publish(marr)
        self.index += 1

    def extract_frame_id(self, det: dict) -> str:
        candidates = ['frame_id', 'sample_idx', 'idx', 'image_idx']
        for k in candidates:
            if k in det:
                return str(det[k])

        # 找不到就用索引占位
        return f'{self.index:06d}'

    def extract_boxes(self, det: dict):
        """
        尽可能兼容不同 pkl 结构
        返回:
            boxes [N,7]
            scores [N]
            labels [N]
            names  [N]
        """
        boxes = None
        for k in ['boxes_lidar', 'pred_boxes', 'boxes', 'box3d_lidar']:
            if k in det:
                boxes = np.asarray(det[k], dtype=np.float32)
                break

        if boxes is None:
            boxes = np.zeros((0, 7), dtype=np.float32)

        scores = None
        for k in ['score', 'scores', 'pred_scores']:
            if k in det:
                scores = np.asarray(det[k], dtype=np.float32)
                break
        if scores is None:
            scores = np.zeros((len(boxes),), dtype=np.float32)

        labels = None
        for k in ['label_preds', 'pred_labels', 'labels']:
            if k in det:
                labels = np.asarray(det[k])
                break
        if labels is None:
            labels = np.zeros((len(boxes),), dtype=np.int32)

        names = None
        for k in ['name', 'names', 'pred_names']:
            if k in det:
                names = [str(x) for x in det[k]]
                break

        if names is None:
            # 尝试从 labels 映射
            label_map = {
                1: 'Car',
                2: 'Pedestrian'
            }
            names = [label_map.get(int(x), f'Class{x}') for x in labels]

        return boxes, scores, labels, names

    def load_frame_points(self, frame_id: str):
        """
        读取对应帧 radar / lidar 点云
        这里写成“搜索式”，适配你本地 Astyx 文件结构。
        你后续只需把文件模式补准即可。
        """
        radar_points = self.find_and_load_points(frame_id, sensor_type='radar')
        lidar_points = self.find_and_load_points(frame_id, sensor_type='lidar')
        return radar_points, lidar_points

    def find_and_load_points(self, frame_id: str, sensor_type: str = 'radar') -> np.ndarray:
        """
        递归查找包含 frame_id 的点云文件。
        """
        patterns = []
        if sensor_type == 'radar':
            patterns = [
                os.path.join(self.data_root, '**', f'*{frame_id}*radar*.txt'),
                os.path.join(self.data_root, '**', f'*{frame_id}*radar*.csv'),
                os.path.join(self.data_root, '**', f'*{frame_id}*radar*.bin'),
                os.path.join(self.data_root, '**', f'*{frame_id}*radar*.pcd'),
                os.path.join(self.data_root, '**', f'*{frame_id}*.txt'),
                os.path.join(self.data_root, '**', f'*{frame_id}*.csv'),
            ]
        else:
            patterns = [
                os.path.join(self.data_root, '**', f'*{frame_id}*lidar*.txt'),
                os.path.join(self.data_root, '**', f'*{frame_id}*lidar*.csv'),
                os.path.join(self.data_root, '**', f'*{frame_id}*lidar*.bin'),
                os.path.join(self.data_root, '**', f'*{frame_id}*lidar*.pcd'),
                os.path.join(self.data_root, '**', f'*{frame_id}*.txt'),
                os.path.join(self.data_root, '**', f'*{frame_id}*.bin'),
            ]

        matches = []
        for p in patterns:
            matches.extend(glob.glob(p, recursive=True))

        # 去重
        matches = sorted(list(set(matches)))

        if len(matches) == 0:
            self.get_logger().warn(f'未找到 {sensor_type} 点云文件 for frame_id={frame_id}')
            return np.zeros((0, 4), dtype=np.float32)

        chosen = self.select_best_match(matches, sensor_type)
        self.get_logger().info(f'{sensor_type} file: {chosen}')
        return self.load_points_from_file(chosen)

    def select_best_match(self, matches: List[str], sensor_type: str) -> str:
        # 简单启发式
        for m in matches:
            low = m.lower()
            if sensor_type in low:
                return m
        return matches[0]

    def load_points_from_file(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == '.bin':
                arr = np.fromfile(path, dtype=np.float32)
                # 猜测列数
                if arr.size % 4 == 0:
                    return arr.reshape(-1, 4)
                elif arr.size % 5 == 0:
                    return arr.reshape(-1, 5)
                elif arr.size % 3 == 0:
                    return arr.reshape(-1, 3)
                else:
                    self.get_logger().warn(f'无法判断 bin 点云列数: {path}')
                    return np.zeros((0, 4), dtype=np.float32)

            elif ext in ['.txt', '.csv']:
                # 自动尝试不同分隔符
                for delim in [None, ',', ';', '\t', ' ']:
                    try:
                        arr = np.loadtxt(path, delimiter=delim, dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        return arr
                    except Exception:
                        continue
                self.get_logger().warn(f'文本点云读取失败: {path}')
                return np.zeros((0, 4), dtype=np.float32)

            elif ext == '.pcd':
                # 如未安装 open3d / pypcd，这里先给出占位
                self.get_logger().warn('暂未实现 .pcd 读取，请改成 txt/bin 或扩展此函数')
                return np.zeros((0, 4), dtype=np.float32)

            else:
                self.get_logger().warn(f'未知点云格式: {path}')
                return np.zeros((0, 4), dtype=np.float32)

        except Exception as e:
            self.get_logger().error(f'读取点云失败 {path}: {e}')
            return np.zeros((0, 4), dtype=np.float32)

    def load_calibration_for_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """
        加载 Astyx 对应帧标定，返回 T_radar_to_lidar [4,4]
        这里需要你根据 Astyx 实际标定文件格式补充。
        当前给一个单位阵占位。
        """
        # TODO: 你需要在这里按 Astyx 标定文件真实格式读取
        # 例如在 data_root 下搜索 calibration json/txt
        # 然后解析出 radar->lidar 4x4 矩阵
        T = np.eye(4, dtype=np.float32)
        return T


def main(args=None):
    rclpy.init(args=args)
    node = AstyxRvizPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()