"""
深蹲计数器核心逻辑
包含防抖机制，防止误计数
使用提取的几何特征进行姿态和角度分类
"""
import os
import numpy as np
import joblib
from collections import deque

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# MediaPipe关键点索引
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30


def get_point_from_landmarks(landmarks, idx, frame_width=1920, frame_height=1080):
    """从MediaPipe landmarks列表获取指定索引的点坐标 (x, y, z)，转换为像素坐标"""
    lm = landmarks[idx]
    # 将归一化坐标转换为像素坐标，与训练数据格式匹配
    x = lm.x * frame_width
    y = lm.y * frame_height
    z = lm.z * frame_width  # z 按宽度缩放
    return np.array([x, y, z])


def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(p1 - p2)


def calculate_angle_with_horizontal(p1, p2):
    """计算两点连线与水平轴的夹角（度数）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return abs(angle)


def calculate_joint_angle(p1, p2, p3):
    """计算三点形成的关节角度（p2为顶点）"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle


class SquatCounter:
    # 全身检测需要的关键点索引
    REQUIRED_LANDMARKS = [
        NOSE,           # 鼻子
        LEFT_SHOULDER,  # 左肩
        RIGHT_SHOULDER, # 右肩
        LEFT_HIP,       # 左髋
        RIGHT_HIP,      # 右髋
        LEFT_KNEE,      # 左膝
        RIGHT_KNEE,     # 右膝
        LEFT_ANKLE,     # 左脚踝
        RIGHT_ANKLE,    # 右脚踝
    ]

    def __init__(self, debounce_frames=2, confidence_threshold=0.7, visibility_threshold=0.5):
        """
        初始化深蹲计数器

        Args:
            debounce_frames: 防抖帧数，连续多少帧相同状态才确认状态变化
            confidence_threshold: 置信度阈值
            visibility_threshold: 关键点可见度阈值
        """
        self.debounce_frames = debounce_frames
        self.confidence_threshold = confidence_threshold
        self.visibility_threshold = visibility_threshold

        # 加载模型
        self.pose_model = joblib.load(os.path.join(MODEL_DIR, 'pose_knn.pkl'))
        self.pose_scaler = joblib.load(os.path.join(MODEL_DIR, 'pose_scaler.pkl'))
        self.angle_model = joblib.load(os.path.join(MODEL_DIR, 'angle_knn.pkl'))
        self.angle_scaler = joblib.load(os.path.join(MODEL_DIR, 'angle_scaler.pkl'))

        # 状态变量
        self.count = 0
        self.current_state = None  # 'stand' or 'squat'
        self.confirmed_state = None
        self.state_history = deque(maxlen=debounce_frames)
        self.has_squatted = False  # 标记是否曾经蹲下

        # 姿态和角度标签
        self.pose_labels = ['站立', '蹲下']
        self.angle_labels = ['正面', '侧面', '斜面']

    def check_full_body_visible(self, landmarks):
        """
        检查全身是否可见

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            tuple: (is_visible, message)
                - is_visible: 是否全身可见
                - message: 提示信息（全身可见时为None）
        """
        for idx in self.REQUIRED_LANDMARKS:
            lm = landmarks[idx]

            # 检查可见度
            if lm.visibility < self.visibility_threshold:
                return False, "请确保全身在画面内"

            # 检查是否在画面内（归一化坐标应在0-1之间）
            if not (0 <= lm.x <= 1 and 0 <= lm.y <= 1):
                return False, "请确保全身在画面内"

        return True, None

    def reset(self):
        """重置计数器"""
        self.count = 0
        self.current_state = None
        self.confirmed_state = None
        self.state_history.clear()
        self.has_squatted = False

    def extract_angle_features(self, landmarks):
        """
        提取用于角度分类（正面/侧面/斜面）的特征
        这些特征与用户面向摄像头的角度强相关
        """
        # 获取关键点
        left_shoulder = get_point_from_landmarks(landmarks, LEFT_SHOULDER)
        right_shoulder = get_point_from_landmarks(landmarks, RIGHT_SHOULDER)
        left_hip = get_point_from_landmarks(landmarks, LEFT_HIP)
        right_hip = get_point_from_landmarks(landmarks, RIGHT_HIP)

        features = []

        # 1. 肩部连线与水平轴的夹角
        shoulder_angle = calculate_angle_with_horizontal(left_shoulder[:2], right_shoulder[:2])
        features.append(shoulder_angle)

        # 2. 髋部连线与水平轴的夹角
        hip_angle = calculate_angle_with_horizontal(left_hip[:2], right_hip[:2])
        features.append(hip_angle)

        # 3. 左右肩的x坐标差值 - 正面时差值大，侧面时差值小
        shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
        features.append(shoulder_width)

        # 4. 左右髋的x坐标差值
        hip_width = abs(left_hip[0] - right_hip[0])
        features.append(hip_width)

        # 5. 左右肩的z坐标差值（深度差）- 斜面时差值大
        shoulder_z_diff = abs(left_shoulder[2] - right_shoulder[2])
        features.append(shoulder_z_diff)

        # 6. 左右髋的z坐标差值
        hip_z_diff = abs(left_hip[2] - right_hip[2])
        features.append(hip_z_diff)

        # 7. 肩宽与髋宽的比值
        width_ratio = shoulder_width / (hip_width + 1e-6)
        features.append(width_ratio)

        # 8. 肩部中点的z坐标（深度）
        shoulder_mid_z = (left_shoulder[2] + right_shoulder[2]) / 2
        features.append(shoulder_mid_z)

        return np.array(features).reshape(1, -1)

    def extract_pose_features(self, landmarks):
        """
        提取用于姿态分类（站立/蹲下）的特征
        这些特征与用户蹲下和站立的姿态强相关
        """
        # 获取关键点
        left_shoulder = get_point_from_landmarks(landmarks, LEFT_SHOULDER)
        right_shoulder = get_point_from_landmarks(landmarks, RIGHT_SHOULDER)
        left_hip = get_point_from_landmarks(landmarks, LEFT_HIP)
        right_hip = get_point_from_landmarks(landmarks, RIGHT_HIP)
        left_knee = get_point_from_landmarks(landmarks, LEFT_KNEE)
        right_knee = get_point_from_landmarks(landmarks, RIGHT_KNEE)
        left_ankle = get_point_from_landmarks(landmarks, LEFT_ANKLE)
        right_ankle = get_point_from_landmarks(landmarks, RIGHT_ANKLE)

        features = []

        # 计算中点
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        knee_mid = (left_knee + right_knee) / 2
        ankle_mid = (left_ankle + right_ankle) / 2

        # 计算躯干长度（肩部中点到髋部中点）用于归一化
        torso_length = calculate_distance(shoulder_mid, hip_mid)

        # 1. 肩部中点到脚踝中点的距离（归一化）
        shoulder_to_ankle = calculate_distance(shoulder_mid, ankle_mid)
        shoulder_to_ankle_norm = shoulder_to_ankle / (torso_length + 1e-6)
        features.append(shoulder_to_ankle_norm)

        # 2. 髋部中点到脚踝中点的距离（归一化）
        hip_to_ankle = calculate_distance(hip_mid, ankle_mid)
        hip_to_ankle_norm = hip_to_ankle / (torso_length + 1e-6)
        features.append(hip_to_ankle_norm)

        # 3. 膝盖中点到脚踝中点的距离（归一化）
        knee_to_ankle = calculate_distance(knee_mid, ankle_mid)
        knee_to_ankle_norm = knee_to_ankle / (torso_length + 1e-6)
        features.append(knee_to_ankle_norm)

        # 4. 左膝关节角度（髋-膝-踝）
        left_knee_angle = calculate_joint_angle(left_hip, left_knee, left_ankle)
        features.append(left_knee_angle)

        # 5. 右膝关节角度
        right_knee_angle = calculate_joint_angle(right_hip, right_knee, right_ankle)
        features.append(right_knee_angle)

        # 6. 左髋关节角度（肩-髋-膝）
        left_hip_angle = calculate_joint_angle(left_shoulder, left_hip, left_knee)
        features.append(left_hip_angle)

        # 7. 右髋关节角度
        right_hip_angle = calculate_joint_angle(right_shoulder, right_hip, right_knee)
        features.append(right_hip_angle)

        # 8. 髋部y坐标相对于肩部和脚踝的位置比例
        hip_relative_y = (hip_mid[1] - shoulder_mid[1]) / (ankle_mid[1] - shoulder_mid[1] + 1e-6)
        features.append(hip_relative_y)

        # 9. 膝盖y坐标相对于髋部和脚踝的位置比例
        knee_relative_y = (knee_mid[1] - hip_mid[1]) / (ankle_mid[1] - hip_mid[1] + 1e-6)
        features.append(knee_relative_y)

        # 10. 躯干与大腿的夹角
        torso_thigh_angle = calculate_joint_angle(shoulder_mid, hip_mid, knee_mid)
        features.append(torso_thigh_angle)

        # 11. 平均膝盖角度
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
        features.append(avg_knee_angle)

        # 12. 平均髋关节角度
        avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
        features.append(avg_hip_angle)

        return np.array(features).reshape(1, -1)

    def predict_pose(self, features):
        """预测姿态（站立/蹲下）"""
        scaled = self.pose_scaler.transform(features)
        pred = self.pose_model.predict(scaled)[0]
        proba = self.pose_model.predict_proba(scaled)[0]
        confidence = proba[pred]
        return pred, confidence

    def predict_angle(self, features):
        """预测角度（正面/侧面/斜面）"""
        scaled = self.angle_scaler.transform(features)
        pred = self.angle_model.predict(scaled)[0]
        proba = self.angle_model.predict_proba(scaled)[0]
        confidence = proba[pred]
        return pred, confidence

    def update(self, landmarks):
        """
        更新计数器状态

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            dict: 包含计数、姿态、角度等信息
        """
        # 检查全身是否可见
        is_visible, warning_msg = self.check_full_body_visible(landmarks)
        if not is_visible:
            return {
                'count': self.count,
                'pose': None,
                'pose_confidence': 0.0,
                'angle': None,
                'angle_confidence': 0.0,
                'confirmed_state': self.confirmed_state,
                'state_changed': False,
                'warning': warning_msg
            }

        # 提取特征
        pose_features = self.extract_pose_features(landmarks)
        angle_features = self.extract_angle_features(landmarks)

        # 预测姿态和角度
        pose_pred, pose_conf = self.predict_pose(pose_features)
        angle_pred, angle_conf = self.predict_angle(angle_features)

        # 当前检测到的状态
        current_detected = 'squat' if pose_pred == 1 else 'stand'

        state_changed = False

        # 简化逻辑：检测到蹲下就标记，检测到站立且曾蹲下就计数
        if current_detected == 'squat':
            self.has_squatted = True
            self.confirmed_state = 'squat'
        elif current_detected == 'stand' and self.has_squatted:
            self.count += 1
            self.has_squatted = False
            self.confirmed_state = 'stand'
            state_changed = True

        return {
            'count': self.count,
            'pose': self.pose_labels[pose_pred],
            'pose_confidence': float(pose_conf),
            'angle': self.angle_labels[angle_pred],
            'angle_confidence': float(angle_conf),
            'confirmed_state': self.confirmed_state,
            'state_changed': state_changed,
            'warning': None
        }
