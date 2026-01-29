"""
深蹲计数器核心逻辑
包含防抖机制，防止误计数
使用提取的几何特征进行姿态和角度分类
包含 DTW 动态时间规整打分功能
"""
import os
import numpy as np
import pandas as pd
import joblib
from collections import deque
from scipy.spatial.distance import euclidean

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_csv', 'csv_standard')

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


def calculate_angle_xz_plane(p1, p2):
    """
    计算两点连线在 xz 平面（俯视图）与 x 轴的夹角（度数）
    用于判断身体朝向：正面接近0°，侧面接近90°，斜面在中间
    """
    dx = p2[0] - p1[0]
    dz = p2[2] - p1[2]
    angle = np.arctan2(abs(dz), abs(dx)) * 180 / np.pi
    return angle


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

        # DTW 打分相关
        self.standard_sequences = self._load_standard_sequences()
        self.frame_buffer = deque(maxlen=150)  # 缓存最近 150 帧（约 5 秒 @30fps）
        self.squat_start_idx = 0  # 蹲下开始的帧索引
        self.frame_count = 0  # 总帧计数

        # 打分结果
        self.current_score = 0.0
        self.score_history = []
        self.avg_score = 0.0

        # 低通滤波器参数
        self.filter_window = 5  # 移动平均窗口大小
        self.hip_y_history = deque(maxlen=self.filter_window)

        # 特征平滑滤波器（用于 DTW 评分）
        self.feature_filter_window = 3  # 特征平滑窗口
        self.feature_history = deque(maxlen=self.feature_filter_window)

        # 加载标准蹲下帧数据（用于矫正）
        self.standard_squat_frames = self._load_standard_squat_frames()
        self.current_corrections = []  # 当前矫正提示

    def _load_standard_squat_frames(self):
        """加载各角度标准蹲下帧的关键点数据，提取角度特定的矫正特征"""
        frames = {}
        file_map = {
            '正面': 'zheng_standard.csv',
            '侧面': 'ce_standard.csv',
            '斜面': 'xie_standard.csv'
        }

        for angle_name, filename in file_map.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    # 读取33个关键点的坐标
                    df = pd.read_csv(file_path, header=None)
                    landmarks_array = df.values  # 33x3 数组
                    # 根据角度类型提取不同的矫正特征
                    features = self._compute_correction_features(landmarks_array, angle_name)
                    frames[angle_name] = features
                    print(f"加载标准蹲下帧: {angle_name}, 特征: {features}")
                except Exception as e:
                    print(f"加载标准蹲下帧失败 {file_path}: {e}")

        return frames

    def _load_standard_sequences(self):
        """加载标准动作序列数据"""
        sequences = {}
        file_map = {
            '正面': 'zheng_squat_1.csv',
            '侧面': 'ce_squat_1.csv',
            '斜面': 'xie_squat_1.csv'
        }

        for angle_name, filename in file_map.items():
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    # 提取关键特征序列（膝关节角度、髋关节角度等）
                    sequences[angle_name] = self._extract_sequence_features(df)
                except Exception as e:
                    print(f"加载标准序列文件失败 {file_path}: {e}")

        return sequences

    def _extract_sequence_features(self, df):
        """从 DataFrame 提取用于 DTW 比较的特征序列"""
        features_list = []

        for _, row in df.iterrows():
            # 提取关键点坐标（跳过 Frame 列）
            coords = row.values[1:].astype(float)

            # 重构为 33x3 的数组
            landmarks_array = coords.reshape(33, 3)

            # 提取关键特征
            features = self._compute_frame_features(landmarks_array)
            features_list.append(features)

        return np.array(features_list)

    def _compute_frame_features(self, landmarks_array):
        """
        计算单帧的特征向量（用于 DTW）
        使用纯角度和比例特征，不受坐标系影响
        """
        # 获取关键点
        left_shoulder = landmarks_array[LEFT_SHOULDER]
        right_shoulder = landmarks_array[RIGHT_SHOULDER]
        left_hip = landmarks_array[LEFT_HIP]
        right_hip = landmarks_array[RIGHT_HIP]
        left_knee = landmarks_array[LEFT_KNEE]
        right_knee = landmarks_array[RIGHT_KNEE]
        left_ankle = landmarks_array[LEFT_ANKLE]
        right_ankle = landmarks_array[RIGHT_ANKLE]

        # 计算中点
        hip_mid = (left_hip + right_hip) / 2
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        ankle_mid = (left_ankle + right_ankle) / 2

        # 特征1: 左膝关节角度（归一化到0-1，180度=1）
        left_knee_angle = self._calc_angle(left_hip, left_knee, left_ankle) / 180.0

        # 特征2: 右膝关节角度
        right_knee_angle = self._calc_angle(right_hip, right_knee, right_ankle) / 180.0

        # 特征3: 左髋关节角度
        left_hip_angle = self._calc_angle(left_shoulder, left_hip, left_knee) / 180.0

        # 特征4: 右髋关节角度
        right_hip_angle = self._calc_angle(right_shoulder, right_hip, right_knee) / 180.0

        # 特征5: 髋部相对高度（髋部在肩部和脚踝之间的位置比例）
        # 站立时约0.4-0.5，蹲下时约0.6-0.8
        total_height = ankle_mid[1] - shoulder_mid[1]
        if abs(total_height) > 1e-6:
            hip_relative_height = (hip_mid[1] - shoulder_mid[1]) / total_height
        else:
            hip_relative_height = 0.5

        # 特征6: 躯干与大腿夹角（归一化）
        torso_thigh_angle = self._calc_angle(shoulder_mid, hip_mid, (left_knee + right_knee) / 2) / 180.0

        return np.array([
            left_knee_angle,
            right_knee_angle,
            left_hip_angle,
            right_hip_angle,
            hip_relative_height,
            torso_thigh_angle,
        ])

    def _compute_correction_features(self, landmarks_array, angle_type):
        """
        根据角度类型计算用于矫正的特征
        不同角度使用不同的特征集
        """
        # 获取关键点
        left_shoulder = landmarks_array[LEFT_SHOULDER]
        right_shoulder = landmarks_array[RIGHT_SHOULDER]
        left_hip = landmarks_array[LEFT_HIP]
        right_hip = landmarks_array[RIGHT_HIP]
        left_knee = landmarks_array[LEFT_KNEE]
        right_knee = landmarks_array[RIGHT_KNEE]
        left_ankle = landmarks_array[LEFT_ANKLE]
        right_ankle = landmarks_array[RIGHT_ANKLE]
        left_foot = landmarks_array[31]  # LEFT_FOOT_INDEX
        right_foot = landmarks_array[32]  # RIGHT_FOOT_INDEX

        # 计算中点
        hip_mid = (left_hip + right_hip) / 2
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        ankle_mid = (left_ankle + right_ankle) / 2
        knee_mid = (left_knee + right_knee) / 2
        foot_mid = (left_foot + right_foot) / 2

        # 计算归一化用的长度
        half_torso = np.linalg.norm(shoulder_mid - hip_mid)  # 半躯干长度
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)  # 肩宽

        # 获取鼻子位置
        nose = landmarks_array[NOSE]

        if angle_type == '正面':
            # 正面特征：
            # 1. 蹲深程度：臀部中点到两脚中点距离 / 鼻子到两脚中点距离
            hip_to_ankle_dist = np.linalg.norm(hip_mid - ankle_mid)
            nose_to_ankle_dist = np.linalg.norm(nose - ankle_mid)
            squat_depth = hip_to_ankle_dist / (nose_to_ankle_dist + 1e-6)

            # 2. 膝盖间距：两膝盖水平距离 / 肩宽
            knee_distance = abs(left_knee[0] - right_knee[0]) / (shoulder_width + 1e-6)

            # 3. 身体左右平衡：左右肩膀高度差 / 肩宽
            shoulder_balance = (left_shoulder[1] - right_shoulder[1]) / (shoulder_width + 1e-6)

            # 4. 重心位置：臀部中点x相对于两脚中点x的偏移 / 肩宽
            center_offset = (hip_mid[0] - ankle_mid[0]) / (shoulder_width + 1e-6)

            return {
                'squat_depth': squat_depth,
                'knee_distance': knee_distance,
                'shoulder_balance': shoulder_balance,
                'center_offset': center_offset
            }

        elif angle_type == '侧面':
            # 侧面特征：
            # 1. 膝盖弯曲角度（取可见侧的膝盖）
            left_knee_angle = self._calc_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = self._calc_angle(right_hip, right_knee, right_ankle)
            knee_angle = (left_knee_angle + right_knee_angle) / 2 / 180.0  # 归一化

            # 2. 背部前倾角度：肩部-髋部连线与垂直线的夹角
            torso_vector = shoulder_mid - hip_mid
            vertical = np.array([0, -1, 0])  # 向上的垂直向量
            cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            back_angle = np.arccos(cos_angle) * 180 / np.pi / 90.0  # 归一化到0-1（90度=1）

            # 3. 膝盖超脚尖程度：膝盖x - 脚尖x，归一化
            knee_x = (left_knee[0] + right_knee[0]) / 2
            foot_x = (left_foot[0] + right_foot[0]) / 2
            knee_over_toe = (knee_x - foot_x) / (half_torso + 1e-6)

            return {
                'knee_angle': knee_angle,
                'back_angle': back_angle,
                'knee_over_toe': knee_over_toe
            }

        else:  # 斜面
            # 斜面特征：综合正面和侧面
            # 1. 蹲深程度：臀部中点到两脚中点距离 / 鼻子到两脚中点距离
            hip_to_ankle_dist = np.linalg.norm(hip_mid - ankle_mid)
            nose_to_ankle_dist = np.linalg.norm(nose - ankle_mid)
            squat_depth = hip_to_ankle_dist / (nose_to_ankle_dist + 1e-6)

            # 2. 背部前倾角度
            torso_vector = shoulder_mid - hip_mid
            vertical = np.array([0, -1, 0])
            cos_angle = np.dot(torso_vector, vertical) / (np.linalg.norm(torso_vector) * np.linalg.norm(vertical) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            back_angle = np.arccos(cos_angle) * 180 / np.pi / 90.0

            return {
                'squat_depth': squat_depth,
                'back_angle': back_angle
            }

    def _smooth_features(self, features):
        """对特征进行平滑滤波，减少抖动"""
        self.feature_history.append(features)
        if len(self.feature_history) >= self.feature_filter_window:
            # 使用移动平均
            return np.mean(list(self.feature_history), axis=0)
        return features

    def _calc_angle(self, p1, p2, p3):
        """计算三点角度"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1, 1)
        return np.arccos(cos_angle) * 180 / np.pi

    def _dtw_distance(self, seq1, seq2):
        """计算两个序列的 DTW 距离"""
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean(seq1[i - 1], seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # 插入
                    dtw_matrix[i, j - 1],      # 删除
                    dtw_matrix[i - 1, j - 1]   # 匹配
                )

        return dtw_matrix[n, m]

    def _compute_dtw_score(self, user_sequence, angle_type):
        """计算 DTW 打分"""
        if angle_type not in self.standard_sequences:
            return 70.0  # 默认分数

        standard_seq = self.standard_sequences[angle_type]

        # 如果用户序列太短，返回 None 表示不计入评分
        if len(user_sequence) < 10:
            return None

        # 计算 DTW 距离
        dtw_dist = self._dtw_distance(user_sequence, standard_seq)

        # 使用两个序列的最大长度归一化，避免帧数差异影响评分
        max_len = max(len(user_sequence), len(standard_seq))
        normalized_dist = dtw_dist / max_len

        # 调试输出：查看用户序列的特征范围
        print(f"[DTW] 用户序列: {len(user_sequence)}帧, 标准序列: {len(standard_seq)}帧")

        # 打印用户序列的关键特征统计
        user_arr = np.array(user_sequence)
        print(f"[DTW] 用户特征 - 左膝角度: {user_arr[:,0].min()*180:.1f}°~{user_arr[:,0].max()*180:.1f}°")
        print(f"[DTW] 用户特征 - 髋部高度: {user_arr[:,4].min():.2f}~{user_arr[:,4].max():.2f}")

        print(f"[DTW] DTW距离: {dtw_dist:.2f}, 归一化距离: {normalized_dist:.4f}")

        # 映射到分数
        score = 100 - 15 * normalized_dist
        score = max(0, min(100, score))

        print(f"[DTW] 最终得分: {score:.1f}")
        print("-" * 50)

        return round(score, 1)

    def _get_filtered_hip_y(self, landmarks):
        """获取低通滤波后的髋部 y 坐标"""
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        hip_y = (left_hip.y + right_hip.y) / 2

        self.hip_y_history.append(hip_y)

        # 移动平均滤波
        if len(self.hip_y_history) >= self.filter_window:
            return sum(self.hip_y_history) / len(self.hip_y_history)
        return hip_y

    def _landmarks_to_array(self, landmarks):
        """将 MediaPipe landmarks 转换为数组格式"""
        arr = np.zeros((33, 3))
        for i in range(33):
            lm = landmarks[i]
            arr[i] = [lm.x * 1920, lm.y * 1080, lm.z * 1920]
        return arr

    def check_full_body_visible(self, landmarks):
        """
        检查全身是否可见（支持侧面站立）

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            tuple: (is_visible, message)
                - is_visible: 是否全身可见
                - message: 提示信息（全身可见时为None）
        """
        # 核心点：必须可见（鼻子用于判断头部）
        nose = landmarks[NOSE]
        if nose.visibility < 0.3 or not (0 <= nose.x <= 1 and 0 <= nose.y <= 1):
            return False, "请确保全身在画面内"

        # 成对的关键点：只要求至少一侧可见（支持侧面站立）
        paired_landmarks = [
            (LEFT_SHOULDER, RIGHT_SHOULDER, "肩部"),
            (LEFT_HIP, RIGHT_HIP, "髋部"),
            (LEFT_KNEE, RIGHT_KNEE, "膝盖"),
            (LEFT_ANKLE, RIGHT_ANKLE, "脚踝"),
        ]

        low_threshold = 0.3  # 侧面时降低阈值

        for left_idx, right_idx, name in paired_landmarks:
            left_lm = landmarks[left_idx]
            right_lm = landmarks[right_idx]

            # 检查是否至少有一侧可见
            left_visible = (left_lm.visibility >= low_threshold and
                           0 <= left_lm.x <= 1 and 0 <= left_lm.y <= 1)
            right_visible = (right_lm.visibility >= low_threshold and
                            0 <= right_lm.x <= 1 and 0 <= right_lm.y <= 1)

            if not (left_visible or right_visible):
                return False, "请确保全身在画面内"

        return True, None

    def reset(self):
        """重置计数器"""
        self.count = 0
        self.current_state = None
        self.confirmed_state = None
        self.state_history.clear()
        self.has_squatted = False
        self.frame_buffer.clear()
        self.squat_start_idx = 0
        self.frame_count = 0
        self.current_score = 0.0
        self.score_history = []
        self.avg_score = 0.0
        self.current_corrections = []
        self.hip_y_history.clear()
        self.feature_history.clear()

    def extract_angle_features(self, landmarks):
        """
        提取用于角度分类（正面/侧面/斜面）的特征
        只使用髋部和肩部连线在XZ平面与X轴的夹角
        正面接近0°，侧面接近90°，斜面在中间
        """
        # 获取关键点
        left_shoulder = get_point_from_landmarks(landmarks, LEFT_SHOULDER)
        right_shoulder = get_point_from_landmarks(landmarks, RIGHT_SHOULDER)
        left_hip = get_point_from_landmarks(landmarks, LEFT_HIP)
        right_hip = get_point_from_landmarks(landmarks, RIGHT_HIP)

        features = []

        # 1. 髋部连线在xz平面（俯视图）与x轴的夹角
        hip_xz_angle = calculate_angle_xz_plane(left_hip, right_hip)
        features.append(hip_xz_angle)

        # 2. 肩部连线在xz平面（俯视图）与x轴的夹角
        shoulder_xz_angle = calculate_angle_xz_plane(left_shoulder, right_shoulder)
        features.append(shoulder_xz_angle)

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
        angle_type = self.angle_labels[angle_pred]

        state_changed = False

        # 将当前帧特征加入缓存（先平滑再存入，同时保存landmarks用于矫正分析）
        landmarks_array = self._landmarks_to_array(landmarks)
        frame_features = self._compute_frame_features(landmarks_array)
        smoothed_features = self._smooth_features(frame_features)
        self.frame_buffer.append({
            'features': smoothed_features,
            'landmarks': landmarks_array,  # 保存完整landmarks用于矫正分析
            'frame_idx': self.frame_count
        })
        self.frame_count += 1

        # 状态防抖：将当前状态加入历史队列
        self.state_history.append(current_detected)

        # 只有连续 debounce_frames 帧都是同一状态才确认状态变化
        if len(self.state_history) >= self.debounce_frames:
            # 检查最近的帧是否都是同一状态
            recent_states = list(self.state_history)[-self.debounce_frames:]
            if all(s == 'squat' for s in recent_states):
                confirmed_detected = 'squat'
            elif all(s == 'stand' for s in recent_states):
                confirmed_detected = 'stand'
            else:
                confirmed_detected = self.confirmed_state  # 保持当前状态
        else:
            confirmed_detected = current_detected

        # 使用防抖后的状态进行计数
        if confirmed_detected == 'squat':
            if not self.has_squatted:
                # 记录蹲下开始的帧索引
                self.squat_start_idx = self.frame_count - 1
            self.has_squatted = True
            self.confirmed_state = 'squat'
        elif confirmed_detected == 'stand' and self.has_squatted:
            self.count += 1

            # 提取本次深蹲的动作序列并计算 DTW 分数
            user_sequence = self._extract_squat_sequence()
            if len(user_sequence) > 0:
                score = self._compute_dtw_score(user_sequence, angle_type)
                # 只有有效评分才更新
                if score is not None:
                    self.current_score = score
                    self.score_history.append(self.current_score)
                    self.avg_score = round(sum(self.score_history) / len(self.score_history), 1)
                    # 分析矫正提示
                    self.current_corrections = self._analyze_corrections(user_sequence, angle_type)
                    if self.current_corrections:
                        print(f"[矫正] 检测到问题: {[c['message'] for c in self.current_corrections]}")
                    else:
                        print(f"[矫正] 动作标准")

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
            'warning': None,
            'score': self.current_score,
            'avg_score': self.avg_score,
            'score_history': self.score_history[-10:],
            'corrections': self.current_corrections
        }

    def _extract_squat_sequence(self):
        """从缓存中提取本次深蹲的动作序列"""
        sequence = []
        for frame_data in self.frame_buffer:
            if frame_data['frame_idx'] >= self.squat_start_idx:
                sequence.append(frame_data['features'])
        return np.array(sequence) if sequence else np.array([])

    def _analyze_corrections(self, user_sequence, angle_type):
        """
        分析动作偏差，返回矫正提示
        根据不同角度类型使用不同的特征进行分析
        """
        # 获取对应角度的标准蹲下帧特征
        if angle_type not in self.standard_squat_frames:
            print(f"[矫正分析] 未找到{angle_type}的标准蹲下帧数据")
            return []

        std_features = self.standard_squat_frames[angle_type]

        # 找到用户蹲到最低点的帧（髋部高度最大的帧，即特征4）
        user_arr = np.array(user_sequence)
        user_lowest_idx = np.argmax(user_arr[:, 4])  # 特征4是髋部相对高度

        # 获取用户最低点帧的landmarks数据
        target_frame_idx = self.squat_start_idx + user_lowest_idx
        user_landmarks = None
        for frame_data in self.frame_buffer:
            if frame_data['frame_idx'] == target_frame_idx:
                user_landmarks = frame_data.get('landmarks')
                break

        if user_landmarks is None:
            print(f"[矫正分析] 未找到最低点帧的landmarks数据")
            return []

        # 计算用户最低点帧的矫正特征
        user_correction_features = self._compute_correction_features(user_landmarks, angle_type)

        corrections = []

        if angle_type == '正面':
            corrections = self._analyze_front_corrections(user_correction_features, std_features)
        elif angle_type == '侧面':
            corrections = self._analyze_side_corrections(user_correction_features, std_features)
        else:  # 斜面
            corrections = self._analyze_diagonal_corrections(user_correction_features, std_features)

        # 调试输出
        print(f"[矫正分析] 角度类型: {angle_type}")
        print(f"[矫正分析] 用户特征: {user_correction_features}")
        print(f"[矫正分析] 标准特征: {std_features}")
        if corrections:
            print(f"[矫正分析] 检测到问题: {[c['message'] for c in corrections]}")
        else:
            print(f"[矫正分析] 动作标准")

        return corrections[:2]  # 最多返回2个问题

    def _analyze_front_corrections(self, user_features, std_features):
        """分析正面角度的矫正"""
        corrections = []

        # 1. 蹲深程度比较（臀部到脚踝距离 / 鼻子到脚踝距离）
        # 值越小表示蹲得越深
        user_squat_depth = user_features.get('squat_depth', 0)
        std_squat_depth = std_features.get('squat_depth', 0)

        if std_squat_depth > 0:
            depth_dev = user_squat_depth - std_squat_depth
            depth_threshold = 0.15 * std_squat_depth  # 15%的相对阈值

            if abs(depth_dev) > depth_threshold:
                if depth_dev > 0:
                    corrections.append({
                        'feature': 'squat_depth',
                        'deviation': float(depth_dev),
                        'message': '蹲得不够深'
                    })
                else:
                    corrections.append({
                        'feature': 'squat_depth',
                        'deviation': float(depth_dev),
                        'message': '蹲得有些深'
                    })

        # 2. 膝盖间距比较（放宽阈值到25%）
        user_knee_dist = user_features.get('knee_distance', 0)
        std_knee_dist = std_features.get('knee_distance', 0)

        if std_knee_dist > 0:
            knee_dev = user_knee_dist - std_knee_dist
            knee_threshold = 0.25 * std_knee_dist  # 25%的相对阈值，放宽

            if abs(knee_dev) > knee_threshold:
                if knee_dev < 0:
                    corrections.append({
                        'feature': 'knee_distance',
                        'deviation': float(knee_dev),
                        'message': '膝盖内扣'
                    })
                else:
                    corrections.append({
                        'feature': 'knee_distance',
                        'deviation': float(knee_dev),
                        'message': '膝盖外展过度'
                    })

        # 3. 身体平衡比较
        user_balance = user_features.get('shoulder_balance', 0)
        std_balance = std_features.get('shoulder_balance', 0)
        balance_dev = user_balance - std_balance

        if abs(balance_dev) > 0.15:  # 绝对阈值
            if balance_dev > 0:
                corrections.append({
                    'feature': 'balance',
                    'deviation': float(balance_dev),
                    'message': '身体向左倾斜'
                })
            else:
                corrections.append({
                    'feature': 'balance',
                    'deviation': float(balance_dev),
                    'message': '身体向右倾斜'
                })

        # 4. 重心位置比较
        user_center = user_features.get('center_offset', 0)
        std_center = std_features.get('center_offset', 0)
        center_dev = user_center - std_center

        if abs(center_dev) > 0.2:  # 绝对阈值
            if center_dev > 0:
                corrections.append({
                    'feature': 'center',
                    'deviation': float(center_dev),
                    'message': '重心偏右'
                })
            else:
                corrections.append({
                    'feature': 'center',
                    'deviation': float(center_dev),
                    'message': '重心偏左'
                })

        # 按偏差程度排序
        corrections.sort(key=lambda x: abs(x['deviation']), reverse=True)
        return corrections

    def _analyze_side_corrections(self, user_features, std_features):
        """分析侧面角度的矫正"""
        corrections = []
        threshold = 0.15

        # 1. 膝盖弯曲角度比较
        user_knee_angle = user_features.get('knee_angle', 0)
        std_knee_angle = std_features.get('knee_angle', 0)
        knee_dev = user_knee_angle - std_knee_angle

        if abs(knee_dev) > threshold:
            if knee_dev > 0:
                corrections.append({
                    'feature': 'knee_angle',
                    'deviation': float(knee_dev),
                    'message': '蹲得不够深'
                })
            else:
                corrections.append({
                    'feature': 'knee_angle',
                    'deviation': float(knee_dev),
                    'message': '蹲得有些深'
                })

        # 2. 背部前倾角度比较
        user_back_angle = user_features.get('back_angle', 0)
        std_back_angle = std_features.get('back_angle', 0)
        back_dev = user_back_angle - std_back_angle

        if abs(back_dev) > threshold:
            if back_dev > 0:
                corrections.append({
                    'feature': 'back_angle',
                    'deviation': float(back_dev),
                    'message': '身体前倾过多'
                })
            else:
                corrections.append({
                    'feature': 'back_angle',
                    'deviation': float(back_dev),
                    'message': '身体过于直立'
                })

        # 3. 膝盖超脚尖比较（仅严重时提示）
        user_knee_over = user_features.get('knee_over_toe', 0)
        std_knee_over = std_features.get('knee_over_toe', 0)
        knee_over_dev = user_knee_over - std_knee_over

        if knee_over_dev > 0.3:  # 只有严重超出时才提示
            corrections.append({
                'feature': 'knee_over_toe',
                'deviation': float(knee_over_dev),
                'message': '膝盖超过脚尖过多'
            })

        # 按偏差程度排序
        corrections.sort(key=lambda x: abs(x['deviation']), reverse=True)
        return corrections

    def _analyze_diagonal_corrections(self, user_features, std_features):
        """分析斜面角度的矫正"""
        corrections = []

        # 1. 蹲深程度比较（臀部到脚踝距离 / 鼻子到脚踝距离）
        user_squat_depth = user_features.get('squat_depth', 0)
        std_squat_depth = std_features.get('squat_depth', 0)

        if std_squat_depth > 0:
            depth_dev = user_squat_depth - std_squat_depth
            depth_threshold = 0.15 * std_squat_depth  # 15%的相对阈值

            if abs(depth_dev) > depth_threshold:
                if depth_dev > 0:
                    corrections.append({
                        'feature': 'squat_depth',
                        'deviation': float(depth_dev),
                        'message': '蹲得不够深'
                    })
                else:
                    corrections.append({
                        'feature': 'squat_depth',
                        'deviation': float(depth_dev),
                        'message': '蹲得有些深'
                    })

        # 2. 背部前倾角度比较
        user_back_angle = user_features.get('back_angle', 0)
        std_back_angle = std_features.get('back_angle', 0)
        back_dev = user_back_angle - std_back_angle

        if abs(back_dev) > 0.15:
            if back_dev > 0:
                corrections.append({
                    'feature': 'back_angle',
                    'deviation': float(back_dev),
                    'message': '身体前倾过多'
                })
            else:
                corrections.append({
                    'feature': 'back_angle',
                    'deviation': float(back_dev),
                    'message': '身体过于直立'
                })

        # 按偏差程度排序
        corrections.sort(key=lambda x: abs(x['deviation']), reverse=True)
        return corrections
