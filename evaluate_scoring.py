"""
打分系统评估脚本
用于验证 DTW 打分的有效性
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_csv')

# MediaPipe 关键点索引
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def calc_angle(p1, p2, p3):
    """计算三点角度"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle) * 180 / np.pi


def compute_frame_features(landmarks_array):
    """计算单帧的特征向量（纯角度和比例特征）"""
    left_shoulder = landmarks_array[LEFT_SHOULDER]
    right_shoulder = landmarks_array[RIGHT_SHOULDER]
    left_hip = landmarks_array[LEFT_HIP]
    right_hip = landmarks_array[RIGHT_HIP]
    left_knee = landmarks_array[LEFT_KNEE]
    right_knee = landmarks_array[RIGHT_KNEE]
    left_ankle = landmarks_array[LEFT_ANKLE]
    right_ankle = landmarks_array[RIGHT_ANKLE]

    hip_mid = (left_hip + right_hip) / 2
    shoulder_mid = (left_shoulder + right_shoulder) / 2
    ankle_mid = (left_ankle + right_ankle) / 2

    # 特征1-2: 膝关节角度（归一化）
    left_knee_angle = calc_angle(left_hip, left_knee, left_ankle) / 180.0
    right_knee_angle = calc_angle(right_hip, right_knee, right_ankle) / 180.0

    # 特征3-4: 髋关节角度（归一化）
    left_hip_angle = calc_angle(left_shoulder, left_hip, left_knee) / 180.0
    right_hip_angle = calc_angle(right_shoulder, right_hip, right_knee) / 180.0

    # 特征5: 髋部相对高度
    total_height = ankle_mid[1] - shoulder_mid[1]
    if abs(total_height) > 1e-6:
        hip_relative_height = (hip_mid[1] - shoulder_mid[1]) / total_height
    else:
        hip_relative_height = 0.5

    # 特征6: 躯干与大腿夹角
    torso_thigh_angle = calc_angle(shoulder_mid, hip_mid, (left_knee + right_knee) / 2) / 180.0

    return np.array([
        left_knee_angle,
        right_knee_angle,
        left_hip_angle,
        right_hip_angle,
        hip_relative_height,
        torso_thigh_angle,
    ])


def extract_sequence_features(df):
    """从 DataFrame 提取特征序列"""
    features_list = []
    for _, row in df.iterrows():
        coords = row.values[1:].astype(float)
        landmarks_array = coords.reshape(33, 3)
        features = compute_frame_features(landmarks_array)
        features_list.append(features)
    return np.array(features_list)


def dtw_distance(seq1, seq2):
    """计算 DTW 距离"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(seq1[i - 1], seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1]
            )
    return dtw_matrix[n, m]


def compute_score(user_seq, standard_seq, debug=False):
    """计算分数"""
    if len(user_seq) < 5:
        return 0
    dtw_dist = dtw_distance(user_seq, standard_seq)

    # 归一化距离
    max_len = max(len(user_seq), len(standard_seq))
    normalized_dist = dtw_dist / max_len

    if debug:
        print(f"  DTW距离: {dtw_dist:.4f}, 归一化距离: {normalized_dist:.4f}")

    # 更宽松的线性映射
    score = 100 * max(0, 1 - normalized_dist / 0.8)

    return round(score, 1)


def add_noise(sequence, noise_level):
    """给序列添加噪声模拟不标准动作"""
    noisy = sequence.copy()
    noise = np.random.normal(0, noise_level, noisy.shape)
    return noisy + noise


def truncate_sequence(sequence, ratio):
    """截断序列模拟不完整动作"""
    end_idx = int(len(sequence) * ratio)
    return sequence[:max(5, end_idx)]


def main():
    print("=" * 60)
    print("深蹲打分系统评估")
    print("=" * 60)

    # 加载标准序列
    standard_files = {
        '正面': 'csv_standard/zheng_squat_1.csv',
        '侧面': 'csv_standard/ce_squat_1.csv',
        '斜面': 'csv_standard/xie_squat_1.csv'
    }

    for angle, filepath in standard_files.items():
        full_path = os.path.join(DATA_DIR, filepath)
        if not os.path.exists(full_path):
            print(f"文件不存在: {full_path}")
            continue

        df = pd.read_csv(full_path)
        standard_seq = extract_sequence_features(df)

        print(f"\n【{angle}】标准序列长度: {len(standard_seq)} 帧")
        print("-" * 40)

        # 测试 1: 标准序列与自身比较（应该接近 100 分）
        score_self = compute_score(standard_seq, standard_seq, debug=True)
        print(f"1. 标准 vs 标准 (理想=100): {score_self}")

        # 测试 2: 添加轻微噪声（应该 80-95 分）
        noisy_light = add_noise(standard_seq, 0.02)
        score_light = compute_score(noisy_light, standard_seq, debug=True)
        print(f"2. 轻微噪声 (理想=80-95): {score_light}")

        # 测试 3: 添加中等噪声（应该 60-80 分）
        noisy_medium = add_noise(standard_seq, 0.05)
        score_medium = compute_score(noisy_medium, standard_seq, debug=True)
        print(f"3. 中等噪声 (理想=60-80): {score_medium}")

        # 测试 4: 添加严重噪声（应该 40-60 分）
        noisy_heavy = add_noise(standard_seq, 0.1)
        score_heavy = compute_score(noisy_heavy, standard_seq)
        print(f"4. 严重噪声 (理想=40-60): {score_heavy}")

        # 测试 5: 截断序列 80%（应该仍有较高分）
        truncated_80 = truncate_sequence(standard_seq, 0.8)
        score_trunc80 = compute_score(truncated_80, standard_seq)
        print(f"5. 80%序列 (理想=70-90): {score_trunc80}")

        # 测试 6: 截断序列 50%（分数应该下降）
        truncated_50 = truncate_sequence(standard_seq, 0.5)
        score_trunc50 = compute_score(truncated_50, standard_seq)
        print(f"6. 50%序列 (理想=50-70): {score_trunc50}")

        # 测试 7: 随机序列（应该很低分）
        random_seq = np.random.rand(len(standard_seq), 6)  # 6维特征
        score_random = compute_score(random_seq, standard_seq)
        print(f"7. 随机序列 (理想=0-30): {score_random}")

    # 跨角度测试
    print("\n" + "=" * 60)
    print("跨角度测试（不同角度之间的相似度）")
    print("=" * 60)

    sequences = {}
    for angle, filepath in standard_files.items():
        full_path = os.path.join(DATA_DIR, filepath)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            sequences[angle] = extract_sequence_features(df)

    angles = list(sequences.keys())
    for i, angle1 in enumerate(angles):
        for angle2 in angles[i+1:]:
            score = compute_score(sequences[angle1], sequences[angle2])
            print(f"{angle1} vs {angle2}: {score}")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
