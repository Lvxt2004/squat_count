"""
调试脚本：对比标准数据和模拟实时数据的特征
"""
import numpy as np
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_csv', 'csv_standard')

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


def compute_features(landmarks_array):
    """计算特征"""
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

    left_knee_angle = calc_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calc_angle(right_hip, right_knee, right_ankle)
    left_hip_angle = calc_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calc_angle(right_shoulder, right_hip, right_knee)

    total_height = ankle_mid[1] - shoulder_mid[1]
    if abs(total_height) > 1e-6:
        hip_relative_height = (hip_mid[1] - shoulder_mid[1]) / total_height
    else:
        hip_relative_height = 0.5

    torso_thigh_angle = calc_angle(shoulder_mid, hip_mid, (left_knee + right_knee) / 2)

    return {
        'left_knee_angle': left_knee_angle,
        'right_knee_angle': right_knee_angle,
        'left_hip_angle': left_hip_angle,
        'right_hip_angle': right_hip_angle,
        'hip_relative_height': hip_relative_height,
        'torso_thigh_angle': torso_thigh_angle,
    }


def main():
    # 加载标准数据
    df = pd.read_csv(os.path.join(DATA_DIR, 'zheng_squat_1.csv'))

    print("=" * 60)
    print("标准数据特征分析（正面深蹲）")
    print("=" * 60)

    # 分析几帧数据
    frames_to_check = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]

    for frame_idx in frames_to_check:
        row = df.iloc[frame_idx]
        coords = row.values[1:].astype(float)
        landmarks = coords.reshape(33, 3)

        features = compute_features(landmarks)

        print(f"\n帧 {frame_idx + 1}/{len(df)}:")
        print(f"  左膝角度: {features['left_knee_angle']:.1f}°")
        print(f"  右膝角度: {features['right_knee_angle']:.1f}°")
        print(f"  左髋角度: {features['left_hip_angle']:.1f}°")
        print(f"  右髋角度: {features['right_hip_angle']:.1f}°")
        print(f"  髋部相对高度: {features['hip_relative_height']:.3f}")
        print(f"  躯干大腿角: {features['torso_thigh_angle']:.1f}°")

    # 统计整个序列的特征范围
    print("\n" + "=" * 60)
    print("整个序列的特征范围")
    print("=" * 60)

    all_features = {k: [] for k in ['left_knee_angle', 'right_knee_angle',
                                     'left_hip_angle', 'right_hip_angle',
                                     'hip_relative_height', 'torso_thigh_angle']}

    for _, row in df.iterrows():
        coords = row.values[1:].astype(float)
        landmarks = coords.reshape(33, 3)
        features = compute_features(landmarks)
        for k, v in features.items():
            all_features[k].append(v)

    for k, v in all_features.items():
        print(f"{k}: min={min(v):.2f}, max={max(v):.2f}, mean={np.mean(v):.2f}")


if __name__ == '__main__':
    main()
