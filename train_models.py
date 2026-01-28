"""
训练两个KNN分类器：
1. 姿态分类器：站立 vs 蹲下 - 使用与身体姿态相关的特征
2. 角度分类器：正面 vs 侧面 vs 斜面 - 使用与面向角度相关的特征
"""
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_csv')
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


def get_point(landmarks, idx):
    """从landmarks数组获取指定索引的点坐标 (x, y, z)"""
    base = idx * 3
    return np.array([landmarks[base], landmarks[base + 1], landmarks[base + 2]])


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


def extract_angle_features(landmarks):
    """
    提取用于角度分类（正面/侧面/斜面）的特征
    只使用髋部和肩部连线在XZ平面与X轴的夹角
    正面接近0°，侧面接近90°，斜面在中间
    """
    # 获取关键点
    left_shoulder = get_point(landmarks, LEFT_SHOULDER)
    right_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    left_hip = get_point(landmarks, LEFT_HIP)
    right_hip = get_point(landmarks, RIGHT_HIP)

    features = []

    # 1. 髋部连线在xz平面（俯视图）与x轴的夹角
    hip_xz_angle = calculate_angle_xz_plane(left_hip, right_hip)
    features.append(hip_xz_angle)

    # 2. 肩部连线在xz平面（俯视图）与x轴的夹角
    shoulder_xz_angle = calculate_angle_xz_plane(left_shoulder, right_shoulder)
    features.append(shoulder_xz_angle)

    return np.array(features)


def extract_pose_features(landmarks):
    """
    提取用于姿态分类（站立/蹲下）的特征
    这些特征与用户蹲下和站立的姿态强相关
    """
    # 获取关键点
    left_shoulder = get_point(landmarks, LEFT_SHOULDER)
    right_shoulder = get_point(landmarks, RIGHT_SHOULDER)
    left_hip = get_point(landmarks, LEFT_HIP)
    right_hip = get_point(landmarks, RIGHT_HIP)
    left_knee = get_point(landmarks, LEFT_KNEE)
    right_knee = get_point(landmarks, RIGHT_KNEE)
    left_ankle = get_point(landmarks, LEFT_ANKLE)
    right_ankle = get_point(landmarks, RIGHT_ANKLE)

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
    # 站立时髋部在中间偏上，蹲下时髋部接近脚踝
    hip_relative_y = (hip_mid[1] - shoulder_mid[1]) / (ankle_mid[1] - shoulder_mid[1] + 1e-6)
    features.append(hip_relative_y)

    # 9. 膝盖y坐标相对于髋部和脚踝的位置比例
    knee_relative_y = (knee_mid[1] - hip_mid[1]) / (ankle_mid[1] - hip_mid[1] + 1e-6)
    features.append(knee_relative_y)

    # 10. 躯干与大腿的夹角（使用髋部中点）
    # 躯干向量：肩部中点 -> 髋部中点
    # 大腿向量：髋部中点 -> 膝盖中点
    torso_thigh_angle = calculate_joint_angle(shoulder_mid, hip_mid, knee_mid)
    features.append(torso_thigh_angle)

    # 11. 平均膝盖角度
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2
    features.append(avg_knee_angle)

    # 12. 平均髋关节角度
    avg_hip_angle = (left_hip_angle + right_hip_angle) / 2
    features.append(avg_hip_angle)

    return np.array(features)


def load_csv_data(file_path):
    """加载CSV文件，返回原始特征数据（去掉第一列图片名）"""
    df = pd.read_csv(file_path, header=None)
    features = df.iloc[:, 1:].values  # 去掉第一列（图片名）
    return features


def prepare_pose_data():
    """准备姿态分类数据（站立/蹲下），提取姿态相关特征"""
    angle_dir = os.path.join(DATA_DIR, 'csv_angle')

    X_list = []
    y_list = []

    # 加载站立数据
    for file in ['zheng_stand.csv', 'ce_stand.csv', 'xie_stand.csv']:
        file_path = os.path.join(angle_dir, file)
        if os.path.exists(file_path):
            raw_data = load_csv_data(file_path)
            for row in raw_data:
                features = extract_pose_features(row)
                X_list.append(features)
                y_list.append(0)  # 0 = 站立
            print(f"加载站立数据: {file}, 样本数: {len(raw_data)}")

    # 加载蹲下数据
    for file in ['zheng_squat.csv', 'ce_squat.csv', 'xie_squat.csv']:
        file_path = os.path.join(angle_dir, file)
        if os.path.exists(file_path):
            raw_data = load_csv_data(file_path)
            for row in raw_data:
                features = extract_pose_features(row)
                X_list.append(features)
                y_list.append(1)  # 1 = 蹲下
            print(f"加载蹲下数据: {file}, 样本数: {len(raw_data)}")

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def prepare_angle_data():
    """准备角度分类数据（正面/侧面/斜面），提取角度相关特征"""
    angle_dir = os.path.join(DATA_DIR, 'csv_angle')

    X_list = []
    y_list = []

    # 正面数据 (zheng)
    for file in ['zheng_stand.csv', 'zheng_squat.csv']:
        file_path = os.path.join(angle_dir, file)
        if os.path.exists(file_path):
            raw_data = load_csv_data(file_path)
            for row in raw_data:
                features = extract_angle_features(row)
                X_list.append(features)
                y_list.append(0)  # 0 = 正面
            print(f"加载正面数据: {file}, 样本数: {len(raw_data)}")

    # 侧面数据 (ce)
    for file in ['ce_stand.csv', 'ce_squat.csv']:
        file_path = os.path.join(angle_dir, file)
        if os.path.exists(file_path):
            raw_data = load_csv_data(file_path)
            for row in raw_data:
                features = extract_angle_features(row)
                X_list.append(features)
                y_list.append(1)  # 1 = 侧面
            print(f"加载侧面数据: {file}, 样本数: {len(raw_data)}")

    # 斜面数据 (xie)
    for file in ['xie_stand.csv', 'xie_squat.csv']:
        file_path = os.path.join(angle_dir, file)
        if os.path.exists(file_path):
            raw_data = load_csv_data(file_path)
            for row in raw_data:
                features = extract_angle_features(row)
                X_list.append(features)
                y_list.append(2)  # 2 = 斜面
            print(f"加载斜面数据: {file}, 样本数: {len(raw_data)}")

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y


def train_and_save_model(X, y, model_name):
    """训练KNN模型并保存，使用网格搜索找最优超参数"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 超参数搜索范围
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # 网格搜索 + 5折交叉验证
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # 最优参数
    print(f"\n{model_name} 最优超参数: {grid_search.best_params_}")
    print(f"{model_name} 交叉验证最佳得分: {grid_search.best_score_:.4f}")

    # 使用最优模型评估测试集
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} 测试集准确率: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # 保存模型和scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_knn, os.path.join(MODEL_DIR, f'{model_name}_knn.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f'{model_name}_scaler.pkl'))
    print(f"模型已保存到 {MODEL_DIR}")

    return best_knn, scaler


def main():
    print("=" * 50)
    print("训练姿态分类器（站立/蹲下）")
    print("使用特征: 归一化距离、关节角度等")
    print("=" * 50)
    X_pose, y_pose = prepare_pose_data()
    print(f"总样本数: {len(y_pose)}, 站立: {sum(y_pose==0)}, 蹲下: {sum(y_pose==1)}")
    print(f"特征维度: {X_pose.shape[1]}")
    train_and_save_model(X_pose, y_pose, 'pose')

    print("\n" + "=" * 50)
    print("训练角度分类器（正面/侧面/斜面）")
    print("使用特征: 髋部XZ夹角、肩部XZ夹角")
    print("=" * 50)
    X_angle, y_angle = prepare_angle_data()
    print(f"总样本数: {len(y_angle)}, 正面: {sum(y_angle==0)}, 侧面: {sum(y_angle==1)}, 斜面: {sum(y_angle==2)}")
    print(f"特征维度: {X_angle.shape[1]}")
    train_and_save_model(X_angle, y_angle, 'angle')

    print("\n训练完成！")


if __name__ == '__main__':
    main()
