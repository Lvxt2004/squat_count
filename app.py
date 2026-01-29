"""
Flask后端应用
提供实时视频流和深蹲计数API
支持摄像头实时检测和上传视频检测
"""
import cv2
import os
import time
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from squat_counter import SquatCounter
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 上传文件配置
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 最大100MB

# MediaPipe初始化
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 深蹲计数器
counter = None
video_counter = None  # 视频检测用的计数器

def init_counter():
    global counter, video_counter
    counter = SquatCounter(debounce_frames=3, confidence_threshold=0.7)
    video_counter = SquatCounter(debounce_frames=3, confidence_threshold=0.7)

# 全局状态
current_status = {
    'count': 0,
    'pose': '未检测',
    'angle': '未检测',
    'pose_confidence': 0,
    'angle_confidence': 0,
    'warning': None,
    'score': 0,
    'avg_score': 0,
    'score_history': []
}

# 视频检测状态
video_status = {
    'processing': False,
    'current_video': None,
    'count': 0,
    'pose': '未检测',
    'angle': '未检测',
    'pose_confidence': 0,
    'angle_confidence': 0,
    'warning': None,
    'score': 0,
    'avg_score': 0,
    'score_history': []
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_frame(frame, squat_counter, status_dict, mirror=False):
    """
    统一的帧处理函数

    Args:
        frame: 输入的视频帧 (BGR格式)
        squat_counter: SquatCounter实例
        status_dict: 状态字典，用于更新检测结果
        mirror: 是否镜像翻转（实时摄像头需要镜像）

    Returns:
        处理后的帧 (BGR格式)
    """
    if mirror:
        frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 姿态检测
    results = pose.process(rgb_frame)

    if results.pose_landmarks and squat_counter:
        # 绘制骨架
        mp_draw.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 更新计数器（内部包含全身检测逻辑，会返回warning字段）
        status = squat_counter.update(results.pose_landmarks.landmark)
        status_dict.update(status)
    else:
        status_dict['warning'] = "未检测到人体"

    return frame


def generate_frames():
    """生成实时摄像头视频帧"""
    global current_status
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲，降低延时

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 使用统一的帧处理函数，实时摄像头需要镜像
        frame = process_frame(frame, counter, current_status, mirror=True)

        # 编码帧，降低质量减少延时
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """视频流"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """获取当前状态"""
    return jsonify(current_status)

@app.route('/reset', methods=['POST'])
def reset():
    """重置计数"""
    if counter:
        counter.reset()
        current_status['count'] = 0
        current_status['score'] = 0
        current_status['avg_score'] = 0
        current_status['score_history'] = []
    return jsonify({'success': True})


# ==================== 视频上传检测功能 ====================

def generate_video_frames(video_path):
    """从上传的视频生成帧"""
    global video_status, video_counter

    video_counter.reset()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        video_status['warning'] = "无法打开视频文件"
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps

    video_status['processing'] = True

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 使用统一的帧处理函数，上传视频不需要镜像
        frame = process_frame(frame, video_counter, video_status, mirror=False)

        # 在帧上绘制计数和分数
        cv2.putText(frame, f"Count: {video_status['count']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {video_status['score']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # 编码帧
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    video_status['processing'] = False


@app.route('/video')
def video_page():
    """视频检测页面"""
    return render_template('video.html')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """上传视频"""
    global video_status

    if 'video' not in request.files:
        return jsonify({'success': False, 'error': '没有选择文件'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        video_status['current_video'] = filepath
        video_status['count'] = 0
        video_status['score'] = 0
        video_status['avg_score'] = 0
        video_status['score_history'] = []

        return jsonify({'success': True, 'filename': filename})

    return jsonify({'success': False, 'error': '不支持的文件格式'})


@app.route('/video_play_feed')
def video_play_feed():
    """播放上传的视频流"""
    if video_status['current_video'] and os.path.exists(video_status['current_video']):
        return Response(generate_video_frames(video_status['current_video']),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return "没有视频", 404


@app.route('/video_status')
def get_video_status():
    """获取视频检测状态"""
    return jsonify(video_status)


@app.route('/video_reset', methods=['POST'])
def video_reset():
    """重置视频检测"""
    global video_counter
    if video_counter:
        video_counter.reset()
    video_status['count'] = 0
    video_status['score'] = 0
    video_status['avg_score'] = 0
    video_status['score_history'] = []
    return jsonify({'success': True})

if __name__ == '__main__':
    init_counter()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)