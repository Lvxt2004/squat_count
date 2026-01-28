"""
Flask后端应用
提供实时视频流和深蹲计数API
"""
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from squat_counter import SquatCounter

app = Flask(__name__)

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

def init_counter():
    global counter
    counter = SquatCounter(debounce_frames=5, confidence_threshold=0.7)

# 全局状态
current_status = {
    'count': 0,
    'pose': '未检测',
    'angle': '未检测',
    'pose_confidence': 0,
    'angle_confidence': 0,
    'warning': None
}

def generate_frames():
    """生成视频帧"""
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

        # 镜像翻转
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 姿态检测
        results = pose.process(rgb_frame)

        if results.pose_landmarks and counter:
            # 绘制骨架
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # 更新计数器
            status = counter.update(results.pose_landmarks.landmark)
            current_status.update(status)
        else:
            # 没有检测到人
            current_status['warning'] = "未检测到人体"

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
    return jsonify({'success': True})

if __name__ == '__main__':
    init_counter()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
