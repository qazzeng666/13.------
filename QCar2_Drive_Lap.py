# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
QCar2_Drive_Lap.py
QCar2 完整一圈行驶 + 双YOLO感知 + 激光雷达成图 + 多决策融合。

架构：
  - 感知线程(daemon)：双YOLO推理 + 雷达建图 + 停车判断
  - 控制线程(daemon)：EKF + QCarDriveController 循迹
  - 主线程：所有 GUI 绘制（两个YOLO窗 + MultiScope三面板）

双YOLO模型：
  - model11(yolov11s.pt)：自定义类，只识别红绿灯/锥桶/斑马线
  - model26(yolo26s.pt)：COCO标准，只识别person(0)→PEOPLE、cow(19)→COW
  - GPU加速(device=0)，两模型结果合并为detected列表

显示窗口：
  1. yolo11 - traffic：红绿灯/锥桶检测结果
  2. yolo26 - people：行人/奶牛检测结果
  3. 环境感知与建图：极坐标雷达 + 局部栅格 + 全局地图(绿规划+红实际)

决策：
  - 红灯停车（右转忽略红灯）；行人/奶牛正前方够近停车
  - 锥桶绕行：左转1.8s → 直行 → 右转回正
  - 鬼探头：车辆y>1.87时触发建筑后行人冲出
  - 天气：y>3.2夜晚开大灯；y<2.2白天；y<2雨天降速0.2
  - LED灯带：绿/刹车红/转向分半
  - 终点0→20→0回正停车

运行方式：直接运行本文件。
"""
import os
import numpy as np
from threading import Thread, Lock
import time
import signal
import cv2

from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.math import wrap_to_2pi, find_overlap
from pal.utilities.scope import MultiScope
from hal.content.qcar_functions import QCarEKF, QCarDriveController
from hal.products.mats import SDCSRoadMap
from ultralytics import YOLO
import pyqtgraph as pg
from scipy.special import logit, expit
from scipy import ndimage

import qlabs_setup_task01
#endregion

# ================ 运行日志 + 原生崩溃捕获（排查无报错退出）================
import faulthandler, sys as _sys
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_log.txt')
_logf = open(_LOG_PATH, 'w', encoding='utf-8', buffering=1)  # 行缓冲，崩溃也能落盘
faulthandler.enable(_logf)          # 原生段错误时把 C 级堆栈写入日志
faulthandler.enable()               # 同时输出到控制台

def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line)
    _logf.write(line + '\n')
    _logf.flush()

# ================ YOLO 类别（与文件夹4一致） ================
class YoloObject:
    CONE = 0; COW = 1; GREEN = 3; PEOPLE = 4; RED = 5; STOP_SIGN = 7

YOLO_LABELS = ['Cone','Cow','Crosswalk','GREEN','People','RED','StopLine','StopSign']
YOLO_COLORS = [(50,50,255),(0,204,0),(194,153,255),(255,204,51),
               (255,102,204),(0,153,255),(255,0,0),(255,255,0)]

# ================ 参数配置（对齐文件夹12） ================
tf = 120
startDelay = 1
controllerUpdateRate = 100
v_ref = 0.3
nodeSequence = [0, 20, 0]
initialPose = [0.0, 0.13, -np.pi / 2]

# 感知停车参数
RED_LIGHT_STOP_AREA = 0.3
# 行人/奶牛检测框占画面面积达到该百分比才停车。值越大→停得越近，越小→停得越远。
# （面积与距离平方成反比；5% 时约在 2.2m 外停车，8% 约停在 1.7m 处，可按实车微调）
PEDESTRIAN_STOP_AREA = 1.8
COW_STOP_AREA = 7.0  # 奶牛体积大，阈值更高，离得更近才停车
PEDESTRIAN_CENTER_TOL = 0.4
# 终点 stop 牌检测框占画面面积达到该百分比才停车（stop牌在路边，不需中心容差）
STOP_SIGN_STOP_AREA = 0.5
# 锥桶绕行：cone在正前方且面积达到该百分比时，触发绕行（向左绕开）
CONE_AVOID_AREA = 2.5
# 以下雷达参数用于建图显示；
LIDAR_OBSTACLE_DIST = 1.0
LIDAR_OBSTACLE_FOV = 36
LIDAR_OBSTACLE_MIN_POINTS = 6

# 迟滞参数
STOP_FRAMES = 2
GO_FRAMES = 6

# 占据栅格参数（与文件夹12默认值一致）
cellWidth = 0.02; r_res = 0.02; r_max = 5
p_low = 0.4; p_high = 0.6; p_prior = 0.5
l_low = logit(p_low); l_high = logit(p_high); l_prior = logit(p_prior)
phiRes = 1 * np.pi / 180
MAP_XMIN, MAP_XMAX = -4, 3
MAP_YMIN, MAP_YMAX = -3, 6

# ================ 路径生成 ================
roadmap = SDCSRoadMap(leftHandTraffic=False)
waypointSequence = roadmap.generate_path(nodeSequence)

# ================ 全局状态 ================
global KILL_THREAD
KILL_THREAD = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

_lock = Lock()
_state = {
    'should_stop': False,
    'stop_reason': '',
    'led_colors': [[0,1,0]] * 33,
    'yolo11_image': None,
    'yolo26_image': None,
    'polar_img': None,
    'patch_img': None,
    'map_dirty': False,
    'actual_pos': None,
    'actual_th': 0,
    'steering_delta': 0.0,
}
_actual_traj = [[], []]

_stop_counter = 0
_go_counter = 0

# 线程未捕获异常钩子：任何线程崩溃都打印完整堆栈，避免静默退出
def _thread_excepthook(args):
    import traceback
    print(f'!!! 线程 {args.thread.name} 崩溃: {args.exc_type.__name__}: {args.exc_value}')
    traceback.print_tb(args.exc_traceback)
import threading as _threading
_threading.excepthook = _thread_excepthook

def wrap_to_pi_vec(th):
    """向量化角度包裹到 [-pi, pi)（pal 的 wrap_to_pi 只支持标量，不能传数组）。"""
    th = np.mod(th, 2*np.pi)
    return np.where(th > np.pi, th - 2*np.pi, th)


#region : 占据栅格地图类（与文件夹12一致）

class OccupancyGrid:
    def __init__(self, x_min=MAP_XMIN, x_max=MAP_XMAX, y_min=MAP_YMIN, y_max=MAP_YMAX,
                 cellWidth=cellWidth, r_max=r_max, r_res=r_res, p_low=p_low, p_high=p_high):
        self.p_low = p_low; self.p_prior = p_prior; self.p_high = p_high
        self.p_sat = 0.001
        self.l_low = logit(p_low); self.l_prior = logit(p_prior)
        self.l_high = logit(p_high); self.l_min = logit(self.p_sat); self.l_max = logit(1-self.p_sat)
        self.cellWidth = cellWidth; self.r_max = r_max; self.r_res = r_res
        self.phiRes = phiRes
        self.mPolarPatch = int(np.ceil(2*np.pi/self.phiRes))
        self.nPolarPatch = int(np.floor(self.r_max/self.r_res))
        self.polarPatch = np.zeros((self.mPolarPatch, self.nPolarPatch), dtype=np.float32)
        self.nPatch = int(2*np.ceil(self.r_max/self.cellWidth) + 1)
        self.patch = np.zeros((self.nPatch, self.nPatch), dtype=np.float32)
        self.x_min = x_min; self.x_max = x_max; self.y_min = y_min; self.y_max = y_max
        self.xLength = x_max-x_min; self.yLength = y_max-y_min
        self.m = int(np.ceil(self.yLength/self.cellWidth))
        self.n = int(np.ceil(self.xLength/self.cellWidth))
        self.map = np.full((self.m, self.n), self.l_prior, dtype=np.float32)

    def update_polar_grid(self, r):
        r = np.int_(np.round(r/self.r_res))
        for i in range(self.mPolarPatch):
            if r[i] > 0:
                self.polarPatch[i, :r[i]] = self.l_low
                self.polarPatch[i, r[i]:r[i]+1] = self.l_high
                self.polarPatch[i, r[i]+1:] = self.l_prior
            else:
                self.polarPatch[i, :] = self.l_prior

    def generate_patch(self, th):
        cx = (self.nPatch*self.cellWidth)/2; cy = cx
        x = np.linspace(-cx, cx, self.nPatch); y = np.linspace(-cy, cy, self.nPatch)
        xv, yv = np.meshgrid(x, y)
        rPatch = np.sqrt(np.square(xv)+np.square(yv))/self.r_res
        phiPatch = wrap_to_2pi(np.arctan2(yv, xv) + th)/self.phiRes
        ndimage.map_coordinates(input=self.polarPatch, coordinates=[phiPatch, rPatch], output=self.patch)

    def xy_to_ij(self, x, y):
        i = int(np.round((self.y_max-y)/self.cellWidth))
        j = int(np.round((x-self.x_min)/self.cellWidth))
        return i, j

    def updateMap(self, x, y, th, angles, distances):
        self.update_polar_grid(distances)
        self.generate_patch(th)
        iy, jx = self.xy_to_ij(x, y)
        iTop = int(iy - np.round((self.nPatch-1)/2))
        jLeft = int(jx - np.round((self.nPatch-1)/2))
        mapSlice, patchSlice = find_overlap(self.map, self.patch, iTop, jLeft)
        self.map[mapSlice] = np.clip(self.map[mapSlice]+self.patch[patchSlice], self.l_min, self.l_max)

#endregion


#region : YOLO 检测（仿文件夹4，只处理数据不显示）

def yolo_detect(hqcar, model11, model26):
    with qlabs_setup_task01._QLABS_LOCK:
        _, img = hqcar.get_image(4)
    if img is None or img.size == 0:
        return [], None, None
    detected = []
    img11 = img.copy()
    img26 = img.copy()
    try:
        # model11：自定义类（红绿灯/锥桶/斑马线等）
        r11 = model11.predict(source=img, verbose=False, save=False, conf=0.5, device=0)
        boxes = r11[0].boxes
        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i])
            # yolo11只管红绿灯/锥桶/斑马线，忽略它识别的行人和奶牛
            if cls_id in (YoloObject.PEOPLE, YoloObject.COW):
                continue
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i]]
            area_pct = round((x2-x1)*(y2-y1)/(img.shape[0]*img.shape[1])*100, 3)
            detected.append((cls_id, area_pct, (x1,y1,x2,y2)))
            color = YOLO_COLORS[cls_id] if cls_id < len(YOLO_COLORS) else (255,255,255)
            cv2.rectangle(img11, (x1,y1), (x2,y2), color, 2)
            label = f'{YOLO_LABELS[cls_id]}({area_pct:.1f}%)'
            cv2.putText(img11, label, (x1+2, max(y1-5,15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # model26：COCO person=0→PEOPLE(4), cow=19→COW(1)
        r26 = model26.predict(source=img, verbose=False, save=False, conf=0.45, device=0)
        boxes = r26[0].boxes
        for i in range(len(boxes.cls)):
            raw = int(boxes.cls[i])
            if raw == 0:
                cls_id = YoloObject.PEOPLE
            elif raw == 19:
                cls_id = YoloObject.COW
            else:
                continue
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i]]
            area_pct = round((x2-x1)*(y2-y1)/(img.shape[0]*img.shape[1])*100, 3)
            detected.append((cls_id, area_pct, (x1,y1,x2,y2)))
            color = YOLO_COLORS[cls_id] if cls_id < len(YOLO_COLORS) else (255,255,255)
            cv2.rectangle(img26, (x1,y1), (x2,y2), color, 2)
            label = f'{YOLO_LABELS[cls_id]}({area_pct:.1f}%)'
            cv2.putText(img26, label, (x1+2, max(y1-5,15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    except Exception:
        import traceback
        log('yolo_detect异常: ' + traceback.format_exc())
    return detected, img11, img26

#endregion


#region : 停车条件判断（连续帧 + 迟滞）

def check_raw_stop_condition(detected, img, steering_delta=0.0, lidar_angles=None, lidar_distances=None):
    """只基于 YOLO 视觉判断是否停车（红绿灯/行人/奶牛）。
    雷达仅用于关联验证：YOLO说前方有行人/奶牛时，查雷达对应方向是否真有近距离障碍。
    steering_delta：当前转向角（正=左转，负=右转）。右转时允许闯红灯（红灯可右转）。
    返回 (raw_stop, reason_en)；reason 用英文以便 cv2.putText 正常显示。"""
    if img is None:
        return False, ''
    img_h, img_w = img.shape[:2]
    has_green = any(c == YoloObject.GREEN for c, _, _ in detected)
    turning_right = steering_delta < -0.05  # 右转超过约2.9°时，红灯不停车

    # 1. 红灯（绿灯优先覆盖；右转时忽略红灯）
    if not turning_right:
        for cls_id, area_pct, _ in detected:
            if cls_id == YoloObject.RED and area_pct >= RED_LIGHT_STOP_AREA and not has_green:
                return True, 'RED LIGHT'

    # 2. 行人/奶牛：正前方 + 面积达阈值（最简条件，确保能停）
    for cls_id, area_pct, (x1,y1,x2,y2) in detected:
        if cls_id in (YoloObject.PEOPLE, YoloObject.COW):
            cx = (x1+x2)/2.0
            area_thresh = PEDESTRIAN_STOP_AREA if cls_id == YoloObject.PEOPLE else COW_STOP_AREA
            if abs(cx-img_w/2.0)/img_w <= PEDESTRIAN_CENTER_TOL and area_pct >= area_thresh:
                return True, ('PEDESTRIAN' if cls_id == YoloObject.PEOPLE else 'COW')

    return False, ''


def update_stop_state(raw_stop, reason):
    global _stop_counter, _go_counter, _state
    with _lock:
        if raw_stop:
            _stop_counter += 1; _go_counter = 0
            if _stop_counter >= STOP_FRAMES:
                _state['should_stop'] = True
                _state['stop_reason'] = reason
        else:
            _go_counter += 1; _stop_counter = 0
            if _go_counter >= GO_FRAMES:
                _state['should_stop'] = False
                _state['stop_reason'] = ''

#endregion


#region : 感知线程（YOLO + 共享gps雷达 + 建图；只处理数据，不调用GUI）

def perceptionLoop(hqcar, model11, model26, gps, og):
    global KILL_THREAD, _state
    frame_n = 0
    log('感知线程进入主循环')
    while not KILL_THREAD:
        loop_start = time.time()
        try:
            # ---- YOLO 检测 ----
            detected, img11, img26 = yolo_detect(hqcar, model11, model26)

            # ---- QLabs灯带更新 ----
            with _lock:
                led_colors = _state.get('led_colors')
            if led_colors is not None:
                try:
                    with qlabs_setup_task01._QLABS_LOCK:
                        hqcar.set_led_strip_individual(led_colors, waitForConfirmation=False)
                except Exception:
                    pass

            # ---- 通过共享 gps 读取雷达（不再单独建 QCarLidar）----
            gps.readLidar()
            if hasattr(gps, 'distances') and gps.distances is not None and len(gps.distances) == og.mPolarPatch:
                distances = np.asarray(gps.distances, dtype=np.float32)
                angles = np.asarray(gps.angles, dtype=np.float32)

                # ---- 占据栅格建图（仅显示，不参与刹车）----
                with _lock:
                    pos = _state.get('actual_pos')
                    th = _state.get('actual_th', 0)
                if pos is not None:
                    px = pos[0] + 0.125*np.cos(th)
                    py = pos[1] + 0.125*np.sin(th)
                    og.updateMap(px, py, th, angles, distances)
                    with _lock:
                        _state['map_dirty'] = True

                # ---- 雷达测距：在YOLO检测框上标注距离 ----
                if img11 is not None and len(detected) > 0:
                    img_w = img11.shape[1]
                    CAM_FOV_HALF = np.pi/4
                    for cls_id, area_pct, (x1,y1,x2,y2) in detected:
                        cx = (x1+x2)/2.0
                        target_angle = (cx - img_w/2.0) / (img_w/2.0) * CAM_FOV_HALF + np.pi/2
                        idx = int(np.argmin(np.abs(angles - target_angle)))
                        ring = distances[max(0,idx-5):min(len(distances),idx+6)]
                        valid = ring[ring > 0.05]
                        if len(valid) > 0:
                            dist_m = float(valid.min())
                            if dist_m <= 5.0:
                                color = YOLO_COLORS[cls_id] if cls_id < len(YOLO_COLORS) else (255,255,255)
                                cv2.putText(img11, f'{dist_m:.1f}m',
                                            (x1+2, y2+15), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, color, 1, cv2.LINE_AA)

                # ---- 准备显示用图像（数据处理，非GUI）----
                with _lock:
                    _state['polar_img'] = expit(og.polarPatch)
                    _state['patch_img'] = expit(og.patch)

            # ---- 停车条件判断：纯 YOLO 视觉（红绿灯/行人/奶牛）----
            with _lock:
                cur_delta = _state.get('steering_delta', 0.0)
            raw_stop, reason = check_raw_stop_condition(detected, img11, cur_delta,
                                                        angles if 'angles' in dir() else None,
                                                        distances if 'distances' in dir() else None)
            update_stop_state(raw_stop, reason)

            # ---- 锥桶绕行检测：cone在正前方时，设置左转+右转回正两个窗口 ----
            now_ts = time.time()
            cone_img_w = img11.shape[1] if img11 is not None else 640
            for cls_id, area_pct, (x1,y1,x2,y2) in detected:
                if cls_id == YoloObject.CONE and area_pct >= CONE_AVOID_AREA:
                    cx = (x1+x2)/2.0
                    if abs(cx-cone_img_w/2.0)/cone_img_w <= 0.4:
                        with _lock:
                            _state['avoid_left_until'] = now_ts + 1.8       # 左转绕开1.8秒
                            _state['avoid_straight_until'] = now_ts + 3.0   # 直行1秒越过锥桶
                            _state['avoid_right_until'] = now_ts + 4.3      # 右转回正1.3秒

            with _lock:
                if img11 is not None:
                    _state['yolo11_image'] = img11
                if img26 is not None:
                    _state['yolo26_image'] = img26

            frame_n += 1
            if frame_n % 125 == 0:  # 约每5秒
                log(f'感知心跳 frame={frame_n} 检测目标数={len(detected)}')

        except Exception:
            import traceback
            log('!!! 感知线程异常:')
            _logf.write(traceback.format_exc() + '\n'); _logf.flush()
            traceback.print_exc()
            time.sleep(0.1)

        # 限速约 25Hz，给 QLabs 通信和 GUI 留余量
        elapsed = time.time()-loop_start
        time.sleep(max(0.0, 1.0/25 - elapsed))
    log('感知线程正常退出')

#endregion


#region : 控制线程（EKF + 循迹 + 停车；共享 gps）

def controlLoop(gps, qlabs=None):
    global KILL_THREAD, _state, _actual_traj
    u = 0; delta = 0; count = 0; countMax = controllerUpdateRate/10
    stop_start_t = 0.0  # 停车开始时间，用于超时保护
    ghost_triggered = False  # 鬼探头触发标志，确保只触发一次
    night_triggered = False  # 夜晚切换标志，确保只调一次
    day_restored = False     # 回到白天标志，确保只切一次
    rain_triggered = False   # 雨天标志，确保只切一次

    ekf = QCarEKF(x_0=initialPose)
    driveController = QCarDriveController(waypointSequence, cyclic=False)
    # 增大速度环比例增益，让刹车时减速更快（默认Kp=0.1偏小，刹车偏慢）
    driveController.speedController.Kp = 0.25
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)

    with qcar:
        t0 = time.time(); t = 0
        last_hb = 0
        was_stopped = False  # 上一拍是否处于停车状态（用于检测停车上升沿）
        try:
            log('控制线程进入主循环')
            while (t < tf+startDelay) and (not KILL_THREAD):
                tp = t; t = time.time()-t0; dt = t-tp

                # 车辆+GPS 读取（pal 层本身支持并发，与文件夹12一致，不额外加锁）
                qcar.read()
                gps_new = gps.readGPS()
                motor_tach = qcar.motorTach
                gyro = qcar.gyroscope[2]
                if gps_new:
                    gps_pos = np.array([gps.position[0], gps.position[1], gps.orientation[2]])
                if gps_new:
                    ekf.update([motor_tach, delta], dt, gps_pos, gyro)
                else:
                    ekf.update([motor_tach, delta], dt, None, gyro)

                x = ekf.x_hat[0,0]; y = ekf.x_hat[1,0]; th = ekf.x_hat[2,0]
                v = motor_tach
                p = np.array([x,y]) + np.array([np.cos(th),np.sin(th)])*0.2

                # 鬼探头触发：车辆北行到达y>1.87（距行人前方约4米）时触发行人冲出
                if not ghost_triggered and y > 1.87:
                    qlabs_setup_task01.GHOST_TRIGGER.set()
                    ghost_triggered = True
                    log('[鬼探头] 车辆到达触发点，行人开始冲出')

                # 到达y>3.2时切换为夜晚（只切一次），之后前大灯常亮
                if not night_triggered and y > 3.2:
                    from qvl.environment_outdoors import QLabsEnvironmentOutdoors
                    with qlabs_setup_task01._QLABS_LOCK:
                        QLabsEnvironmentOutdoors(qlabs).set_time_of_day(21.0)
                    night_triggered = True
                    log('[夜晚] 时间切换为21点，前大灯开启')

                # 回到终点y<2.2时切回白天（只切一次），关大灯
                if night_triggered and not day_restored and y < 2.2:
                    from qvl.environment_outdoors import QLabsEnvironmentOutdoors
                    with qlabs_setup_task01._QLABS_LOCK:
                        QLabsEnvironmentOutdoors(qlabs).set_time_of_day(12.0)
                    day_restored = True
                    log('[白天] 时间切回12点，前大灯关闭')

                # 继续到y<2时切换为雨天（只切一次）
                if day_restored and not rain_triggered and y < 2.0:
                    from qvl.environment_outdoors import QLabsEnvironmentOutdoors
                    with qlabs_setup_task01._QLABS_LOCK:
                        QLabsEnvironmentOutdoors(qlabs).set_weather_preset(
                            QLabsEnvironmentOutdoors.RAIN)
                    rain_triggered = True
                    log('[雨天] 天气切换为雨天')

                # 先读停车状态，再算控制量（停车时目标速度也要设0，防止速度环积分饱和）
                with _lock:
                    stop_now = _state['should_stop']
                    stop_reason = _state['stop_reason']

                if stop_now and not was_stopped:
                    # 刚进入停车：清零速度环积分，记录停车开始时间
                    driveController.speedController.reset()
                    stop_start_t = t
                was_stopped = stop_now

                if t < startDelay:
                    u, delta = 0, 0
                else:
                    # 停车期间目标速度给0：误差≈0，积分不再累积，松刹车后平顺起步
                    target_v = 0.0 if stop_now else (0.2 if rain_triggered else v_ref)
                    u, delta = driveController.update(p, th, v, target_v, dt)

                if stop_now:
                    # 保留PID负值主动制动（比u=0滑行更快停住），限制最大刹车力度
                    u = max(u, -0.15)
                    delta = 0  # 停车时同时回正方向盘，避免保持转弯

                # 锥桶两阶段绕行：先左转绕开，再右转越过锥桶回正
                now_ctrl = time.time()
                with _lock:
                    avoid_left = _state.get('avoid_left_until', 0)
                    avoid_straight = _state.get('avoid_straight_until', 0)
                    avoid_right = _state.get('avoid_right_until', 0)
                if not stop_now and now_ctrl < avoid_left:
                    delta += 0.5  # 阶段1：左转绕到锥桶左侧
                elif not stop_now and now_ctrl < avoid_straight:
                    pass  # 阶段2：直行越过锥桶，不叠加转向
                elif not stop_now and now_ctrl < avoid_right:
                    delta -= 0.30  # 阶段3：右转回正

                # QLabs灯带：33个LED，左转左半橙右半绿，右转右半橙左半绿，刹车全红，正常全绿
                left_signal = 1 if delta > 0.03 else 0
                right_signal = 1 if delta < -0.03 else 0
                LEDs = [left_signal, right_signal, left_signal, right_signal,
                        1 if stop_now else 0, 0,
                        1 if ((night_triggered and not day_restored) or rain_triggered) else 0,
                        1 if ((night_triggered and not day_restored) or rain_triggered) else 0]
                qcar.write(u, delta, LEDs)

                with _lock:
                    if stop_now:
                        _state['led_colors'] = [[1,0,0]] * 33  # 全红
                    elif left_signal:
                        _state['led_colors'] = [[0,1,0]]*11 + [[1,0.5,0]]*22  # 左转：左绿右橙
                    elif right_signal:
                        _state['led_colors'] = [[1,0.5,0]]*11 + [[0,1,0]]*22  # 右转：左橙右绿
                    else:
                        _state['led_colors'] = [[0,1,0]] * 33  # 全绿

                # 实时共享转向角，供感知线程判断是否在右转（右转时允许闯红灯）
                with _lock:
                    _state['steering_delta'] = delta

                # 每2秒一次心跳，定位退出前最后位置
                if t - last_hb >= 2:
                    last_hb = t
                    log(f'控制心跳 t={t:5.1f}s pos=({x:.2f},{y:.2f}) v={v:.2f} stop={stop_now}/{stop_reason}')

                count += 1
                if count >= countMax and t > startDelay:
                    with _lock:
                        _actual_traj[0].append(p[0]); _actual_traj[1].append(p[1])
                        _state['actual_pos'] = (p[0], p[1]); _state['actual_th'] = th
                    count = 0

                if driveController.steeringController.pathComplete:
                    log('路径完成，直行回正后停车。')
                    # 先直行1秒把车身走直（方向盘回正），再刹车停稳
                    for _ in range(30):  # 约0.6秒，直行回正
                        try:
                            qcar.write(0.08, 0)
                        except Exception:
                            pass
                        time.sleep(0.02)
                    # 强制停车：持续发送throttle=0约3秒，确保车完全停稳
                    for _ in range(150):
                        try:
                            qcar.write(0, 0)
                        except Exception:
                            pass
                        time.sleep(0.02)
                    log('车辆已停稳，停留5秒后退出。')
                    time.sleep(5.0)
                    break
        except Exception:
            import traceback
            log('!!! 控制线程异常:')
            _logf.write(traceback.format_exc() + '\n'); _logf.flush()
            traceback.print_exc()
        finally:
            try:
                qcar.read_write_std(throttle=0, steering=0)
            except Exception:
                pass
            log('控制循环结束。')

#endregion


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : 实验主流程
if __name__ == '__main__':

    def main():
        global KILL_THREAD  # 本函数内会赋值 KILL_THREAD，必须声明为全局变量
        log('程序启动')
        log('=' * 50)
        log(f'路径: {nodeSequence}  速度: {v_ref} m/s  迟滞: {STOP_FRAMES}帧停/{GO_FRAMES}帧走')

        # ---- 第一步：加载环境 ----
        if not IS_PHYSICAL_QCAR:
            hqcar, _, _, _, _, qlabs = qlabs_setup_task01.setup(
                initialPosition=[0, 1.3, 0],
                initialOrientation=[0, 0, -np.pi/2]
            )
            log('环境加载完成')

        # ---- 第二步：加载两个 YOLO 模型 ----
        # model11：自定义训练，管红绿灯/锥桶/斑马线/停止线
        # model26：标准COCO，管行人和奶牛（person=0→PEOPLE, cow=19→COW）
        model11_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov11s.pt')
        model26_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo26s.pt')
        model11 = YOLO(model11_path)
        model26 = YOLO(model26_path)
        # 尝试GPU加速
        try:
            import torch
            if torch.cuda.is_available():
                model11.to('cuda'); model26.to('cuda')
                log('双YOLO模型已加载（GPU加速）')
            else:
                log('双YOLO模型已加载（CPU，无GPU）')
        except Exception as e:
            log(f'双YOLO模型已加载（CPU）: {e}')

        # ---- 第三步：创建唯一的 QCarGPS（内部含雷达），等待传感器就绪 ----
        gps = QCarGPS(initialPose=initialPose, calibrate=False)
        while (not KILL_THREAD) and (gps.readGPS() or gps.readLidar()):
            pass
        og = OccupancyGrid()
        log('GPS/激光雷达就绪')

        # ---- 第四步：创建 YOLO 窗口（主线程创建+显示）----
        cv2.namedWindow('yolo11 - traffic', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('yolo11 - traffic', 832, 624)
        cv2.namedWindow('yolo26 - people', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('yolo26 - people', 832, 624)

        # ---- 第五步：配置三面板显示窗口（与文件夹12一致，均为图像）----
        scope = MultiScope(rows=2, cols=2, title='环境感知与建图', fps=30)
        scope.graphicsLayoutWidget.resize(900, 560)

        scope.addXYAxis(row=0, col=0, xLabel='角度 [度]', yLabel='距离 [米]')
        scope.axes[0].attachImage()
        scope.addXYAxis(row=1, col=0, xLabel='X轴位置 [米]', yLabel='Y轴位置 [米]')
        scope.axes[1].attachImage()
        scope.addXYAxis(row=0, col=1, rowSpan=2, xLabel='X轴位置 [米]', yLabel='Y轴位置 [米]',
                        xLim=(MAP_XMIN, MAP_XMAX), yLim=(MAP_YMIN, MAP_YMAX))
        scope.axes[2].attachImage()

        scope.axes[0].images[0].rotation = 90
        scope.axes[0].images[0].scale = (og.r_res, -og.phiRes*180/np.pi)
        scope.axes[0].images[0].offset = (0, 0)
        scope.axes[0].images[0].levels = (0, 1)

        scope.axes[1].images[0].scale = (og.r_res, -og.r_res)
        scope.axes[1].images[0].offset = (-og.nPatch/2, -og.nPatch/2)
        scope.axes[1].images[0].levels = (0, 1)

        scope.axes[2].images[0].scale = (og.cellWidth, -og.cellWidth)
        scope.axes[2].images[0].offset = (og.x_min/og.cellWidth, -og.y_max/og.cellWidth)
        scope.axes[2].images[0].levels = (0, 1)

        ref_path = pg.PlotDataItem(pen={'color':(85,168,104),'width':2}, name='参考路径')
        ref_path.setData(waypointSequence[0,:], waypointSequence[1,:])
        scope.axes[2].plot.addItem(ref_path)
        actual_path = pg.PlotDataItem(pen={'color':(196,78,82),'width':2}, name='实际轨迹')
        scope.axes[2].plot.addItem(actual_path)
        car_arrow = pg.ArrowItem(angle=180, tipAngle=60, headLen=12, tailLen=12, tailWidth=4,
                                  pen={'color':'w','width':1}, brush=[196,78,82])
        car_arrow.setPos(initialPose[0], initialPose[1])
        scope.axes[2].plot.addItem(car_arrow)
        scope.axes[2].plot.addLegend()
        log('显示窗口创建完成')

        # ---- 第六步：启动后台线程 ----
        perceptionThread = Thread(target=perceptionLoop, args=(hqcar, model11, model26, gps, og),
                                  daemon=True, name='Perception')
        controlThread = Thread(target=controlLoop, args=(gps, qlabs), daemon=True, name='Control')
        controlThread.start()
        perceptionThread.start()
        log('后台线程已启动，车辆开始行驶')

        # ---- 第七步：主线程负责所有 GUI 绘制 ----
        gui_n = 0
        try:
            while controlThread.is_alive() and (not KILL_THREAD):
                if not perceptionThread.is_alive():
                    log('感知线程已退出，正在重启...')
                    perceptionThread = Thread(target=perceptionLoop, args=(hqcar, model11, model26, gps, og),
                                              daemon=True, name='Perception')
                    perceptionThread.start()

                with _lock:
                    yolo11_img = _state['yolo11_image']
                    yolo26_img = _state['yolo26_image']
                    polar_img = _state['polar_img']
                    patch_img = _state['patch_img']
                    xs = list(_actual_traj[0]); ys = list(_actual_traj[1])
                    pos = _state.get('actual_pos'); th = _state.get('actual_th', 0)
                    should_stop = _state['should_stop']; stop_reason = _state['stop_reason']
                    map_dirty = _state['map_dirty']; _state['map_dirty'] = False

                try:
                    # 每帧显示，缩小一半再渲染降低imshow开销
                    if yolo11_img is not None:
                        disp11 = yolo11_img.copy()
                        cv2.putText(disp11, 'YOLO11: Traffic Lights / Cone', (10, disp11.shape[0]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
                        if should_stop:
                            cv2.putText(disp11, f'STOP: {stop_reason}', (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(disp11, 'GO', (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        disp11_small = cv2.resize(disp11, (disp11.shape[1]//2, disp11.shape[0]//2))
                        cv2.imshow('yolo11 - traffic', disp11_small)
                    if yolo26_img is not None:
                        disp26 = yolo26_img.copy()
                        cv2.putText(disp26, 'YOLO26: Pedestrian / Cow', (10, disp26.shape[0]-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
                        if should_stop:
                            cv2.putText(disp26, f'STOP: {stop_reason}', (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(disp26, 'GO', (10,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                        disp26_small = cv2.resize(disp26, (disp26.shape[1]//2, disp26.shape[0]//2))
                        cv2.imshow('yolo26 - people', disp26_small)
                    cv2.waitKey(1)

                    if polar_img is not None:
                        scope.axes[0].images[0].setImage(image=polar_img)
                    if patch_img is not None:
                        scope.axes[1].images[0].setImage(image=patch_img)
                    if map_dirty:
                        scope.axes[2].images[0].setImage(image=expit(og.map))

                    if len(xs) > 1:
                        actual_path.setData(xs, ys)
                    if pos is not None:
                        car_arrow.setPos(pos[0], pos[1])
                        car_arrow.setStyle(angle=180-th*180/np.pi)

                    MultiScope.refreshAll()
                except Exception:
                    import traceback
                    _logf.write('GUI异常:\n' + traceback.format_exc() + '\n'); _logf.flush()
                    traceback.print_exc()

                gui_n += 1
                if gui_n % 200 == 0:
                    log(f'GUI心跳 刷新{gui_n}次 controlAlive={controlThread.is_alive()} '
                        f'percepAlive={perceptionThread.is_alive()}')
                time.sleep(0.015)
        finally:
            KILL_THREAD = True

        log(f'主GUI循环退出 controlAlive={controlThread.is_alive()} '
            f'percepAlive={perceptionThread.is_alive()} KILL={KILL_THREAD}')
        controlThread.join(timeout=5)
        perceptionThread.join(timeout=5)

        # ---- 第八步：清理 ----
        log('正在清理...')
        try:
            gps.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()
        if not IS_PHYSICAL_QCAR:
            # 程序结束恢复晴朗白天，避免下次运行残留雨天/夜晚
            try:
                from qvl.environment_outdoors import QLabsEnvironmentOutdoors
                env = QLabsEnvironmentOutdoors(qlabs)
                env.set_weather_preset(QLabsEnvironmentOutdoors.CLEAR_SKIES)
                env.set_time_of_day(12.0)
            except Exception:
                pass
            qlabs_setup_task01.terminate()
        log('完成。')

    try:
        main()
    except Exception:
        import traceback
        log('!!! 主线程致命异常，程序即将退出:')
        err = traceback.format_exc()
        _logf.write(err + '\n'); _logf.flush()
        print(err)
    finally:
        _logf.close()
        # 暂停，避免 VSCode 终端一闪而过看不到信息
        try:
            input('程序结束，按回车键关闭窗口...（详细日志见 run_log.txt）')
        except Exception:
            time.sleep(3)
#endregion
