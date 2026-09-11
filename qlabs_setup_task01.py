# region: package imports
import os
import sys
import math
import time
import threading

from qvl.qlabs import QuanserInteractiveLabs
from qvl.qcar2 import QLabsQCar2
from qvl.free_camera import QLabsFreeCamera
from qvl.real_time import QLabsRealTime
from qvl.system import QLabsSystem
from qvl.crosswalk import QLabsCrosswalk
from qvl.traffic_light import QLabsTrafficLight
from qvl.person import QLabsPerson
from qvl.animal import QLabsAnimal
import pal.resources.rtmodels as rtmodels
#endregion

# 坐标说明：
#   本脚本所有坐标均使用 QLabs 世界坐标（单位：米）。
#   地图（SDCS RoadMap，右侧通行）上标注的米值 × 10 即为世界坐标，
#   例如地图上中央路口约 (0.1, 0.9) → 世界坐标约 (1.3, 9.5)。
#   路口四周的斑马线/红绿灯坐标取自项目中已验证可用的位置。


# QLabs 通信层不是线程安全的（单套接字、无锁），多个行人线程发指令必须串行化
_QLABS_LOCK = threading.Lock()


def _send_move(person, target, speed):
    """线程安全地下发一条移动指令。

    必须在全局锁内以 waitForConfirmation=True 调用：flush_receive + 等待 ACK
    全程串行，三个线程才不会互相吃掉对方的回执。ACK 只代表“指令已被接收”，
    并不等行人走到，走完所需时间由调用方按 路程/速度 估算。
    """
    try:
        with _QLABS_LOCK:
            return person.move_to(location=target, speed=speed, waitForConfirmation=True)
    except Exception:
        return False


def _pedestrian_patrol(person, start, other, speed, pause=1.0, initial_delay=0.0):
    """让行人在斑马线两端 start <-> other 之间持续来回穿行（后台线程运行）。

    QLabs 的 move_to 不会等行人走到，因此按 路程/速度 自行等待其真正走完，
    再在路缘停留 pause 秒后折返。
    """
    def travel_sec(a, b):
        dist = math.hypot(a[0] - b[0], a[1] - b[1])
        return dist / max(speed, 1e-6) + 0.5  # 0.5s 余量，确保走完

    time.sleep(initial_delay)  # 错峰启动，避免同一瞬间抢套接字
    current = start
    while True:
        if _send_move(person, other, speed) is False:
            time.sleep(0.2)
            continue
        time.sleep(travel_sec(current, other))
        current = other
        time.sleep(pause)  # 在对面路缘短暂停留

        if _send_move(person, start, speed) is False:
            time.sleep(0.2)
            continue
        time.sleep(travel_sec(current, start))
        current = start
        time.sleep(pause)


def _set_light_color(light, color):
    """线程安全地设置一个红绿灯颜色（加锁，等 ACK）。"""
    try:
        with _QLABS_LOCK:
            return light.set_color(color, waitForConfirmation=True)
    except Exception:
        return False


def _traffic_light_cycle(ns_lights, ew_lights,
                          green_sec=5.0, yellow_sec=1.5, all_red_sec=1.0):
    """后台线程：按标准时序循环切换红绿灯颜色。

    时序：NS绿/EW红 → NS黄/EW红 → 全红 → EW绿/NS红 → EW黄/NS红 → 全红 → 循环。
    """
    def _set_group(group, color):
        for light in group:
            _set_light_color(light, color)

    while True:
        # 1) 南北向放行
        _set_group(ns_lights, QLabsTrafficLight.COLOR_GREEN)
        _set_group(ew_lights, QLabsTrafficLight.COLOR_RED)
        time.sleep(green_sec)
        # 2) 南北向黄灯
        _set_group(ns_lights, QLabsTrafficLight.COLOR_YELLOW)
        time.sleep(yellow_sec)
        # 3) 全红清场
        _set_group(ns_lights, QLabsTrafficLight.COLOR_RED)
        time.sleep(all_red_sec)
        # 4) 东西向放行
        _set_group(ew_lights, QLabsTrafficLight.COLOR_GREEN)
        _set_group(ns_lights, QLabsTrafficLight.COLOR_RED)
        time.sleep(green_sec)
        # 5) 东西向黄灯
        _set_group(ew_lights, QLabsTrafficLight.COLOR_YELLOW)
        time.sleep(yellow_sec)
        # 6) 全红清场
        _set_group(ew_lights, QLabsTrafficLight.COLOR_RED)
        time.sleep(all_red_sec)


def setup(
        # 从地图 node 0 起步（与文件夹12多传感融合绘制地图一致），车头朝南(-y)
        # node 0 米坐标 (0, 0.13) → QLabs 世界坐标 (0, 1.3)
        initialPosition=[0, 1.3, 0],
        initialOrientation=[0, 0, -math.pi/2],
        rtModel=rtmodels.QCAR2
    ):

    # Try to connect to Qlabs
    os.system('cls')
    qlabs = QuanserInteractiveLabs()
    print("Connecting to QLabs...")
    if (not qlabs.open("localhost")):
        print("Unable to connect to QLabs")
        sys.exit()
        return
    print("Connected to QLabs")

    # 开跑前清空场景，保证本次只看到本次布置的对象（K01）
    qlabs.destroy_all_spawned_actors()
    QLabsRealTime().terminate_all_real_time_models()

    # 设置窗口标题
    hSystem = QLabsSystem(qlabs)
    hSystem.set_title_string('QCar 完整一圈场景布置')

    # ---- 生成 QCar ----
    hqcar = QLabsQCar2(qlabs)
    hqcar.spawn_id(
        actorNumber=0,
        location=initialPosition,
        rotation=initialOrientation,
        waitForConfirmation=True
    )

    # ---- 中央十字路口：4 条斑马线 ----
    # 路口中心约 (1.3, 9.5)；北/南横跨南北向道路，东/西横跨东西向道路
    crosswalks = []
    for _ in range(4):
        crosswalks.append(QLabsCrosswalk(qlabs))

    # 北侧斑马线（横跨南北向道路，y≈16）
    crosswalks[0].spawn_degrees(
        location=[1.3, 16.0, 0.02], rotation=[0, 0, 0],
        scale=[1, 1, 0.75], configuration=0, waitForConfirmation=True)
    # 西侧斑马线（横跨东西向道路，x≈-5）
    crosswalks[1].spawn_degrees(
        location=[-5.0, 9.5, 0.02], rotation=[0, 0, 90],
        scale=[1, 1, 0.75], configuration=0, waitForConfirmation=True)
    # 东侧斑马线（横跨东西向道路，x≈7.7）
    crosswalks[2].spawn_degrees(
        location=[7.7, 9.5, 0.02], rotation=[0, 0, 90],
        scale=[1, 1, 0.75], configuration=0, waitForConfirmation=True)
    # 南侧斑马线（横跨南北向道路，y≈3）
    crosswalks[3].spawn_degrees(
        location=[1.3, 3.0, 0.02], rotation=[0, 0, 0],
        scale=[1, 1, 0.75], configuration=0, waitForConfirmation=True)

    # ---- 中央十字路口：4 个红绿灯（带 ID，方便后续切换颜色）----
    # 南北向放行（绿灯），东西向停车（红灯），对应右侧通行规则
    lights = []
    for i in range(4):
        lights.append(QLabsTrafficLight(qlabs))

    # 西北角（面向由北向南来车）
    lights[0].spawn_id_degrees(
        actorNumber=1, location=[-3.77, 13.0, 0], rotation=[0, 0, 90],
        configuration=0, waitForConfirmation=True)
    # 东北角（面向由北向南来车，南北向绿灯）
    lights[1].spawn_id_degrees(
        actorNumber=2, location=[4.9, 14.8, 0], rotation=[0, 0, 0],
        configuration=0, waitForConfirmation=True)
    # 东南角（面向由东向西来车，东西向红灯）
    lights[2].spawn_id_degrees(
        actorNumber=3, location=[6.7, 5.7, 0], rotation=[0, 0, -90],
        configuration=0, waitForConfirmation=True)
    # 西南角（面向由南向北来车，南北向绿灯）
    lights[3].spawn_id_degrees(
        actorNumber=4, location=[-2.0, 4.27, 0], rotation=[0, 0, 180],
        configuration=0, waitForConfirmation=True)

    # 设置初始灯色：南北向绿、东西向红
    lights[1].set_color(lights[1].COLOR_GREEN)
    lights[3].set_color(lights[3].COLOR_GREEN)
    lights[0].set_color(lights[0].COLOR_RED)
    lights[2].set_color(lights[2].COLOR_RED)

    # 中央路口西侧（图3 左侧弯道进口）再补一个红绿灯，面向东侧来车方向
    tl_center_west = QLabsTrafficLight(qlabs)
    tl_center_west.spawn_id_degrees(
        actorNumber=5, location=[-5.5, 6.5, 0], rotation=[0, 0, 90],
        configuration=0, waitForConfirmation=True)
    tl_center_west.set_color(tl_center_west.COLOR_RED)
    lights.append(tl_center_west)

    # ---- 右侧纵向道路上的两处斑马线（对应图上右侧红蓝线区域）----
    side_crosswalks = []
    for _ in range(2):
        side_crosswalks.append(QLabsCrosswalk(qlabs))
    # 右上三叉路口处斑马线（configuration=0 为白色条纹斑马线）
    side_crosswalks[0].spawn_degrees(
        location=[21.733, 16.0, 0.02], rotation=[0, 0, 0],
        scale=[1, 1, 0.75], configuration=0, waitForConfirmation=True)
    # 右下三叉路口处斑马线
    side_crosswalks[1].spawn_degrees(
        location=[21.733, 3.347, 0.02], rotation=[0, 0, 0],
        scale=[1, 1, 0.75], configuration=0, waitForConfirmation=True)

    # ---- 右上三叉（T 型）路口：只保留 1 个红绿灯，面向南方（服务由北向南来车）----
    # 路口中心约 (21.2, 18.5)，灯放在北进口东侧车道，yaw=0 即面向南方
    tl_t_upper = QLabsTrafficLight(qlabs)
    tl_t_upper.spawn_id_degrees(
        actorNumber=6, location=[24.0, 20.8, 0], rotation=[0, 0, 0],
        configuration=0, waitForConfirmation=True)
    tl_t_upper.set_color(tl_t_upper.COLOR_GREEN)
    lights.append(tl_t_upper)
    # 右下三叉路口：按要求不放置红绿灯（仅保留斑马线）

    # ---- 过斑马线的行人（K04：人/动物专用类，可用 move_to 行走）----
    # 注意：路面 z=0，人行道比路面高；起终点必须落在路面（斑马线）范围内，
    # 放到抬高的人行道上会出现“陷进地里/坐下”的情况。
    # 每个行人由后台线程驱动，在斑马线两端持续来回穿行。
    persons = []
    patrol_threads = []
    # (实例ID, 外观, 起点, 折返点, 初始朝向角)
    # 起终点设在斑马线两端的路缘处：既完全离开行车道、车辆可驶过，
    # 又不踩到抬高的人行道（楼角弧形路缘）上，避免陷地/坐下。
    # z 保持 0.005 路面高度。
    pedestrian_cfg = [
        (10, 6, [-5.0, 6.5, 0.005],  [-5.0, 12.5, 0.005], 90),   # 中央西斑马线（原南斑马线移至此处）
        (11, 7, [4.3, 16.0, 0.005],  [-1.6, 16.0, 0.005], 180),  # 中央北斑马线
        (12, 8, [18.2, 16.0, 0.005], [24.2, 16.0, 0.005], 0),    # 右上三叉斑马线
    ]
    patrol_plan = []
    for num, cfg_id, start, other, yaw in pedestrian_cfg:
        p = QLabsPerson(qlabs)
        p.spawn_id(actorNumber=num, location=start,
                   rotation=[0, 0, math.radians(yaw)], scale=[1, 1, 1],
                   configuration=cfg_id, waitForConfirmation=True)
        persons.append(p)
        patrol_plan.append((p, start, other, p.WALK))

    # ---- 点20附近穿行马路的奶牛（K04：动物专用类，可用 move_to 行走）----
    # 点20在地图最北端，QLabs 世界坐标约 (0, 45)，该处道路为东西向；
    # 奶牛沿南北向（y 轴）从道路南侧路边走到北侧路边再折返，与道路垂直横穿，
    # 两端延伸到路缘以外，确保奶牛完全离开路面、车辆可驶过。
    animals = []
    cow_start = [0.0, 39.5, 0.005]
    cow_other = [0.0, 49.5, 0.005]
    cow = QLabsAnimal(qlabs)
    cow.spawn_id(actorNumber=13, location=cow_start,
                  rotation=[0, 0, math.pi/2], scale=[1, 1, 1],
                  configuration=cow.COW, waitForConfirmation=True)
    animals.append(cow)
    patrol_plan.append((cow, cow_start, cow_other, cow.COW_WALK))

    # ---- 创建相机并跟随 QCar ----
    hcamera = QLabsFreeCamera(qlabs)
    hcamera.spawn()
    hqcar.possess()

    # 启动实时模型
    QLabsRealTime().start_real_time_model(rtModel)

    # 所有对象生成完毕后，再错峰启动行人/动物往返线程（避免与主线程抢套接字）
    patrol_threads = []
    for idx, (p, start, other, speed) in enumerate(patrol_plan):
        t = threading.Thread(
            target=_pedestrian_patrol,
            args=(p, start, other, speed, 1.0, idx * 0.8),
            daemon=True
        )
        t.start()
        patrol_threads.append(t)

    # ---- 红绿灯颜色循环（后台线程，标准时序：绿→黄→全红→换向）----
    # 南北向组（初始绿灯）：东北ID2、西南ID4、右上T ID6
    ns_lights = [lights[1], lights[3], lights[5]]
    # 东西向组（初始红灯）：西北ID1、东南ID3、中央西侧ID5
    ew_lights = [lights[0], lights[2], lights[4]]
    tl_thread = threading.Thread(
        target=_traffic_light_cycle,
        args=(ns_lights, ew_lights),
        daemon=True
    )
    tl_thread.start()

    # 返回控制句柄，供后续决策/红绿灯切换脚本使用
    return hqcar, lights, crosswalks, persons, animals


def terminate():
    QLabsRealTime().terminate_real_time_model("QCar2_Workspace")


if __name__ == '__main__':
    # 单独运行本文件布置场景：setup() 内部用 daemon 后台线程驱动行人往返，
    # 因此主进程必须保持存活，线程才不会随脚本结束而被杀死。
    setup()
    print("场景布置完成，行人正在斑马线持续往返。按 Ctrl+C 结束并关闭实时模型。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在结束……")
    finally:
        terminate()
