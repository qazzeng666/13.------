# QCar2 完整一圈行驶项目

## 项目简介

BIT机车学院工程实践自动驾驶实践

基于 Quanser QLabs 虚拟仿真平台，在 Cityscape 城市场景中实现 QCar2 智能小车完成 0→20→0 节点的完整一圈自动驾驶。功能包括：

- 双 YOLO 模型感知（红绿灯/锥桶 + 行人/奶牛）
- 红绿灯识别停车（右转可通行）
- 行人、奶牛识别停车等待
- 静态锥桶自动绕行
- 鬼探头极端场景紧急制动
- 激光雷达占据栅格建图与路径显示
- LED 灯带（刹车灯、转向灯）
- 天气/时间自适应（夜晚开大灯、雨天降速）

## 文件说明

| 文件 | 作用 |
|------|------|
| `qlabs_setup_task01.py` | 场景布置脚本，运行后在 QLabs 中生成红绿灯、行人、奶牛、锥桶等 |
| `QCar2_Drive_Lap.py` | 主程序，车辆启动后完成一圈行驶并执行所有感知与决策逻辑 |
| `yolov11s.pt` | 自定义 YOLO 模型，识别红绿灯、锥桶、斑马线、停止标志 |
| `yolo26s.pt` | COCO 标准 YOLO 模型，识别行人（person）和奶牛（cow） |

## 环境要求

- **操作系统**：Windows 10/11
- **Python**：3.11（需安装 Quanser pal 库）
- **QLabs**：Quanser QLabs 已安装并可启动 Cityscape 场景
- **GPU**：NVIDIA 显卡（推荐 RTX 系列），需安装 CUDA 12.x
- **Python 依赖**：
  ```
  ultralytics
  opencv-python
  numpy
  scipy
  torch（CUDA 版本，如 torch 2.11.0+cu128）
  ```

## 运行步骤

### 1. 启动 QLabs

打开 Quanser QLabs，选择 **Cityscape** 场景并等待加载完成。

### 2. 运行场景布置

```bash
python qlabs_setup_task01.py
```

等待终端显示"环境加载完成"。

### 3. 运行主程序

```bash
python QCar2_Drive_Lap.py
```

程序启动后会自动：
- 连接 QLabs
- 加载两个 YOLO 模型
- 启动感知、控制、GUI 三个线程
- 车辆自动开始一圈行驶

### 4. 观察窗口

程序运行时会打开以下窗口：
- **环境感知与建图**：激光雷达极坐标图、局部栅格、全局路径地图
- **yolov11s**：模型一的实时检测画面（红绿灯/锥桶）
- **yolo26s**：模型二的实时检测画面（行人/奶牛）
- **信息面板**：实时显示车速、GPS 坐标、FPS

## 关键参数说明

如需调整识别灵敏度，修改 `QCar2_Drive_Lap.py` 中的以下常量：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `RED_LIGHT_STOP_AREA` | 0.3 | 红灯检测框面积阈值（占画面百分比） |
| `PEDESTRIAN_STOP_AREA` | 1.8 | 行人检测框面积阈值（占画面百分比） |
| `COW_STOP_AREA` | 7.0 | 奶牛检测框面积阈值 |
| `PEDESTRIAN_CENTER_TOL` | 0.4 | 正前方检测角度容忍度 |
| `STOP_FRAMES` | 2 | 连续检测几帧才停车 |
| `GO_FRAMES` | 6 | 目标消失几帧后恢复行驶 |
| `v_ref` | 0.3 | 正常行驶速度（m/s） |
| 雨天速度 | 0.2 | 雨天行驶速度（m/s） |
| yolov11s 置信度 | 0.4 | 红绿灯/锥桶模型置信度 |
| yolo26s 置信度 | 0.45 | 行人/奶牛模型置信度 |

## 常见问题

1. **`No module named 'quanser'`**：需使用安装了 Quanser pal 库的 Python 3.11 环境运行。
2. **YOLO 无法识别/识别慢/帧率低**：确认 PyTorch 为 CUDA 版本（`torch.cuda.is_available()` 返回 True）。
3. **QLabs 连接失败**：确保 QLabs 已启动且 Cityscape 场景已加载。
4. **车辆撞到行人/锥桶**：适当调大 `PEDESTRIAN_STOP_AREA` 或调整锥桶绕行参数。
