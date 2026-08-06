# VD Suit UDP → SMPL 实时转换器

该程序接收 VD Suit 的 UDP JSON 数据，把 23 关节的世界空间位置和四元数实时重定向为标准 SMPL 参数，再通过 UDP 转发和/或保存为 SMPL `.npz` 动作文件。它只依赖 NumPy，不需要 SMPL 模型文件、PyTorch 或 GPU，能够轻松处理 60Hz 数据。

## 安装

需要 Python 3.10 或更新版本。推荐使用虚拟环境隔离依赖：

```powershell
# 在项目目录下创建虚拟环境（.venv 已加入 .gitignore，不会被 Git 追踪）
python -m venv .venv

# 激活虚拟环境
.venv\Scripts\activate

# 安装项目本体 + 可视化依赖（matplotlib、pyglet）
python -m pip install -e ".[visualization]"

# 安装测试运行器
python -m pip install pytest
```

之后每次打开新终端只需激活已有的虚拟环境：

```powershell
.venv\Scripts\activate
```

如果不需要 GPU 可视化（Matplotlib / Pyglet），可以略过 `[visualization]` 只安装核心模块：

```powershell
.venv\Scripts\activate
python -m pip install -e .
```

## 运行

假设动捕发送端把数据发往本机 UDP 7001，下游程序监听 UDP 7002：

```powershell
python -m mocap_receiver --listen-port 7001 --target-port 7002
```

只录制为本地 SMPL 动作文件，不进行 UDP 转发：

```powershell
python -m mocap_receiver `
  --listen-port 7001 `
  --output-file .\motion.npz
```

同时录制 SMPL 动作文件和 UDP 转发：

```powershell
python -m mocap_receiver `
  --listen-port 7001 `
  --target-port 7002 `
  --output-file .\motion.npz
```

`.npz` 文件中的 `mocap_framerate` 默认为 60，可以根据发送端设置修改：

```powershell
python -m mocap_receiver `
  --listen-port 7001 `
  --output-file .\motion.npz `
  --mocap-framerate 60
```

按 `Ctrl+C` 正常停止录制时，程序会完成 `.npz` 文件封装。`.npz` 不是流式格式，因此不要通过任务管理器强制终止程序。

默认情况下，如果文件已经存在，程序会拒绝启动，避免覆盖已有动作。覆盖时必须显式指定：

```powershell
python -m mocap_receiver --listen-port 7001 `
  --output-file .\motion.npz --overwrite-output
```

如果仍需要原来的逐帧 JSONL，可使用 `.jsonl` 扩展名；该格式支持追加：

```powershell
python -m mocap_receiver --listen-port 7001 `
  --output-file .\smpl_frames.jsonl --append-output
```

程序根据扩展名自动选择格式，也可以使用 `--output-format smpl-npz` 或 `--output-format jsonl` 明确指定。SMPL `.npz` 是一个完整动作片段，因此不支持 `--append-output`。

也可以明确配置地址：

```powershell
vdsuit-smpl-receiver `
  --listen-host 0.0.0.0 `
  --listen-port 7001 `
  --target-host 127.0.0.1 `
  --target-port 7002 `
  --output-file .\motion.npz `
  --log-level INFO
```

按 `Ctrl+C` 停止。程序接受每个 UDP 包一个 JSON 对象，也接受像 `vdsuit_udp_stream_example.json` 一样的 JSONL（每行一个对象）。

## SMPL 动作文件格式

`.npz` 使用常见的 AMASS 风格字段，可通过 `numpy.load()` 直接读取：

- `poses`：`float32 (N, 72)`，每帧为 `global_orient(3) + body_pose(69)`。
- `trans`：`float32 (N, 3)`，每帧的 pelvis 全局平移，单位为米。
- `betas`：`float32 (10,)`，当前固定为中性平均体型的零向量。
- `gender`：字符串标量 `neutral`。
- `mocap_framerate`：浮点标量，默认为 `60.0`。

读取示例：

```python
import numpy as np

motion = np.load("motion.npz", allow_pickle=False)
poses = motion["poses"]       # (N, 72)
trans = motion["trans"]       # (N, 3)
betas = motion["betas"]       # (10,)
fps = motion["mocap_framerate"]
```

## 可视化 SMPL 动作

交互播放默认使用 Pyglet/OpenGL GPU 后端。CPU 只负责快速 SMPL LBS，完整的 13776 个三角面通过索引缓冲上传并由显卡完成深度测试和着色：

```powershell
python -m mocap_receiver.visualize .\motion.npz
```

窗口标题会显示实际 OpenGL renderer 和渲染 FPS。操作方式：

- `Space`：暂停或继续。
- `Left/Right`：逐帧移动。
- `+/-`：调整播放速度。
- 鼠标左键拖动：旋转视角。
- 鼠标滚轮：缩放。
- `W`：切换实体/线框显示。
- `R`：重置视角。
- `Esc`：退出。

GPU 窗口相关参数：

```powershell
python -m mocap_receiver.visualize .\motion.npz `
  --window-width 1280 `
  --window-height 900
```

如需禁用垂直同步可添加 `--no-vsync`。若 GPU/OpenGL 不可用，可显式切回原来的 Matplotlib 后端：

```powershell
python -m mocap_receiver.visualize .\motion.npz --backend matplotlib
```

安装项目后也可以使用命令行入口：

```powershell
vdsuit-smpl-viewer .\motion.npz
```

常用选项：

```powershell
# 两倍速播放，每两帧显示一帧，摄像机显示完整运动轨迹
python -m mocap_receiver.visualize .\motion.npz `
  --speed 2 `
  --stride 2 `
  --fixed-camera

# 导出 GIF
python -m mocap_receiver.visualize .\motion.npz `
  --save .\motion_preview.gif

# 导出 MP4，需要系统安装 ffmpeg
python -m mocap_receiver.visualize .\motion.npz `
  --save .\motion_preview.mp4
```

还支持 `--start-frame`、`--end-frame`、`--view-radius` 和 `--no-loop`。默认摄像机会跟随 pelvis 水平运动。

播放器默认从以下目录加载 SMPL v1.1 模型，并渲染完整的 6890 顶点、13776 三角面人体网格：

```text
D:\Users\budali11\Documents\phibotics\SMPL_python_v.1.1.0\smpl\models
```

NPZ 的 `gender` 会自动选择 neutral、male 或 female 模型，`betas` 会参与体型计算。也可以手动选择模型或目录：

```powershell
python -m mocap_receiver.visualize .\motion.npz `
  --gender female `
  --model-dir D:\path\to\smpl\models
```

`--mesh-face-step` 只影响 Matplotlib 后端；GPU 后端始终绘制完整网格。使用 `--skeleton` 会自动切换到 Matplotlib 的轻量骨架模式。

项目只读取本机已有模型，不会复制或重新分发受 SMPL 许可约束的模型文件。旧版 pickle 通过项目内的纯 NumPy 加载器读取，不需要安装 SciPy、Chumpy、OpenCV、PyTorch 或 smplx。GPU 窗口使用轻量的 Pyglet 2.1/OpenGL 3.3。

## 实时动捕动画

实时查看器作为独立程序运行，监听转换器发出的 SMPL UDP JSON。建议先启动查看器：

```powershell
# 终端 1：监听转换后的 SMPL 帧
python -m mocap_receiver.live_visualize `
  --listen-port 7002 `
  --render-fps 60 `
  --gender neutral
```

然后启动动捕接收和转换程序：

```powershell
# 终端 2：接收原始动捕 7001，转换后发往实时查看器 7002
python -m mocap_receiver `
  --listen-port 7001 `
  --target-port 7002
```

也可以在实时显示的同时录制 SMPL 动作：

```powershell
python -m mocap_receiver `
  --listen-port 7001 `
  --target-port 7002 `
  --output-file .\motion.npz
```

安装项目后还可以使用：

```powershell
vdsuit-smpl-live-viewer --listen-port 7002
```

实时查看器只保留最新帧，不会因为渲染暂时变慢而积压旧帧。窗口标题会显示当前帧号、接收延迟、源帧缺口、被新帧覆盖的渲染帧数和无效包数量；超过 `--stale-timeout` 未收到数据时会显示 `SIGNAL LOST`。

实时模式默认使用 SMPL 模型计算全部 6890 顶点，并显示每两个顶点中的一个，以降低 Matplotlib 的刷新开销。可用 `--mesh-vertex-step 1` 显示所有顶点。若需要三角表面，可以使用：

```powershell
python -m mocap_receiver.live_visualize `
  --listen-port 7002 `
  --surface `
  --mesh-face-step 1
```

Matplotlib 的完整三角表面刷新速度明显低于顶点模式，因此实时显示默认采用顶点模式。`--skeleton` 是速度最快的回退模式。其他常用选项包括 `--fixed-camera`、`--view-radius`、`--render-fps`、`--stale-timeout` 和 `--log-level`。

## UDP 和 JSONL 帧格式

每个有效的 `frame` 会产生一个紧凑的 UTF-8 JSON 对象；UDP 输出每帧一个包，文件输出每帧一行：

```json
{
  "type": "smpl_frame",
  "version": 1,
  "frame_index": 123,
  "coordinate_system": "SMPL_Xleft_Yup_Zforward",
  "rotation_representation": "axis_angle",
  "transl": [0.0, 1.11, 0.0],
  "global_orient": [0.0, 0.0, 0.0],
  "body_pose": ["69 axis-angle values"],
  "betas": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

- `transl`：3 维根平移，单位为米。
- `global_orient`：3 维 pelvis 全局 axis-angle。
- `body_pose`：按标准 SMPL 关节顺序展开的 23×3 局部 axis-angle。
- `betas`：固定为 10 个零，对应 SMPL 中性平均体型。
- 坐标从源 `[右, 前, 上]` 转换为 SMPL `[左, 上, 前]`，即 `[x, y, z] → [-x, z, y]`。

`skeleton` 消息只用于验证关节拓扑并更新脊柱骨长，不会转发。若先收到 `frame`，程序会立即使用示例中的内置 23 关节骨架；收到真实 `skeleton` 后自动更新。

## SMPL 关节映射说明

腿、颈、头、肩和手臂按名称映射。源 `Foot/Toe` 对应 SMPL `Ankle/Foot`，源 `Hand` 对应 SMPL `Wrist`；源数据没有独立的末端 Hand 旋转，因此 SMPL 的 `left_hand/right_hand` 使用单位局部旋转。

源骨架的四段 Spine 会根据静态骨长，在脊柱总长度的 1/3、2/3 和末端处对全局四元数做 SLERP，从而得到 SMPL 的三段 Spine，并保留上躯干最终朝向。

## 测试

```powershell
python -m unittest discover -v
```

测试覆盖示例转换、坐标和全局/局部旋转、脊柱重采样、非法输入、先帧后骨架、SMPL `.npz` 字段与形状，以及真实 UDP 回环收发。
