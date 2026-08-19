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

## 命令行脚本总览

项目提供 7 个可直接运行的命令行脚本。既可以使用 `python -m ...`，也可以在
执行 `python -m pip install -e .` 后使用对应的短命令。

| 功能 | Python 模块 | 安装后的命令 |
| --- | --- | --- |
| 身体 UDP 转 SMPL、转发及录制 | `python -m mocap_receiver` | `vdsuit-smpl-receiver` |
| 播放身体 SMPL NPZ | `python -m mocap_receiver.visualize` | `vdsuit-smpl-viewer` |
| 实时显示身体 SMPL UDP | `python -m mocap_receiver.live_visualize` | `vdsuit-smpl-live-viewer` |
| 转换身体 SMPL NPZ 坐标系 | `python -m mocap_receiver.coordinate_converter` | `vdsuit-coord-convert` |
| 手部 UDP 转 SMPL-X、转发及录制 | `python -m mocap_receiver.hand_smplx_forwarder` | `vdsuit-smplx-hand-forwarder` |
| 按帧号合并身体和手部 NPZ | `python -m mocap_receiver.merge_body_hand` | `vdsuit-merge-body-hand` |
| 播放合并后的 SMPL-X NPZ | `python -m mocap_receiver.smplx_visualize` | `vdsuit-smplx-viewer` |

所有脚本都支持 `--help`。例如：

```powershell
python -m mocap_receiver.merge_body_hand --help
```

`server.py`、`recording.py`、`converter.py`、`smpl_model.py` 和
`gpu_visualize.py` 是上述命令使用的内部模块，不是独立命令行程序。

## 脚本 1：身体 UDP 转 SMPL

入口：`python -m mocap_receiver` 或 `vdsuit-smpl-receiver`。

该脚本接收 VD Suit 身体 UDP JSON，将每个 `frame` 转换为经典 SMPL 参数，并可
同时转发 UDP、保存 `.npz` 或保存逐帧 JSONL。`--target-port` 和
`--output-file` 至少需要指定一个。

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

### 身体接收脚本参数

| 参数 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--listen-host` | 否 | `0.0.0.0` | 本机绑定地址。 |
| `--listen-port` | 是 | 无 | 接收原始 VD Suit 身体数据的 UDP 端口。 |
| `--target-host` | 否 | `127.0.0.1` | 转换后 SMPL UDP 的目标地址。 |
| `--target-port` | 条件必需 | 无 | 转换后 SMPL UDP 的目标端口；不转发时省略。 |
| `--output-file` | 条件必需 | 无 | 保存为 `.npz` 或 `.jsonl`；不保存时省略。 |
| `--output-format` | 否 | `auto` | `auto`、`smpl-npz` 或 `jsonl`。 |
| `--mocap-framerate` | 否 | `60` | 写入 NPZ 的帧率元数据，不会插帧。 |
| `--append-output` | 否 | 关闭 | 追加 JSONL；不能用于 NPZ。 |
| `--overwrite-output` | 否 | 关闭 | 覆盖已有输出文件，与 `--append-output` 互斥。 |
| `--receive-size` | 否 | `65535` | 单个 UDP 数据报最大读取字节数。 |
| `--log-level` | 否 | `INFO` | `DEBUG`、`INFO`、`WARNING` 或 `ERROR`。 |

## SMPL 动作文件格式

`.npz` 使用常见的 AMASS 风格字段，可通过 `numpy.load()` 直接读取：

- `poses`：`float64 (N, 72)`，每帧为 `global_orient(3) + body_pose(69)`。
- `trans`：`float64 (N, 3)`，每帧的 pelvis 全局平移，单位为米。
- `frame_index`：`int64 (N,)`，发送端的原始帧号，用于和手部动作精确对齐。
- `betas`：`float64 (10,)`，当前固定为中性平均体型的零向量。
- `gender`：字符串标量 `neutral`。
- `mocap_framerate`：浮点标量，默认为 `60.0`。

读取示例：

```python
import numpy as np

motion = np.load("motion.npz", allow_pickle=False)
poses = motion["poses"]       # (N, 72)
trans = motion["trans"]       # (N, 3)
frame_index = motion["frame_index"]  # (N,)
betas = motion["betas"]       # (10,)
fps = motion["mocap_framerate"]
```

## 脚本 2：播放身体 SMPL NPZ

入口：`python -m mocap_receiver.visualize` 或 `vdsuit-smpl-viewer`。

输入必须是经典 SMPL 动作文件，其中 `poses` 为 `(N,72)`；该脚本不能播放
`(N,165)` 的 SMPL-X 合并文件，后者请使用脚本 7。

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

### 身体 NPZ 播放脚本参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `motion_file` | 无 | 必需位置参数，经典 SMPL `.npz` 文件。 |
| `--start-frame` | `0` | 第一帧索引，包含该帧。 |
| `--end-frame` | 文件末尾 | 结束帧索引，不包含该帧。 |
| `--stride` | `1` | 每隔多少帧显示一帧。 |
| `--speed` | `1.0` | 播放速度倍率。 |
| `--view-radius` | `1.2` | 跟随摄像机半径，单位为米。 |
| `--fixed-camera` | 关闭 | 使用覆盖完整根轨迹的固定摄像机。 |
| `--no-loop` | 关闭 | 播放到末尾后停止，不循环。 |
| `--save` | 无 | 导出 `.gif` 或 `.mp4`，不打开交互窗口。 |
| `--dpi` | `120` | 导出动画的 DPI。 |
| `--model-dir` | 内置默认路径 | SMPL v1.1 三个人体模型所在目录。 |
| `--gender` | `auto` | `auto`、`neutral`、`male` 或 `female`。 |
| `--skeleton` | 关闭 | 不加载网格，改用 Matplotlib 骨架预览。 |
| `--mesh-face-step` | `1` | Matplotlib 每隔多少个三角面绘制一个。 |
| `--backend` | `gpu` | 交互后端：`gpu` 或 `matplotlib`。 |
| `--window-width` | `1000` | GPU 窗口宽度。 |
| `--window-height` | `800` | GPU 窗口高度。 |
| `--no-vsync` | 关闭 | 禁用 GPU 窗口垂直同步。 |

## 脚本 3：实时显示身体 SMPL UDP

入口：`python -m mocap_receiver.live_visualize` 或
`vdsuit-smpl-live-viewer`。

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

实时模式默认使用 GPU/OpenGL 绘制完整 SMPL 三角网格。显式选择
`--backend matplotlib` 时，默认使用更快的顶点模式，并显示每两个顶点中的一个；
可用 `--mesh-vertex-step 1` 显示全部顶点。Matplotlib 后端如需三角表面，可以使用：

```powershell
python -m mocap_receiver.live_visualize `
  --listen-port 7002 `
  --backend matplotlib `
  --surface `
  --mesh-face-step 1
```

Matplotlib 的完整三角表面刷新速度明显低于顶点模式。`--skeleton` 是速度最快的
回退模式，并会自动使用 Matplotlib。其他常用选项包括 `--fixed-camera`、
`--view-radius`、`--render-fps`、`--stale-timeout` 和 `--log-level`。

实时查看器还会把收到的原始 UDP 数据报保存到
`<output-dir>/<启动时间>/datagram_*.jsonl`，便于复现网络输入。

### 实时身体查看脚本参数

| 参数 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--listen-host` | 否 | `0.0.0.0` | 监听转换后 SMPL UDP 的本机地址。 |
| `--listen-port` | 是 | 无 | 转换后 SMPL UDP 端口，应与身体接收器的 `--target-port` 一致。 |
| `--render-fps` | 否 | `60` | Matplotlib 窗口目标刷新率。 |
| `--view-radius` | 否 | `1.2` | 摄像机半径，单位为米。 |
| `--fixed-camera` | 否 | 关闭 | 摄像机锁定在第一帧 pelvis 附近。 |
| `--stale-timeout` | 否 | `0.5` | 超过该秒数无新帧时显示信号丢失。 |
| `--receive-size` | 否 | `65535` | 单个 UDP 数据报最大读取字节数。 |
| `--log-level` | 否 | `INFO` | 日志级别。 |
| `--model-dir` | 否 | SMPL 默认目录 | SMPL v1.1 模型目录。 |
| `--gender` | 否 | `neutral` | `neutral`、`male` 或 `female`。 |
| `--backend` | 否 | `gpu` | `gpu` 或 `matplotlib`。 |
| `--window-width` | 否 | `1000` | GPU 窗口宽度。 |
| `--window-height` | 否 | `800` | GPU 窗口高度。 |
| `--no-vsync` | 否 | 关闭 | 禁用 GPU 垂直同步。 |
| `--output-dir` | 否 | `output` | 原始 UDP 数据报保存目录。 |
| `--skeleton` | 否 | 关闭 | 使用轻量骨架；与 `--surface` 互斥。 |
| `--surface` | 否 | 关闭 | Matplotlib 绘制三角表面；默认是顶点模式。 |
| `--mesh-vertex-step` | 否 | `2` | Matplotlib 顶点模式的顶点采样步长。 |
| `--mesh-face-step` | 否 | `1` | Matplotlib 表面模式的三角面采样步长。 |

## 脚本 4：转换身体 SMPL NPZ 坐标系

入口：`python -m mocap_receiver.coordinate_converter` 或
`vdsuit-coord-convert`。

该脚本读取经典 SMPL `(N,72)` NPZ，把全局根旋转和 `trans` 转为指定坐标系；
所有身体局部旋转、体型、性别、帧数和帧率保持不变。默认同时生成 Y-up、Z-up
和 X-up 三份结果：

```powershell
python -m mocap_receiver.coordinate_converter `
  .\output\body.npz `
  --output-dir .\output\coordinate_variants
```

只生成 Z-up：

```powershell
python -m mocap_receiver.coordinate_converter `
  .\output\body.npz `
  --systems z_up `
  --output-dir .\output\coordinate_variants
```

输出目录结构为：

```text
coordinate_variants/
└── YYYYMMDD_HHMMSS/
    ├── y_up/body.npz
    ├── z_up/body.npz
    └── x_up/body.npz
```

坐标约定：

- `y_up`：X 左、Y 上、Z 前，即原生 SMPL 坐标。
- `z_up`：X 右、Y 前、Z 上。
- `x_up`：X 上、Y 前、Z 左。

该独立转换脚本输出经典 SMPL 字段，不保留用于身体/手部合并的
`frame_index`。如果还需要合并手部，请先合并，再通过 merge 脚本的
`--axis-up` 直接选择最终坐标系。

### 坐标转换脚本参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `motion_file` | 无 | 必需位置参数，经典 SMPL `.npz`。 |
| `--output-dir` | `output` | 时间戳输出目录的父目录。 |
| `--systems` | `y_up z_up x_up` | 要生成的一种或多种坐标系。 |
| `--log-level` | `INFO` | 日志级别。 |

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

## 脚本 5：手部 UDP 转 SMPL-X

入口：`python -m mocap_receiver.hand_smplx_forwarder` 或
`vdsuit-smplx-hand-forwarder`。

`mocap_receiver.hand_smplx_forwarder` 用于处理
`vdsuit_hand_udp_stream_example.json` 所示的 40 关节双手数据，将其转换成
Stage-II 风格 SMPL-X 手部参数。`--target-port` 和 `--output-file` 至少指定一个。
示例命令：

```powershell
python -m mocap_receiver.hand_smplx_forwarder `
  --listen-port 7101 `
  --target-host 192.168.1.20 `
  --target-port 7102
```

在 UDP 转发的同时保存为 SMPL-X 动作文件：

```powershell
python -m mocap_receiver.hand_smplx_forwarder `
  --listen-port 7101 `
  --target-host 192.168.1.20 `
  --target-port 7102 `
  --output-file .\output\hands.npz
```

也可以不转发，只保存本地文件：

```powershell
python -m mocap_receiver.hand_smplx_forwarder `
  --listen-port 7101 `
  --output-file .\output\hands.npz
```

`.npz` 会在按 `Ctrl-C` 正常退出时原子封装完成，包含 `(N,165)` 的
`poses`、`(N,90)` 的 `pose_hand` 以及参考 Stage-II 文件中的其他姿态字段。
所有姿态、位移和体型浮点数组均保存为 `float64`。
运行期间需要逐帧立即落盘或查看正在增长的文件时，可保存为 JSONL：

```powershell
python -m mocap_receiver.hand_smplx_forwarder `
  --listen-port 7101 `
  --output-file .\output\hands.jsonl
```

默认不会覆盖已有文件；使用 `--overwrite-output` 明确覆盖。只有 JSONL 支持
`--append-output`，NPZ 不支持追加写入。

输入支持单个 UTF-8 JSON 对象或同一 UDP 包中的 JSONL。`skeleton` 只用于校验
40 关节名称、父子关系和静态偏移，不会转发；每个 `frame` 会产生一个
`smplx_hand_frame` UDP JSON 包。接收循环使用 0.2 秒超时，因此尚未收到
`skeleton` 时按 `Ctrl-C` 也能正常退出。

输出字段与 `OK_B_stageii.npz` 对齐：`poses` 为 165 维，并严格按
`root_orient(3) + pose_body(63) + pose_jaw(3) + pose_eye(6) + pose_hand(90)`
拼接。`pose_hand` 是左手 45 维后接右手 45 维，每只手的 15 个关节顺序为：

```text
index1/2/3, middle1/2/3, pinky1/2/3, ring1/2/3, thumb1/2/3
```

源数据中食指、中指、无名指和小指各有一个额外掌骨节点。程序使用
`*Finger1/2/3` 映射 SMPL-X 的三节手指，并通过世界旋转求相对旋转，把被跳过
掌骨节点的旋转效果合并到第一节。四元数按 `wxyz` 解析为世界旋转，再转换为
SMPL-X 局部 axis-angle；坐标从 `[右, 前, 上]` 转为 `[左, 上, 前]`。

手部流无法独立推断 pelvis、身体和面部，所以这些标准 SMPL-X 参数以及
`trans` 固定填零，`betas` 使用 16 个零。为避免丢失信息，输出额外保留
`hand_positions` 和 `hand_global_orient`，顺序均由
`hand_side_order=["left", "right"]` 指定。帧率元数据默认 60Hz，可用
`--mocap-framerate` 修改。安装项目后也可使用：

```powershell
vdsuit-smplx-hand-forwarder --listen-port 7101 --target-port 7102
```

### 手部接收脚本参数

| 参数 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--listen-host` | 否 | `0.0.0.0` | 本机 UDP 绑定地址。 |
| `--listen-port` | 是 | 无 | 接收原始 40 关节手部数据的端口。 |
| `--target-host` | 否 | `127.0.0.1` | 转换后 SMPL-X UDP 目标地址。 |
| `--target-port` | 条件必需 | 无 | 转换后帧的目标端口；不转发时省略。 |
| `--output-file` | 条件必需 | 无 | 保存为 `.npz` 或 `.jsonl`；不保存时省略。 |
| `--output-format` | 否 | `auto` | `auto`、`smplx-npz` 或 `jsonl`。 |
| `--append-output` | 否 | 关闭 | 追加已有 JSONL；不能用于 NPZ。 |
| `--overwrite-output` | 否 | 关闭 | 覆盖已有输出，与 `--append-output` 互斥。 |
| `--mocap-framerate` | 否 | `60` | 写入 NPZ 的帧率元数据，不会插帧。 |
| `--receive-size` | 否 | `65535` | 单个 UDP 输入数据报最大字节数。 |
| `--log-level` | 否 | `INFO` | 日志级别。 |

## 脚本 6：按帧号合并身体和手部 NPZ

入口：`python -m mocap_receiver.merge_body_hand` 或
`vdsuit-merge-body-hand`。

身体和手部发送端使用同一个全局帧号时，可以只保留两份录制中共同存在的帧，
并合成为完整 SMPL-X Stage-II 动作：

```powershell
python -m mocap_receiver.merge_body_hand `
  .\output\body.npz `
  .\output\hands.npz `
  --output .\output\merged_smplx.npz `
  --axis-up y
```

安装项目后也可使用：

```powershell
vdsuit-merge-body-hand body.npz hands.npz --output merged_smplx.npz
```

合并器读取两边的 `frame_index`，求精确交集并按帧号升序配对。例如身体为
`[10,12,13,15]`、手部为 `[9,10,11,13,15,16]` 时，输出只包含帧
`10、13、15` 对应的动作。不会插值、复制临近帧或根据数组行号猜测；旧身体
NPZ 如果没有 `frame_index`，必须使用新版接收器重新录制。

身体提供 `root_orient(3)`、`pose_body(63)` 和 `trans(3)`；手部提供
`pose_hand(90)`、`pose_jaw(3)` 和 `pose_eye(6)`。完整姿态严格拼接为：

```text
poses = root_orient + pose_body + pose_jaw + pose_eye + pose_hand
      =       3     +     63    +     3    +     6    +     90
      = 165
```

最终 NPZ 的字段集合与 `OK_B_stageii.npz` 一致，不写入 `frame_index`、骨架消息、
`hand_positions` 或 `hand_global_orient`。实时输入没有 Stage-II marker 拟合结果，
因此 `markers_latent` 为 `(0,3)`、`latent_labels` 为空，
`markers_latent_vids` 为空字典；不会复制参考动作中无关的 marker 数据。
对应参考文件的所有浮点字段均为 `float64`；合并不插帧或重采样，输出帧率保持
两份输入共同的原始帧率。
默认拒绝覆盖已有输出，确需覆盖时添加 `--overwrite`。

`--axis-up` 用于选择合并文件的世界坐标系，默认为 `y`：

- `--axis-up y`：SMPL 原生坐标，X 向左、Y 向上、Z 向前。
- `--axis-up z`：X 向右、Y 向前、Z 向上。
- `--axis-up x`：X 向上、Y 向前、Z 向左。

坐标转换只修改 `root_orient`、`trans` 以及 `poses[:,0:3]`，不会修改
`pose_body`、`pose_hand`、`pose_jaw` 或 `pose_eye` 等局部旋转。例如生成 Z-up：

```powershell
python -m mocap_receiver.merge_body_hand `
  .\output\body.npz `
  .\output\hands.npz `
  --output .\output\merged_smplx_z_up.npz `
  --axis-up z `
  --overwrite
```

### 身体/手部合并脚本参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `body_file` | 无 | 必需位置参数，带 `frame_index` 的身体 SMPL NPZ。 |
| `hand_file` | 无 | 必需位置参数，带 `frame_index` 的手部 SMPL-X NPZ。 |
| `--output` | 无 | 必需，目标 `.npz`；其父目录必须已经存在。 |
| `--overwrite` | 关闭 | 覆盖已有目标文件。 |
| `--axis-up` | `y` | 输出上轴：`x`、`y` 或 `z`。 |
| `--log-level` | `INFO` | 日志级别。 |

## 脚本 7：播放合并后的 SMPL-X NPZ

入口：`python -m mocap_receiver.smplx_visualize` 或
`vdsuit-smplx-viewer`。

SMPL-X 查看器读取合并器生成的 `(N,165)` Stage-II NPZ，并使用官方
`smplx.lbs` 对 10475 顶点模型进行蒙皮。模型目录应直接包含
`SMPLX_NEUTRAL.npz`、`SMPLX_MALE.npz` 和 `SMPLX_FEMALE.npz`。

需要在运行该脚本的环境中安装 PyTorch、官方 `smplx` 包以及可视化依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install "smplx[all]"
.\.venv\Scripts\python.exe -m pip install -e ".[visualization]"
```

使用项目 `.venv` 和默认模型目录播放：

```powershell
.\.venv\Scripts\python.exe -m mocap_receiver.smplx_visualize `
  .\output\merged_smplx.npz `
  --model-dir .\smplx_model `
  --backend gpu `
  --device auto `
  --fixed-camera `
  --window-width 1280 `
  --window-height 720
```

GPU 窗口控制：空格暂停，左右方向键逐帧，`+/-` 调速，鼠标左键拖动旋转，
滚轮缩放，`W` 切换线框，`R` 重置视角，`Esc` 退出。`--device auto` 会优先
选择 CUDA；如果安装的是 CPU 版 PyTorch，蒙皮在 CPU 计算，OpenGL 网格仍由
显卡渲染。

导出 GIF（自动使用 Matplotlib）：

```powershell
.\.venv\Scripts\python.exe -m mocap_receiver.smplx_visualize `
  .\output\merged_smplx.npz `
  --model-dir .\smplx_model `
  --start-frame 0 `
  --end-frame 180 `
  --mesh-face-step 4 `
  --save .\output\merged_preview.gif
```

重新执行 `python -m pip install -e .` 后，也可使用命令行入口：

```powershell
vdsuit-smplx-viewer .\output\merged_smplx.npz --model-dir .\smplx_model
```

SMPL-X 查看器的摄像机以 Y 为上轴。希望人物正常竖直显示时，合并阶段应使用
`--axis-up y`；X-up/Z-up 输出主要供对应坐标约定的下游程序使用。

### SMPL-X 播放脚本参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `motion_file` | 无 | 必需位置参数，合并后的 `(N,165)` SMPL-X NPZ。 |
| `--model-dir` | 项目 `smplx_model` | 直接包含三份 `SMPLX_*.npz` 的目录。 |
| `--gender` | `auto` | `auto`、`neutral`、`male` 或 `female`。 |
| `--device` | `auto` | LBS 计算设备：`auto`、`cpu` 或 `cuda`。 |
| `--backend` | `gpu` | 交互渲染后端：`gpu` 或 `matplotlib`。 |
| `--start-frame` | `0` | 第一帧索引，包含该帧。 |
| `--end-frame` | 文件末尾 | 结束帧索引，不包含该帧。 |
| `--stride` | `1` | 每隔多少帧显示一帧。 |
| `--speed` | `1.0` | 播放速度倍率。 |
| `--view-radius` | `1.2` | 摄像机半径，单位为米。 |
| `--fixed-camera` | 关闭 | 显示完整根运动轨迹。 |
| `--no-loop` | 关闭 | 到达最后一帧后停止。 |
| `--save` | 无 | 导出 `.gif` 或 `.mp4`，自动使用 Matplotlib。 |
| `--dpi` | `120` | 导出动画 DPI。 |
| `--mesh-face-step` | `4` | Matplotlib 每隔多少个三角面绘制一个。 |
| `--window-width` | `1000` | GPU 窗口宽度。 |
| `--window-height` | `800` | GPU 窗口高度。 |
| `--no-vsync` | 关闭 | 禁用 GPU 垂直同步。 |

## 测试

```powershell
python -m unittest discover -v
```

测试覆盖示例转换、坐标和全局/局部旋转、脊柱重采样、非法输入、先帧后骨架、SMPL `.npz` 字段与形状，以及真实 UDP 回环收发。
