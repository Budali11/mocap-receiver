"""Low-latency UDP viewer for converted SMPL frames."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .converter import SMPL_JOINT_NAMES, SMPL_PARENTS
from .protocol import decode_json_messages
from .smpl_model import DEFAULT_SMPL_MODEL_DIR, SmplBody, SmplModel, SmplModelError
from .visualize import canonical_smpl_rest_offsets, forward_kinematics


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

VERTEX_SHADER = """#version 330 core
in vec3 position;
out vec3 view_position;
uniform WindowBlock {
    mat4 projection;
    mat4 view;
} window;
void main()
{
    vec4 view_pos = window.view * vec4(position, 1.0);
    view_position = view_pos.xyz;
    gl_Position = window.projection * view_pos;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 view_position;
out vec4 final_color;
uniform vec4 base_color;
void main()
{
    vec3 dx = dFdx(view_position);
    vec3 dy = dFdy(view_position);
    vec3 normal = normalize(cross(dx, dy));
    if (!gl_FrontFacing) {
        normal = -normal;
    }
    vec3 light_direction = normalize(vec3(-0.35, 0.75, 0.55));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float rim = pow(1.0 - max(dot(normal, normalize(-view_position)), 0.0), 2.0);
    float lighting = 0.28 + 0.68 * diffuse + 0.12 * rim;
    final_color = vec4(base_color.rgb * lighting, base_color.a);
}
"""


class LiveFrameError(ValueError):
    """Raised when a converted UDP frame cannot be visualized safely."""


@dataclass(frozen=True)
class LiveFrame:
    frame_index: int
    pose: FloatArray
    translation: FloatArray
    received_at: float
    betas: FloatArray | None = None


@dataclass(frozen=True)
class LiveStats:
    datagrams: int
    valid_frames: int
    invalid_messages: int
    sequence_gaps: int
    overwritten_frames: int


def _finite_array(value: Any, shape: tuple[int, ...], field: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LiveFrameError(f"{field} must contain numeric values") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise LiveFrameError(f"{field} must have shape {shape} and contain finite values")
    return array


def parse_smpl_frame(message: Mapping[str, Any], received_at: float | None = None) -> LiveFrame:
    """Validate one converter output and combine its 72 SMPL pose values."""

    if message.get("type") != "smpl_frame":
        raise LiveFrameError(f"message type must be 'smpl_frame', got {message.get('type')!r}")
    if message.get("version") != 1:
        raise LiveFrameError(f"smpl_frame.version must be 1, got {message.get('version')!r}")
    if message.get("coordinate_system") != "SMPL_Xleft_Yup_Zforward":
        raise LiveFrameError("unsupported or missing SMPL coordinate_system")
    if message.get("rotation_representation") != "axis_angle":
        raise LiveFrameError("unsupported or missing rotation_representation")
    frame_index = message.get("frame_index")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool):
        raise LiveFrameError("frame_index must be an integer")

    global_orient = _finite_array(message.get("global_orient"), (3,), "global_orient")
    body_pose = _finite_array(message.get("body_pose"), (69,), "body_pose")
    translation = _finite_array(message.get("transl"), (3,), "transl")
    betas = _finite_array(message.get("betas"), (10,), "betas")
    return LiveFrame(
        frame_index=frame_index,
        pose=np.concatenate((global_orient, body_pose)),
        translation=translation,
        received_at=time.monotonic() if received_at is None else received_at,
        betas=betas,
    )


class LatestFrameBuffer:
    """Thread-safe single-frame mailbox that never builds rendering latency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._latest: LiveFrame | None = None
        self._revision = 0
        self._consumed_revision = 0
        self._datagrams = 0
        self._valid_frames = 0
        self._invalid_messages = 0
        self._sequence_gaps = 0
        self._overwritten_frames = 0
        self._last_input_index: int | None = None

    def note_datagram(self) -> None:
        with self._lock:
            self._datagrams += 1

    def note_invalid(self, count: int = 1) -> None:
        with self._lock:
            self._invalid_messages += count

    def publish(self, frame: LiveFrame) -> None:
        with self._lock:
            if self._revision > self._consumed_revision:
                self._overwritten_frames += 1
            if self._last_input_index is not None and frame.frame_index > self._last_input_index + 1:
                self._sequence_gaps += frame.frame_index - self._last_input_index - 1
            self._last_input_index = frame.frame_index
            self._latest = frame
            self._revision += 1
            self._valid_frames += 1
            self._updated.set()

    def consume_latest(self) -> tuple[LiveFrame | None, bool, LiveStats]:
        with self._lock:
            is_new = self._revision != self._consumed_revision
            if is_new:
                self._consumed_revision = self._revision
            stats = self._stats_unlocked()
            return self._latest, is_new, stats

    def wait_for_first_frame(self, timeout: float) -> bool:
        return self._updated.wait(timeout)

    def stats(self) -> LiveStats:
        with self._lock:
            return self._stats_unlocked()

    def _stats_unlocked(self) -> LiveStats:
        return LiveStats(
            datagrams=self._datagrams,
            valid_frames=self._valid_frames,
            invalid_messages=self._invalid_messages,
            sequence_gaps=self._sequence_gaps,
            overwritten_frames=self._overwritten_frames,
        )


class LiveUdpReceiver:
    """Background UDP receiver that publishes only the latest valid frame."""

    def __init__(
        self,
        listen_host: str,
        listen_port: int,
        buffer: LatestFrameBuffer,
        receive_size: int = 65535,
        on_datagram: Callable[[bytes, tuple[str, int]], None] | None = None,
    ) -> None:
        self.buffer = buffer
        self.receive_size = receive_size
        self.on_datagram = on_datagram
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((listen_host, listen_port))
        self._socket.settimeout(0.2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._socket.getsockname()
        return str(host), int(port)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("live UDP receiver has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="smpl-live-udp",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._socket.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload, source_address = self._socket.recvfrom(self.receive_size)
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            self.buffer.note_datagram()
            if self.on_datagram is not None:
                try:
                    self.on_datagram(payload, source_address)
                except Exception:
                    LOGGER.exception("Raw datagram callback failed")
            messages, errors = decode_json_messages(payload)
            if errors:
                self.buffer.note_invalid(len(errors))
                for error in errors:
                    LOGGER.warning("Dropped live packet from %s: %s", source_address, error)
            for message in messages:
                try:
                    frame = parse_smpl_frame(message)
                except LiveFrameError as exc:
                    self.buffer.note_invalid()
                    LOGGER.warning("Dropped live packet from %s: %s", source_address, exc)
                    continue
                self.buffer.publish(frame)


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display converted SMPL UDP frames as a low-latency 3D SMPL body."
    )
    parser.add_argument(
        "--listen-host", default="0.0.0.0", help="local IPv4 address (default: 0.0.0.0)"
    )
    parser.add_argument("--listen-port", required=True, type=_port, help="SMPL UDP input port")
    parser.add_argument(
        "--render-fps", type=_positive_float, default=60.0, help="window refresh rate (default: 60)"
    )
    parser.add_argument(
        "--view-radius",
        type=_positive_float,
        default=1.2,
        help="camera half-width in meters (default: 1.2)",
    )
    parser.add_argument(
        "--fixed-camera", action="store_true", help="lock the camera around the first received root"
    )
    parser.add_argument(
        "--stale-timeout",
        type=_positive_float,
        default=0.5,
        help="seconds before showing a signal-lost warning (default: 0.5)",
    )
    parser.add_argument(
        "--receive-size", type=_positive_int, default=65535, help="maximum UDP datagram bytes"
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_SMPL_MODEL_DIR,
        help=f"directory containing SMPL v1.1 model files (default: {DEFAULT_SMPL_MODEL_DIR})",
    )
    parser.add_argument(
        "--gender",
        choices=("neutral", "male", "female"),
        default="neutral",
        help="SMPL mesh gender (default: neutral)",
    )
    parser.add_argument(
        "--backend",
        choices=("gpu", "matplotlib"),
        default="gpu",
        help="interactive rendering backend (default: gpu)",
    )
    parser.add_argument(
        "--window-width", type=_positive_int, default=1000, help="GPU window width"
    )
    parser.add_argument(
        "--window-height", type=_positive_int, default=800, help="GPU window height"
    )
    parser.add_argument("--no-vsync", action="store_true", help="disable GPU vertical sync")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory to save raw UDP datagrams (default: output/)",
    )
    render_mode = parser.add_mutually_exclusive_group()
    render_mode.add_argument(
        "--skeleton",
        action="store_true",
        help="render the lightweight skeleton instead of the SMPL mesh",
    )
    render_mode.add_argument(
        "--surface",
        action="store_true",
        help="render triangle surfaces instead of the faster SMPL vertex cloud",
    )
    parser.add_argument(
        "--mesh-vertex-step",
        type=_positive_int,
        default=2,
        help="draw every Nth SMPL vertex in live point mode (default: 2)",
    )
    parser.add_argument(
        "--mesh-face-step",
        type=_positive_int,
        default=1,
        help="draw every Nth triangle in --surface mode (default: 1)",
    )
    return parser


def _set_camera(ax: object, root: FloatArray, radius: float) -> None:
    # Matplotlib axes are [SMPL X, SMPL Z, SMPL Y] so SMPL Y appears vertical.
    ax.set_xlim(root[0] - radius, root[0] + radius)
    ax.set_ylim(root[2] - radius, root[2] + radius)
    ax.set_zlim(root[1] - radius, root[1] + radius)


def show_live_window(
    buffer: LatestFrameBuffer,
    render_fps: float,
    view_radius: float,
    fixed_camera: bool,
    stale_timeout: float,
    smpl_model: SmplModel | None = None,
    surface: bool = False,
    mesh_vertex_step: int = 2,
    mesh_face_step: int = 1,
) -> None:
    """Run the Matplotlib window on the main thread until it is closed."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required; install it with: "
            "python -m pip install -e .[visualization]"
        ) from exc

    offsets = canonical_smpl_rest_offsets()
    bones = [(parent, index) for index, parent in enumerate(SMPL_PARENTS) if parent >= 0]
    joint_colors = [
        "#2878d0"
        if name.startswith("left_")
        else "#dc4c4c"
        if name.startswith("right_")
        else "#333333"
        for name in SMPL_JOINT_NAMES
    ]
    bone_colors = [joint_colors[child] for _parent, child in bones]

    waiting_translation = np.array([0.0, 1.11, 0.0])
    smpl_body: SmplBody | None = (
        smpl_model.with_betas(np.zeros(10)) if smpl_model is not None else None
    )
    current_betas = np.zeros(10)
    waiting_mesh = (
        smpl_body.pose(np.zeros(72), waiting_translation) if smpl_body is not None else None
    )
    waiting_joints = (
        waiting_mesh.joints
        if waiting_mesh is not None
        else forward_kinematics(np.zeros(72), waiting_translation, offsets)
    )
    plotted = waiting_joints[:, [0, 2, 1]]

    figure = plt.figure(figsize=(8, 8))
    ax = figure.add_subplot(111, projection="3d")
    ax.set_xlabel("SMPL X (left)")
    ax.set_ylabel("SMPL Z (forward)")
    ax.set_zlabel("SMPL Y (up)")
    ax.set_box_aspect((1.0, 1.0, 1.3))
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    collection = Line3DCollection(
        [[plotted[parent], plotted[child]] for parent, child in bones],
        colors=bone_colors,
        linewidths=1.5 if waiting_mesh is not None else 3.0,
        alpha=0.65 if waiting_mesh is not None else 1.0,
    )
    ax.add_collection3d(collection)
    scatter = ax.scatter(
        plotted[:, 0],
        plotted[:, 1],
        plotted[:, 2],
        c=joint_colors,
        s=9 if waiting_mesh is not None else 22,
        depthshade=True,
    )
    mesh_collection = None
    mesh_faces = None
    vertex_collection = None
    vertex_indices = None
    if waiting_mesh is not None and smpl_model is not None and surface:
        mesh_faces = smpl_model.faces[::mesh_face_step]
        mesh_points = waiting_mesh.vertices[:, [0, 2, 1]]
        mesh_collection = Poly3DCollection(
            mesh_points[mesh_faces],
            facecolor="#d7a07d",
            edgecolor="none",
            alpha=0.92,
        )
        ax.add_collection3d(mesh_collection)
    elif waiting_mesh is not None:
        vertex_indices = np.arange(0, waiting_mesh.vertices.shape[0], mesh_vertex_step)
        mesh_points = waiting_mesh.vertices[vertex_indices][:, [0, 2, 1]]
        vertex_collection = ax.scatter(
            mesh_points[:, 0],
            mesh_points[:, 1],
            mesh_points[:, 2],
            c="#d7a07d",
            s=0.8,
            depthshade=False,
            alpha=0.85,
        )
    title = ax.set_title("Waiting for SMPL UDP frames...")
    _set_camera(ax, waiting_translation, view_radius)
    fixed_camera_initialized = False

    def update(_tick: int) -> tuple[object, ...]:
        nonlocal fixed_camera_initialized, smpl_body, current_betas
        frame, is_new, stats = buffer.consume_latest()
        if frame is None:
            title.set_text(
                f"Waiting for SMPL UDP frames... | packets={stats.datagrams} "
                f"invalid={stats.invalid_messages}"
            )
            return collection, scatter, title

        if is_new:
            if smpl_model is not None and frame.betas is not None:
                if smpl_body is None or not np.allclose(frame.betas, current_betas):
                    smpl_body = smpl_model.with_betas(frame.betas)
                    current_betas = frame.betas.copy()
            mesh_frame = (
                smpl_body.pose(frame.pose, frame.translation)
                if smpl_body is not None
                else None
            )
            joints = (
                mesh_frame.joints
                if mesh_frame is not None
                else forward_kinematics(frame.pose, frame.translation, offsets)
            )
            plot_points = joints[:, [0, 2, 1]]
            collection.set_segments(
                [[plot_points[parent], plot_points[child]] for parent, child in bones]
            )
            scatter._offsets3d = (plot_points[:, 0], plot_points[:, 1], plot_points[:, 2])
            if mesh_collection is not None and mesh_faces is not None and mesh_frame is not None:
                mesh_points = mesh_frame.vertices[:, [0, 2, 1]]
                mesh_collection.set_verts(mesh_points[mesh_faces])
            if vertex_collection is not None and vertex_indices is not None and mesh_frame is not None:
                mesh_points = mesh_frame.vertices[vertex_indices][:, [0, 2, 1]]
                vertex_collection._offsets3d = (
                    mesh_points[:, 0],
                    mesh_points[:, 1],
                    mesh_points[:, 2],
                )
            if not fixed_camera:
                _set_camera(ax, joints[0], view_radius)
            elif not fixed_camera_initialized:
                _set_camera(ax, joints[0], view_radius)
                fixed_camera_initialized = True

        age = max(0.0, time.monotonic() - frame.received_at)
        status = "SIGNAL LOST" if age > stale_timeout else "LIVE"
        title.set_text(
            f"{status} | frame={frame.frame_index} | age={age * 1000.0:.0f} ms | "
            f"gaps={stats.sequence_gaps} overwritten={stats.overwritten_frames} "
            f"invalid={stats.invalid_messages}"
        )
        title.set_color("#c62828" if status == "SIGNAL LOST" else "#188038")
        artists: list[object] = [collection, scatter, title]
        if mesh_collection is not None:
            artists.append(mesh_collection)
        if vertex_collection is not None:
            artists.append(vertex_collection)
        return tuple(artists)

    animation = FuncAnimation(
        figure,
        update,
        interval=1000.0 / render_fps,
        blit=False,
        cache_frame_data=False,
    )
    # Keep a strong reference for the entire blocking window lifetime.
    figure._smpl_live_animation = animation
    plt.show()


def show_live_window_gpu(
    buffer: LatestFrameBuffer,
    smpl_body: SmplBody,
    view_radius: float = 1.2,
    fixed_camera: bool = False,
    stale_timeout: float = 0.5,
    width: int = 1000,
    height: int = 800,
    vsync: bool = True,
) -> dict[str, Any]:
    """Run a Pyglet/OpenGL window that renders the latest SMPL UDP frame."""

    try:
        import pyglet
        from pyglet import gl
        from pyglet.graphics.shader import Shader, ShaderProgram
        from pyglet.math import Mat4, Vec3
        from pyglet.window import key, mouse
    except ImportError as exc:
        raise RuntimeError(
            "The GPU viewer requires Pyglet; install it with: "
            "python -m pip install -e .[visualization]"
        ) from exc

    try:
        config = gl.Config(double_buffer=True, depth_size=24, major_version=3, minor_version=3)
        window = pyglet.window.Window(
            width=width, height=height, caption="SMPL Live GPU",
            resizable=True, vsync=vsync, visible=True, config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"cannot create an OpenGL 3.3 window: {exc}") from exc

    renderer = gl.gl_info.get_renderer() or "unknown"
    vendor = gl.gl_info.get_vendor() or "unknown"
    gl_version = gl.gl_info.get_version() or "unknown"

    vertex_shader = Shader(VERTEX_SHADER, "vertex")
    fragment_shader = Shader(FRAGMENT_SHADER, "fragment")
    program = ShaderProgram(vertex_shader, fragment_shader)

    initial_translation = np.array([0.0, 1.11, 0.0], dtype=np.float64)
    initial_mesh = smpl_body.pose(np.zeros(72), initial_translation)
    vertices = np.ascontiguousarray(initial_mesh.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(smpl_body.model.faces.reshape(-1), dtype=np.uint32)
    vertex_list = program.vertex_list_indexed(
        vertices.shape[0], gl.GL_TRIANGLES, indices,
        position=("f", vertices.reshape(-1)),
    )

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glClearColor(0.94, 0.95, 0.97, 1.0)
    program.use()
    program["base_color"] = (0.80, 0.52, 0.36, 1.0)
    program.stop()

    state: dict[str, Any] = {
        "yaw": 0.0,
        "pitch": 0.08,
        "distance": view_radius * 2.5,
        "target": initial_mesh.joints[0].astype(np.float64).copy(),
        "wireframe": False,
        "draw_count": 0,
        "fps_count": 0,
        "fps_started": time.perf_counter(),
        "measured_fps": 0.0,
        "closed": False,
        "last_frame": None,
        "last_age": 0.0,
    }
    if fixed_camera:
        state["fixed_target"] = state["target"].copy()

    def update_projection() -> None:
        window.projection = Mat4.perspective_projection(
            window.aspect_ratio, z_near=0.01, z_far=1000.0, fov=45.0,
        )

    def update_camera() -> None:
        target = np.asarray(state["target"], dtype=np.float64)
        yaw = float(state["yaw"])
        pitch = float(state["pitch"])
        distance = float(state["distance"])
        eye = target + distance * np.array([
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            math.cos(yaw) * math.cos(pitch),
        ])
        window.view = Mat4.look_at(
            Vec3(*eye.tolist()), Vec3(*target.tolist()), Vec3(0.0, 1.0, 0.0),
        )

    def update_caption() -> None:
        now = time.perf_counter()
        elapsed = now - float(state["fps_started"])
        if elapsed >= 0.5:
            state["measured_fps"] = float(state["fps_count"]) / elapsed
            state["fps_count"] = 0
            state["fps_started"] = now
        status = "SIGNAL LOST" if state["last_age"] > stale_timeout else "LIVE"
        caption = (
            f"SMPL Live GPU | {status} | {state['measured_fps']:.1f} FPS | {renderer}"
        )
        window.set_caption(caption)

    @window.event
    def on_draw() -> None:
        window.clear()
        update_camera()
        if state["wireframe"]:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        program.use()
        vertex_list.draw(gl.GL_TRIANGLES)
        program.stop()
        state["draw_count"] += 1
        state["fps_count"] += 1

    @window.event
    def on_resize(new_width: int, new_height: int) -> Any:
        update_projection()
        return pyglet.event.EVENT_HANDLED

    @window.event
    def on_key_press(symbol: int, modifiers: int) -> None:
        if symbol == key.W:
            state["wireframe"] = not state["wireframe"]
        elif symbol == key.R:
            state["yaw"] = 0.0
            state["pitch"] = 0.08
            state["distance"] = view_radius * 2.5
        elif symbol == key.ESCAPE:
            window.close()

    @window.event
    def on_mouse_drag(x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> None:
        if buttons & mouse.LEFT:
            state["yaw"] -= dx * 0.01
            state["pitch"] = float(np.clip(state["pitch"] + dy * 0.01, -1.35, 1.35))
            update_camera()

    @window.event
    def on_mouse_scroll(x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        state["distance"] = float(np.clip(state["distance"] * math.exp(-scroll_y * 0.12), 0.25, 100.0))
        update_camera()

    @window.event
    def on_close() -> None:
        state["closed"] = True
        pyglet.clock.unschedule(tick)

    def tick(delta_time: float) -> None:
        frame, is_new, stats = buffer.consume_latest()
        if is_new and frame is not None:
            mesh_frame = smpl_body.pose(frame.pose, frame.translation)
            updated = np.ascontiguousarray(mesh_frame.vertices, dtype=np.float32)
            vertex_list.position[:] = updated.reshape(-1)
            if fixed_camera and "fixed_target" in state:
                state["target"] = state["fixed_target"].copy()
            else:
                state["target"] = mesh_frame.joints[0].astype(np.float64)
            state["last_frame"] = frame
            state["last_age"] = 0.0
            state["last_stats"] = stats
        elif state["last_frame"] is not None:
            state["last_age"] = max(0.0, time.monotonic() - state["last_frame"].received_at)
        update_caption()
        window.invalid = True

    update_projection()
    update_camera()
    update_caption()
    pyglet.clock.schedule_interval(tick, 1.0 / 120.0)
    try:
        pyglet.app.run(1.0 / 120.0)
    finally:
        pyglet.clock.unschedule(tick)
        vertex_list.delete()
        program.delete()
        if not state["closed"]:
            window.close()

    return {
        "renderer": str(renderer),
        "vendor": str(vendor),
        "opengl_version": str(gl_version),
        "draw_count": int(state["draw_count"]),
        "measured_fps": float(state["measured_fps"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    use_gpu = args.backend == "gpu" and not args.skeleton
    if args.skeleton and args.backend == "gpu":
        LOGGER.info("--skeleton uses the Matplotlib backend")
        use_gpu = False

    # Create timestamped output subfolder for raw datagrams.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    datagram_counter = 0
    datagram_lock = threading.Lock()

    def save_datagram(payload: bytes, source_address: tuple[str, int]) -> None:
        nonlocal datagram_counter
        with datagram_lock:
            datagram_counter += 1
            index = datagram_counter
        filename = output_dir / f"datagram_{index:06d}.jsonl"
        try:
            filename.write_bytes(payload)
        except OSError:
            LOGGER.exception("Failed to write raw datagram to %s", filename)

    LOGGER.info("Saving raw datagrams to %s", output_dir)

    buffer = LatestFrameBuffer()
    smpl_model: SmplModel | None = None
    smpl_body: SmplBody | None = None
    if use_gpu or not args.skeleton:
        try:
            smpl_model = SmplModel.from_directory(args.model_dir, args.gender)
            smpl_body = smpl_model.with_betas(np.zeros(10))
        except SmplModelError as exc:
            parser.error(f"{exc}; use --skeleton for model-free preview")
        LOGGER.info(
            "Loaded %s SMPL mesh: %d vertices, %d faces",
            args.gender,
            smpl_model.v_template.shape[0],
            smpl_model.faces.shape[0],
        )

    try:
        receiver = LiveUdpReceiver(
            args.listen_host, args.listen_port, buffer,
            receive_size=args.receive_size,
            on_datagram=save_datagram,
        )
    except OSError as exc:
        parser.error(f"cannot bind udp://{args.listen_host}:{args.listen_port}: {exc}")

    receiver.start()
    LOGGER.info("Live viewer listening on udp://%s:%d", *receiver.address)
    try:
        if use_gpu and smpl_body is not None:
            gpu_info = show_live_window_gpu(
                buffer=buffer,
                smpl_body=smpl_body,
                view_radius=args.view_radius,
                fixed_camera=args.fixed_camera,
                stale_timeout=args.stale_timeout,
                width=args.window_width,
                height=args.window_height,
                vsync=not args.no_vsync,
            )
            LOGGER.info(
                "OpenGL renderer: %s | vendor=%s | version=%s",
                gpu_info["renderer"], gpu_info["vendor"], gpu_info["opengl_version"],
            )
        else:
            show_live_window(
                buffer=buffer,
                render_fps=args.render_fps,
                view_radius=args.view_radius,
                fixed_camera=args.fixed_camera,
                stale_timeout=args.stale_timeout,
                smpl_model=smpl_model,
                surface=args.surface,
                mesh_vertex_step=args.mesh_vertex_step,
                mesh_face_step=args.mesh_face_step,
            )
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")
    except RuntimeError as exc:
        parser.error(str(exc))
    finally:
        receiver.stop()

    stats = buffer.stats()
    LOGGER.info(
        "Stats: datagrams=%d valid_frames=%d invalid=%d gaps=%d overwritten=%d",
        stats.datagrams, stats.valid_frames, stats.invalid_messages,
        stats.sequence_gaps, stats.overwritten_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
