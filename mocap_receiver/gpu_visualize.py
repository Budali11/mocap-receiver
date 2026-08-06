"""Pyglet/OpenGL GPU player for SMPL mesh motion."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .smpl_model import SmplBody
from .visualize import SmplMotion


FloatArray = NDArray[np.float64]


VERTEX_SHADER = """#version 330 core
in vec3 position;

out vec3 view_position;

uniform WindowBlock
{
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


def play_motion_gpu(
    motion: SmplMotion,
    frame_indices: NDArray[np.int64],
    smpl_body: SmplBody,
    speed: float,
    view_radius: float,
    fixed_camera: bool,
    repeat: bool,
    width: int = 1000,
    height: int = 800,
    vsync: bool = True,
    visible: bool = True,
    auto_close_after_frames: int | None = None,
) -> dict[str, Any]:
    """Play a SMPL motion with indexed OpenGL triangles on the active GPU.

    ``visible`` and ``auto_close_after_frames`` exist to support automated GPU
    smoke tests. Normal callers should keep their defaults.
    """

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

    if frame_indices.ndim != 1 or frame_indices.size == 0:
        raise ValueError("frame_indices must contain at least one frame")
    if width <= 0 or height <= 0:
        raise ValueError("GPU window dimensions must be greater than zero")

    try:
        config = gl.Config(
            double_buffer=True,
            depth_size=24,
            major_version=3,
            minor_version=3,
        )
        window = pyglet.window.Window(
            width=width,
            height=height,
            caption="SMPL GPU Player",
            resizable=True,
            vsync=vsync,
            visible=visible,
            config=config,
        )
    except Exception as exc:
        raise RuntimeError(f"cannot create an OpenGL 3.3 window: {exc}") from exc

    renderer = gl.gl_info.get_renderer()
    vendor = gl.gl_info.get_vendor()
    gl_version = gl.gl_info.get_version()
    if not isinstance(renderer, str):
        renderer = str(renderer)
    if not isinstance(vendor, str):
        vendor = str(vendor)
    if not isinstance(gl_version, str):
        gl_version = str(gl_version)

    vertex_shader = Shader(VERTEX_SHADER, "vertex")
    fragment_shader = Shader(FRAGMENT_SHADER, "fragment")
    program = ShaderProgram(vertex_shader, fragment_shader)

    first_frame_index = int(frame_indices[0])
    mesh_frame = smpl_body.pose(
        motion.poses[first_frame_index], motion.trans[first_frame_index]
    )
    vertices = np.ascontiguousarray(mesh_frame.vertices, dtype=np.float32)
    indices = np.ascontiguousarray(smpl_body.model.faces.reshape(-1), dtype=np.uint32)
    vertex_list = program.vertex_list_indexed(
        vertices.shape[0],
        gl.GL_TRIANGLES,
        indices,
        position=("f", vertices.reshape(-1)),
    )

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glDepthFunc(gl.GL_LEQUAL)
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glClearColor(0.94, 0.95, 0.97, 1.0)

    program.use()
    program["base_color"] = (0.80, 0.52, 0.36, 1.0)
    program.stop()

    trajectory = motion.trans[frame_indices]
    if fixed_camera:
        fixed_target = trajectory.mean(axis=0)
        horizontal_span = float(
            np.max(np.linalg.norm(trajectory[:, (0, 2)] - fixed_target[(0, 2)], axis=1))
        )
    else:
        fixed_target = mesh_frame.joints[0].astype(np.float64)
        horizontal_span = 0.0

    state: dict[str, Any] = {
        "sequence_position": 0.0,
        "current_sequence_index": 0,
        "playing": True,
        "speed": float(speed),
        "yaw": 0.0,
        "pitch": 0.08,
        "distance": max(view_radius * 2.5, horizontal_span * 1.3 + view_radius * 2.0),
        "target": fixed_target.copy(),
        "wireframe": False,
        "draw_count": 0,
        "fps_count": 0,
        "fps_started": time.perf_counter(),
        "measured_fps": 0.0,
        "pixel_standard_deviation": None,
        "closed": False,
    }

    frame_step = 1 if frame_indices.size < 2 else int(frame_indices[1] - frame_indices[0])
    sequence_rate = motion.mocap_framerate / frame_step

    def update_projection() -> None:
        window.projection = Mat4.perspective_projection(
            window.aspect_ratio,
            z_near=0.01,
            z_far=1000.0,
            fov=45.0,
        )

    def update_camera() -> None:
        target = np.asarray(state["target"], dtype=np.float64)
        yaw = float(state["yaw"])
        pitch = float(state["pitch"])
        distance = float(state["distance"])
        eye = target + distance * np.array(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ]
        )
        window.view = Mat4.look_at(
            Vec3(*eye.tolist()),
            Vec3(*target.tolist()),
            Vec3(0.0, 1.0, 0.0),
        )

    def upload_sequence_frame(sequence_index: int) -> None:
        source_frame = int(frame_indices[sequence_index])
        posed = smpl_body.pose(motion.poses[source_frame], motion.trans[source_frame])
        updated_vertices = np.ascontiguousarray(posed.vertices, dtype=np.float32)
        vertex_list.position[:] = updated_vertices.reshape(-1)
        if not fixed_camera:
            state["target"] = posed.joints[0].astype(np.float64)
        state["current_sequence_index"] = sequence_index
        update_camera()

    def update_caption() -> None:
        now = time.perf_counter()
        elapsed = now - float(state["fps_started"])
        if elapsed >= 0.5:
            state["measured_fps"] = float(state["fps_count"]) / elapsed
            state["fps_count"] = 0
            state["fps_started"] = now
        sequence_index = int(state["current_sequence_index"])
        source_frame = int(frame_indices[sequence_index])
        playback_state = "PLAY" if state["playing"] else "PAUSE"
        window.set_caption(
            f"SMPL GPU | {playback_state} | frame {source_frame}/{motion.frame_count - 1} | "
            f"{state['speed']:.2f}x | {state['measured_fps']:.1f} FPS | {renderer}"
        )

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
        if auto_close_after_frames is not None and state["draw_count"] >= auto_close_after_frames:
            gl.glFinish()
            color_buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = color_buffer.get_image_data().get_data("RGB", window.width * 3)
            state["pixel_standard_deviation"] = float(
                np.frombuffer(image_data, dtype=np.uint8).std()
            )
            # Closing inside on_draw destroys the context before Pyglet can
            # swap buffers. Defer it until the current draw/flip completes.
            pyglet.clock.schedule_once(lambda _dt: window.close(), 0.0)

    @window.event
    def on_resize(new_width: int, new_height: int) -> Any:
        update_projection()
        return pyglet.event.EVENT_HANDLED

    @window.event
    def on_key_press(symbol: int, modifiers: int) -> None:
        if symbol == key.SPACE:
            state["playing"] = not state["playing"]
        elif symbol == key.LEFT:
            state["playing"] = False
            state["sequence_position"] = max(0.0, state["sequence_position"] - 1.0)
            upload_sequence_frame(int(state["sequence_position"]))
        elif symbol == key.RIGHT:
            state["playing"] = False
            state["sequence_position"] = min(
                float(len(frame_indices) - 1), state["sequence_position"] + 1.0
            )
            upload_sequence_frame(int(state["sequence_position"]))
        elif symbol == key.HOME:
            state["sequence_position"] = 0.0
            upload_sequence_frame(0)
        elif symbol == key.END:
            state["sequence_position"] = float(len(frame_indices) - 1)
            upload_sequence_frame(len(frame_indices) - 1)
        elif symbol in (key.PLUS, key.NUM_ADD, key.EQUAL):
            state["speed"] = min(8.0, state["speed"] * 1.25)
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            state["speed"] = max(0.1, state["speed"] / 1.25)
        elif symbol == key.W:
            state["wireframe"] = not state["wireframe"]
        elif symbol == key.R:
            state["yaw"] = 0.0
            state["pitch"] = 0.08
            state["distance"] = max(
                view_radius * 2.5, horizontal_span * 1.3 + view_radius * 2.0
            )
        elif symbol == key.ESCAPE:
            window.close()
        update_caption()

    @window.event
    def on_mouse_drag(
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int,
    ) -> None:
        if buttons & mouse.LEFT:
            state["yaw"] -= dx * 0.01
            state["pitch"] = float(
                np.clip(state["pitch"] + dy * 0.01, -1.35, 1.35)
            )
            update_camera()

    @window.event
    def on_mouse_scroll(x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        state["distance"] = float(
            np.clip(state["distance"] * math.exp(-scroll_y * 0.12), 0.25, 100.0)
        )
        update_camera()

    @window.event
    def on_close() -> None:
        state["closed"] = True
        pyglet.clock.unschedule(tick)

    def tick(delta_time: float) -> None:
        if state["playing"]:
            new_position = state["sequence_position"] + (
                delta_time * sequence_rate * state["speed"]
            )
            if new_position >= len(frame_indices):
                if repeat:
                    new_position %= len(frame_indices)
                else:
                    new_position = float(len(frame_indices) - 1)
                    state["playing"] = False
            state["sequence_position"] = new_position
            sequence_index = int(new_position)
            if sequence_index != state["current_sequence_index"]:
                upload_sequence_frame(sequence_index)
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
        "renderer": renderer,
        "vendor": vendor,
        "opengl_version": gl_version,
        "draw_count": int(state["draw_count"]),
        "measured_fps": float(state["measured_fps"]),
        "pixel_standard_deviation": state["pixel_standard_deviation"],
    }
