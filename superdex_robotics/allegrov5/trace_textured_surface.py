"""Kinova Gen3 7-DOF + Allegro Hand v5: Fixed-Height Circular Motion on Textured Surface.

1. Descends smoothly until contact reaches 2.0 N (touchdown at ~106.5 mm).
2. Locks the commanded height (fixed_cmd_z) without stepping/bouncing.
3. Holds the fixed Z height while tracing an 8 cm radius circle in XY for 12 seconds.
   When the fingertip encounters hills, contact force naturally rises and exceeds
   the 2.0 +/- 0.2 N tolerance band.
4. Retracts smoothly back to hover height.
5. Visualizes a 4-panel plot including:
   - Normal force with 2.0 +/- 0.2 N tolerance band.
   - Fixed height vs terrain elevation.
   - XY trajectory overlaid directly on a 2D topographic terrain heatmap.
   - Tracking error vs time.

Finite State Machine:
  HOVER -> DESCEND -> TRACE_AND_HOLD (12s, Constant Z) -> RETRACT -> HOVER_AFTER_RETRACT -> DONE
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset, resolve_asset_root
from superdex.physics.utils.scene_helpers import find_actor

# ==============================================================================
# Configuration & Control Targets
# ==============================================================================
COMBO_BOT_PATH = (
    "assets/bots/arm_hand_combos/"
    "kinova_allegro_right/kinova_allegro_right.superdex_bot"
)

# Control rate (200 Hz = 5 ms step)
CONTROL_RATE_HZ = 200.0
TIME_STEP = 1.0 / CONTROL_RATE_HZ

# IK weights
IK_POSITION_WEIGHT = 1.0e4
IK_ROTATION_WEIGHT = 1.0e2
EE_DOWN_ROTATION_VECTOR = [0.0, 0.0, 0.0]  # Points index finger straight DOWN

# Force & Contact Parameters
TARGET_FORCE_N = 2.0         # 2.0 N nominal target contact force
FORCE_TOLERANCE_N = 0.2      # +/- 0.2 N tolerance band [1.8 N, 2.2 N]
TRACE_DURATION_SEC = 12.0    # 12.0 seconds contact trace duration (3 revolutions)
HOVER_HEIGHT_Z  = 0.2000     # 20 cm standoff height (well clear of terrain)
DESCENT_SPEED_MPS = 0.035    # 35 mm/s descent speed
RETRACT_SPEED_MPS = 0.040    # 40 mm/s retraction speed
CONTACT_STIFFNESS_N_PER_M = 1000.0

# Circular Trajectory Parameters (XY Plane)
CIRCLE_RADIUS = 0.08         # 8 cm radius
CIRCLE_PERIOD_SEC = 4.0      # 4.0 seconds per revolution (12s = 3 revolutions)

# Pointed finger configuration for 16 Allegro hand joints
HAND_POINTED_QPOS = [
    0.0, 0.05, 0.07, 0.02,   # Index (straight)
    0.0, 1.55, 1.55, 1.30,   # Middle (curled)
    0.0, 1.55, 1.55, 1.30,   # Ring (curled)
    1.20, 0.60, 1.20, 0.80,  # Thumb (tucked)
]


# ==============================================================================
# Helpers: Asset & Link Resolution
# ==============================================================================
def resolve_combo_path() -> str:
    if Path(COMBO_BOT_PATH).exists():
        return COMBO_BOT_PATH
    try:
        resolved = resolve_asset(
            "bots/arm_hand_combos/kinova_allegro_right/kinova_allegro_right.superdex_bot"
        )
        return str(resolved)
    except Exception:
        return COMBO_BOT_PATH


def create_combo_robot(scene: physics.Scene, bot_path: str, ctx: robotics.RoboticsContext) -> robotics.Bot:
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False
    return robotics.create_bot(scene, bot_prefab, ctx)


def find_bracelet_link_handle(scene: physics.Scene, bot_actor) -> tuple[int, str]:
    candidates = ["bracelet_link", "tool_frame", "flange", "link_7", "arm_link_7"]
    handles = bot_actor.get_nested_link_actors()
    for cand in candidates:
        for handle in handles:
            name = scene.get_actor(handle).get_name().lower()
            if cand in name:
                return handle, scene.get_actor(handle).get_name()
    return handles[7], scene.get_actor(handles[7]).get_name()


def find_fingertip_actors(scene: physics.Scene, bot_actor) -> tuple[any, list]:
    """Finds tip link for kinematics and collision links for contact force querying."""
    handles = bot_actor.get_nested_link_actors()
    tip_actor = None
    contact_actors = []

    for h in handles:
        actor = scene.get_actor(h)
        name = actor.get_name().lower()
        if "allegro_link_3" in name or "link_3" in name or "index" in name or "tip" in name:
            contact_actors.append(actor)
        if "link_3_tip" in name or "allegro_link_3_tip" in name:
            tip_actor = actor

    if tip_actor is None:
        tip_actor = contact_actors[-1] if contact_actors else scene.get_actor(handles[-1])

    return tip_actor, contact_actors


# ==============================================================================
# Solid Lipschitz Textured Surface Construction
# ==============================================================================
def create_lipschitz_solid_plate(
    center_xy=(0.48, 0.0),
    size_xy=(1.2, 0.8),
    grid_res=60,
    base_z=0.0784,
    num_centers=32,
    thickness=0.040,
    amplitude=0.012,
    sigma=0.045,
    seed=103,
) -> tuple[physics.ShapeHandle, callable]:
    rng = np.random.default_rng(seed)
    cx, cy = center_xy
    sx, sy = size_xy

    centers = np.column_stack([
        rng.uniform(cx - sx / 2, cx + sx / 2, num_centers),
        rng.uniform(cy - sy / 2, cy + sy / 2, num_centers),
    ])
    weights = rng.uniform(-1.0, 1.0, num_centers)

    def landscape_h(coords_2d):
        coords_2d = np.atleast_2d(coords_2d)
        diffs = coords_2d[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diffs**2, axis=-1)
        raw = np.sum(weights * np.exp(-dist_sq / (2.0 * sigma**2)), axis=-1)
        raw -= np.mean(raw)
        h = amplitude * (raw / max(np.max(np.abs(raw)), 1e-6))
        dist = np.linalg.norm(coords_2d - [cx, cy], axis=-1)
        w = np.clip(1.0 - (dist / (0.45 * sx)) ** 2, 0.0, 1.0) ** 2
        return h * w

    xs = np.linspace(cx - sx / 2, cx + sx / 2, grid_res)
    ys = np.linspace(cy - sy / 2, cy + sy / 2, grid_res)
    GX, GY = np.meshgrid(xs, ys)
    coords_2d = np.stack([GX.ravel(), GY.ravel()], axis=-1)

    h_top = landscape_h(coords_2d)
    top_verts = np.column_stack([GX.ravel(), GY.ravel(), base_z + h_top])
    bot_verts = np.column_stack([GX.ravel(), GY.ravel(), np.full_like(h_top, base_z - thickness)])

    all_verts = np.vstack([top_verts, bot_verts])
    N = len(top_verts)

    faces = []
    for j in range(grid_res - 1):
        for i in range(grid_res - 1):
            v1, v2 = j * grid_res + i, j * grid_res + i + 1
            v3, v4 = (j + 1) * grid_res + i, (j + 1) * grid_res + i + 1
            faces.extend([(v1, v2, v4), (v1, v4, v3)])
            faces.extend([(N + v1, N + v4, N + v2), (N + v1, N + v3, N + v4)])

    for i in range(grid_res - 1):
        faces.extend([(i, N + i, i + 1), (i + 1, N + i, N + i + 1)])
        t1, t2 = (grid_res - 1) * grid_res + i, (grid_res - 1) * grid_res + i + 1
        faces.extend([(t1, t2, N + t1), (t2, N + t2, N + t1)])
    for j in range(grid_res - 1):
        t1, t2 = j * grid_res, (j + 1) * grid_res
        faces.extend([(t1, t2, N + t1), (t2, N + t2, N + t1)])
        t1, t2 = j * grid_res + (grid_res - 1), (j + 1) * grid_res + (grid_res - 1)
        faces.extend([(t1, N + t1, t2), (t2, N + t1, N + t2)])

    faces = np.array(faces, dtype=np.int32)
    coords_flat = np.ascontiguousarray(all_verts.ravel(), dtype=np.float32)
    faces_flat = np.ascontiguousarray(faces.ravel(), dtype=np.int32)

    try:
        mesh_shape = physics.create_mesh_shape(
            nodes_per_element=3, coordinates=coords_flat, connectivity=faces_flat
        )
    except TypeError:
        d_coords = physics.DynamicArrayReal(len(coords_flat))
        for idx, val in enumerate(coords_flat):
            d_coords[idx] = float(val)

        int_type = getattr(physics, "DynamicArrayInt", getattr(physics, "DynamicArrayInt32", list))
        d_faces = int_type(len(faces_flat)) if int_type is not list else faces_flat.tolist()
        if int_type is not list:
            for idx, val in enumerate(faces_flat):
                d_faces[idx] = int(val)

        mesh_shape = physics.create_mesh_shape(
            nodes_per_element=3, coordinates=d_coords, connectivity=d_faces
        )

    return mesh_shape, landscape_h


def spawn_visible_table(scene: physics.Scene, center_x: float = 0.75) -> callable:
    try:
        table_prefab_path = str(resolve_asset("table/table.mochi_scene"))
        physics.prefab.add_to_scene(
            prefab_path=table_prefab_path,
            root_path=str(resolve_asset_root("table/table.mochi_scene")),
            scene=scene,
            params=physics.prefab.PrefabParams(
                name="tablePrefab",
                translation=[float(center_x), 0.0, -0.80],
            ),
        )
    except Exception as e:
        print(f"[Setup] Note on table prefab: {e}")

    plate_shape, landscape_fn = create_lipschitz_solid_plate()
    scene.create_rigid_actor(name="textured_solid_plate", shape=plate_shape, is_static=True)
    print("[Setup] Spawned solid watertight textured contact block.")
    return landscape_fn


# ==============================================================================
# Robust Multi-Link Contact Force Sensor Interface
# ==============================================================================
class FingertipForceSensor:
    def __init__(self, contact_actors: list, stiffness_n_per_m: float = CONTACT_STIFFNESS_N_PER_M):
        self.contact_actors = contact_actors
        self.k_contact = stiffness_n_per_m
        self.free_air_offset = 0.0
        self.query_handles = {}
        self.target_qt = None

        if hasattr(physics, "QueryType"):
            for cand in ["TOTAL_CONTACT_FORCE", "TotalContactForce", "CONTACT_FORCE"]:
                if hasattr(physics.QueryType, cand):
                    self.target_qt = getattr(physics.QueryType, cand)
                    break

            if self.target_qt is not None:
                for actor in self.contact_actors:
                    if hasattr(actor, "register_query"):
                        try:
                            qh = actor.register_query(self.target_qt)
                            self.query_handles[actor] = qh
                        except Exception:
                            pass

    def calibrate_free_air(self, actual_tip_z: float, cmd_tip_z: float):
        self.free_air_offset = actual_tip_z - cmd_tip_z

    def read_force_z(self, actual_tip_z: float, cmd_tip_z: float) -> float:
        # 1. Native physics query across all finger contact actors (allegro_link_3)
        if self.target_qt is not None:
            max_native_f = 0.0
            for actor in self.contact_actors:
                qh = self.query_handles.get(actor)
                res = None
                if qh is not None and hasattr(actor, "get_query_result"):
                    try:
                        res = actor.get_query_result(qh)
                    except Exception:
                        pass
                if res is None and hasattr(actor, "get_query_result"):
                    try:
                        res = actor.get_query_result(self.target_qt)
                    except Exception:
                        pass

                if res is not None:
                    res_arr = np.asarray(res, dtype=float)
                    if res_arr.size >= 3:
                        f_z = abs(float(res_arr[2]))
                        if f_z > max_native_f:
                            max_native_f = f_z
            if max_native_f > 0.05:
                return max_native_f

        # 2. Native actor getters fallback
        max_getter_f = 0.0
        for actor in self.contact_actors:
            for getter_name in ["get_total_contact_force", "get_total_contact_force_world", "get_contact_force"]:
                if hasattr(actor, getter_name):
                    try:
                        res = getattr(actor, getter_name)()
                        if res is not None and len(res) >= 3:
                            f_z = abs(float(res[2]))
                            if f_z > max_getter_f:
                                max_getter_f = f_z
                    except Exception:
                        pass
        if max_getter_f > 0.05:
            return max_getter_f

        # 3. Physical Deflection Fallback
        raw_deflection = (actual_tip_z - cmd_tip_z) - self.free_air_offset
        deflection = max(0.0, raw_deflection)
        return float(self.k_contact * deflection)

    def teardown(self):
        for actor, qh in self.query_handles.items():
            if hasattr(actor, "cancel_query"):
                try:
                    actor.cancel_query(qh)
                except Exception:
                    pass


# ==============================================================================
# Telemetry Logger & 4-Panel Plotter
# ==============================================================================
class ContactCircleTelemetryLogger:
    def __init__(self):
        self.times: list[float] = []
        self.forces: list[float] = []
        self.actual_positions: list[np.ndarray] = []
        self.terrain_elevations: list[float] = []
        self.target_positions: list[np.ndarray] = []
        self.states: list[str] = []
        self.events: dict[str, float] = {}

    def record(
        self,
        t: float,
        force: float,
        actual_xyz: np.ndarray,
        target_xyz: np.ndarray,
        terrain_z: float,
        state: str,
    ):
        self.times.append(t)
        self.forces.append(force)
        self.actual_positions.append(np.array(actual_xyz, dtype=float))
        self.target_positions.append(np.array(target_xyz, dtype=float))
        self.terrain_elevations.append(terrain_z)
        self.states.append(state)

    def mark_event(self, name: str, t: float):
        self.events[name] = t

    def generate_plots(
        self,
        image_path: str = "kinova_allegro_fixed_height_textured_plot.png",
        landscape_fn: callable = None,
        fixed_contact_z: float = None,
        h_touchdown: float = 0.0,
        show_contours: bool = True,
    ):
        if not self.times:
            print("[Logger] No data to plot.")
            return

        t = np.array(self.times)
        forces = np.array(self.forces)
        actual_pos = np.array(self.actual_positions)
        target_pos = np.array(self.target_positions)
        terrain_z = np.array(self.terrain_elevations) * 1000.0  # mm
        heights = actual_pos[:, 2] * 1000.0                    # mm
        xy_err = np.linalg.norm(actual_pos[:, :2] - target_pos[:, :2], axis=1) * 1000.0

        states = np.array(self.states)
        contact_mask = states == "TRACE_AND_HOLD"

        fig, axs = plt.subplots(2, 2, figsize=(16, 11))

        # 1. Normal Contact Force vs Tolerance Band (Top-Left)
        ax_f = axs[0, 0]
        ax_f.plot(t, forces, "b-", linewidth=2.0, label="Fingertip Force (Fz)")
        ax_f.axhline(TARGET_FORCE_N, color="red", linestyle="--", linewidth=1.8, label=f"Target Force ({TARGET_FORCE_N:.1f} N)")
        ax_f.axhline(TARGET_FORCE_N + FORCE_TOLERANCE_N, color="green", linestyle=":", label="Tolerance Band (±0.2 N)")
        ax_f.axhline(TARGET_FORCE_N - FORCE_TOLERANCE_N, color="green", linestyle=":")
        ax_f.fill_between(
            t,
            TARGET_FORCE_N - FORCE_TOLERANCE_N,
            TARGET_FORCE_N + FORCE_TOLERANCE_N,
            color="green",
            alpha=0.15,
        )
        for name, ev_t in self.events.items():
            ax_f.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
            ax_f.text(ev_t + 0.05, max(forces) * 0.75, name, rotation=90, color="dimgray", fontsize=8)
        ax_f.set_xlabel("Time [s]")
        ax_f.set_ylabel("Normal Force [N]")
        ax_f.set_title("Normal Contact Force vs Tolerance Band (2.0 ± 0.2 N)")
        ax_f.grid(True, linestyle="--", alpha=0.6)
        ax_f.legend(loc="upper left")

        # 2. Fixed Fingertip Height vs Undulating Terrain (Top-Right)
        ax_h = axs[0, 1]
        ax_h.plot(t, heights, "k-", linewidth=2.0, label="Fingertip Height Z (Fixed in Contact)")
        ax_h.plot(t, terrain_z, "g--", linewidth=1.5, alpha=0.8, label="Surface Elevation Z(x, y)")
        for name, ev_t in self.events.items():
            ax_h.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("Height [mm]")
        ax_h.set_title("Constant Fingertip Height vs Undulating Terrain")
        ax_h.grid(True, linestyle="--", alpha=0.6)
        ax_h.legend(loc="upper right")

        # 3. 2D Trajectory with Topographic Terrain Overlay (Bottom-Left)
        ax_xy = axs[1, 0]
        if landscape_fn is not None:
            if show_contours:
                cx = float(np.mean(target_pos[:, 0]))
                cy = float(np.mean(target_pos[:, 1]))
                margin = 1.45 * CIRCLE_RADIUS
                gx = np.linspace(cx - margin, cx + margin, 120)
                gy = np.linspace(cy - margin, cy + margin, 120)
                GX, GY = np.meshgrid(gx, gy)
                coords_flat = np.column_stack([GX.ravel(), GY.ravel()])

                h_grid = landscape_fn(coords_flat).reshape(GX.shape)
                if fixed_contact_z is not None:
                    Z_map_mm = (fixed_contact_z + (h_grid - h_touchdown)) * 1000.0
                    cbar_label = "Surface Elevation Z [mm]"
                else:
                    Z_map_mm = h_grid * 1000.0
                    cbar_label = "Terrain Undulation [mm]"

                cf = ax_xy.contourf(GX, GY, Z_map_mm, levels=25, cmap="terrain", alpha=0.55)
                cs = ax_xy.contour(GX, GY, Z_map_mm, levels=10, colors="dimgray", alpha=0.35, linewidths=0.6)
                ax_xy.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
                cbar = fig.colorbar(cf, ax=ax_xy, fraction=0.046, pad=0.04)
                cbar.set_label(cbar_label, fontsize=9)

        ax_xy.plot(target_pos[:, 0], target_pos[:, 1], "r--", linewidth=2.0, label="Nominal Circle", zorder=3)
        if np.any(contact_mask):
            ax_xy.plot(
                actual_pos[contact_mask, 0],
                actual_pos[contact_mask, 1],
                "b-",
                linewidth=2.0,
                label="Actual Path (In Contact)",
                zorder=4,
            )
        else:
            ax_xy.plot(actual_pos[:, 0], actual_pos[:, 1], "b-", linewidth=1.5, label="Actual Path", zorder=4)

        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.set_title("Fingertip Trajectory over Textured Surface (XY Plane)")
        ax_xy.axis("equal")
        ax_xy.grid(True, linestyle="--", alpha=0.4)
        ax_xy.legend(loc="upper right")

        # 4. In-Plane Tracking Error (Bottom-Right)
        ax_err = axs[1, 1]
        ax_err.plot(t, xy_err, color="purple", linewidth=1.5, label="XY Error")
        for name, ev_t in self.events.items():
            ax_err.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
        mean_contact_err = np.mean(xy_err[contact_mask]) if np.any(contact_mask) else np.mean(xy_err)
        ax_err.set_xlabel("Time [s]")
        ax_err.set_ylabel("Error [mm]")
        ax_err.set_title(f"In-Plane Tracking Error (Mean: {mean_contact_err:.2f} mm)")
        ax_err.grid(True, linestyle="--", alpha=0.6)
        ax_err.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        print(f"\n[Logger] Multi-axis plot with terrain contour saved to: {image_path}")


# ==============================================================================
# Controller Parameter Synthesizer
# ==============================================================================
def get_or_create_combo_pose_params(bot_path: str, num_dofs: int) -> robotics.ControllerMochiArticulatedPoseParams:
    bot_dir = Path(bot_path).parent
    out_dir = bot_dir / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl_file = out_dir / f"{Path(bot_path).stem}_pose.superdex_controller"

    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
    num_links = len(bot_prefab.links)

    kinova_joint_stiffness = [600.0, 600.0, 500.0, 450.0, 200.0, 150.0, 100.0]
    kinova_joint_damping = [50.0, 50.0, 40.0, 35.0, 20.0, 15.0, 10.0]
    hand_joint_stiffness = 15.0
    hand_joint_damping = 1.0

    joint_tracking = []
    arm_idx = 0

    for link in bot_prefab.links:
        lname = link.name.lower()
        if "base" in lname or "palm" in lname or "tip" in lname or "flange" in lname or "tool" in lname:
            kp, kd = 0.0, 0.0
        elif "allegro" in lname:
            kp, kd = hand_joint_stiffness, hand_joint_damping
        elif arm_idx < 7:
            kp = kinova_joint_stiffness[arm_idx]
            kd = kinova_joint_damping[arm_idx]
            arm_idx += 1
        else:
            kp, kd = 0.0, 0.0

        joint_tracking.append({"damping": float(kd), "saturation": -1.0, "stiffness": float(kp)})

    zero_tracking = [{"damping": 0.0, "saturation": -1.0, "stiffness": 0.0} for _ in range(num_links)]
    config = {
        "poseControllerParams": {
            "jointTracking": joint_tracking,
            "linkPosTracking": zero_tracking,
            "linkRotTracking": zero_tracking,
        }
    }
    with open(ctrl_file, "w") as f:
        json.dump(config, f, indent=2)

    return robotics.ControllerMochiArticulatedPoseParams.load_from_file(str(ctrl_file))


# ==============================================================================
# Main Simulation Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Kinova + Allegro Fixed-Height Textured Trajectory")
    parser.add_argument("path", type=str, nargs="?", default=None, help="Path to combo .superdex_bot")
    args = parser.parse_args()

    scene = None
    ik_scene = None
    bot = None
    ik_bot = None
    ik_solver = None
    ik_wrist_handle = None
    sensor = None
    logger = None
    landscape_fn = None
    fixed_contact_z = None
    h_touchdown = 0.0

    try:
        bot_path = args.path if args.path else resolve_combo_path()
        print(f"[Init] Loading robot combo from: {bot_path}")

        physics.initialize(num_worker_threads=0)

        # 1. Physics Scene
        scene = physics.create_scene("Kinova Fixed-Height Textured Scene")
        scene.set_gravity([0.0, 0.0, -9.81])

        robotics_context = robotics.create_context()
        bot = create_combo_robot(scene, bot_path, robotics_context)
        bot_actor = bot.get_articulated_actor()
        num_dofs = bot_actor.get_num_dofs()

        sim_wrist_handle, _ = find_bracelet_link_handle(scene, bot_actor)
        sim_tip_actor, sim_contact_actors = find_fingertip_actors(scene, bot_actor)
        sim_wrist_actor = scene.get_actor(sim_wrist_handle)

        # Spawn solid watertight textured surface block
        landscape_fn = spawn_visible_table(scene, center_x=0.75)

        # 2. Kinematic Twin Scene (IK)
        ik_scene = physics.create_scene("Kinova IK Scene")
        ik_bot = create_combo_robot(ik_scene, bot_path, robotics_context)
        ik_actor = ik_bot.get_articulated_actor()

        ik_wrist_handle, _ = find_bracelet_link_handle(ik_scene, ik_actor)
        ik_tip_actor, _ = find_fingertip_actors(ik_scene, ik_actor)
        ik_wrist_actor = ik_scene.get_actor(ik_wrist_handle)

        ik_solver = physics.experimental.create_ik_solver(ik_scene)
        ik_position_target = ik_solver.create_position_target(
            ik_wrist_handle, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], IK_POSITION_WEIGHT
        )
        ik_solver.create_rotation_target(
            ik_wrist_handle, [0.0, 0.0, 0.0], EE_DOWN_ROTATION_VECTOR, IK_ROTATION_WEIGHT
        )
        ik_pose = physics.DynamicArrayReal(num_dofs)

        # 3. Controller Setup
        pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        pose_params = get_or_create_combo_pose_params(bot_path, num_dofs)
        pose_controller.set_params(pose_params)
        pose_controller.initialize(True)

        pose_obsv = robotics.ControllerMochiArticulatedPoseObsv()
        pose_target = robotics.ControllerMochiArticulatedPoseTarget()
        pose_target.world_from_root = bot_actor.get_root_transform()

        # 4. Trajectory Geometry Setup
        root_pos = np.asarray(bot_actor.get_root_transform().translation, dtype=float)
        circle_center_xy = np.array([root_pos[0] + 0.48, root_pos[1]])
        start_tip_xy = np.array([circle_center_xy[0] + CIRCLE_RADIUS, circle_center_xy[1]])
        hover_tip_xyz = np.array([start_tip_xy[0], start_tip_xy[1], HOVER_HEIGHT_Z])

        print(f"[Trajectory] Circle Center: {circle_center_xy}, Radius: {CIRCLE_RADIUS*1000:.1f} mm")
        print(f"[Trajectory] Touchdown Point: {start_tip_xy}")

        # 5. Measure Kinematic Offset in Downward Pose
        ik_position_target.set_target_position([hover_tip_xyz[0], hover_tip_xyz[1], hover_tip_xyz[2] + 0.22])
        for _ in range(40):
            ik_solver.solve_ik()

        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        p_wrist_ik = np.asarray(ik_wrist_actor.get_root_transform().translation, dtype=float)
        p_tip_ik = np.asarray(ik_tip_actor.get_root_transform().translation, dtype=float)
        tip_offset_world = p_tip_ik - p_wrist_ik

        # Warm up robot at hover position
        start_wrist_xyz = hover_tip_xyz - tip_offset_world
        ik_position_target.set_target_position(start_wrist_xyz.tolist())
        for _ in range(30):
            ik_solver.solve_ik()
        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        pose_target.pose_dofs = ik_pose
        pose_controller.compute_output(pose_obsv, pose_target)
        for _ in range(200):  # Settle
            scene.step(TIME_STEP)

        sensor = FingertipForceSensor(sim_contact_actors)
        logger = ContactCircleTelemetryLogger()

        init_tip_z = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)[2]
        sensor.calibrate_free_air(init_tip_z, hover_tip_xyz[2])

        # In-Plane Cartesian PI Gains (XY plane only)
        KP_CART = 0.30
        KI_CART = 0.50
        MAX_I_CLIP = 0.015
        tip_error_integral_xy = np.zeros(2, dtype=float)

        # State Machine Initialization
        STATE = "HOVER"
        cmd_tip_z = hover_tip_xyz[2]
        fixed_cmd_z = None
        trace_start_time = None
        retract_finish_time = None
        step_count = 0

        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

        print("\n[Simulation] Connecting to debugger...")
        print(f"[Mission] Hover -> Descend to {TARGET_FORCE_N:.1f}N -> Latch Height -> Trace Circle at Constant Z -> Retract\n")

        start_time = scene.get_total_simulation_time()

        if physics.debugger.attach():
            while physics.debugger.is_attached() and STATE != "DONE":
                sim_time = scene.get_total_simulation_time() - start_time

                # A. Read current fingertip position & force
                actual_tip_xyz = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)
                actual_tip_z = actual_tip_xyz[2]
                current_force_z = sensor.read_force_z(actual_tip_z, cmd_tip_z)

                target_tip_xy = start_tip_xy.copy()

                # Dynamic surface elevation for telemetry
                if STATE == "TRACE_AND_HOLD":
                    terrain_delta_h = float(landscape_fn(actual_tip_xyz[:2])[0]) - h_touchdown
                    current_surface_z = fixed_contact_z + terrain_delta_h
                elif fixed_contact_z is not None:
                    current_surface_z = fixed_contact_z
                else:
                    current_surface_z = actual_tip_z

                # ==============================================================
                # B. State Machine Transitions
                # ==============================================================
                if STATE == "HOVER":
                    cmd_tip_z = hover_tip_xyz[2]
                    target_tip_xy = start_tip_xy.copy()
                    if sim_time >= 0.8:
                        STATE = "DESCEND"
                        logger.mark_event("Start Descent", sim_time)
                        print(f"[{sim_time:.2f}s] Descending toward textured surface...")

                elif STATE == "DESCEND":
                    # Slow down as contact approaches
                    if current_force_z < 0.30:
                        speed = DESCENT_SPEED_MPS
                    else:
                        speed = DESCENT_SPEED_MPS * max(0.20, 1.0 - (current_force_z / TARGET_FORCE_N))

                    cmd_tip_z -= speed * TIME_STEP
                    target_tip_xy = start_tip_xy.copy()

                    # TOUCHDOWN: Trigger at 2.0 N contact
                    if current_force_z >= TARGET_FORCE_N:
                        STATE = "TRACE_AND_HOLD"
                        trace_start_time = sim_time

                        # LOCK COMMANDED DEPTH (Preserves 2.0 N baseline, avoids bounce!)
                        fixed_cmd_z = cmd_tip_z
                        fixed_contact_z = actual_tip_z
                        h_touchdown = float(landscape_fn(start_tip_xy)[0])

                        tip_error_integral_xy = np.zeros(2, dtype=float)
                        logger.mark_event("Touchdown / Height Locked", sim_time)
                        print(f"[{sim_time:.2f}s] Touchdown at Fz = {current_force_z:.2f} N!")
                        print(f"[{sim_time:.2f}s] Contact height LOCKED at Z = {fixed_contact_z * 1000.0:.1f} mm. Tracing circle at constant commanded Z...")

                elif STATE == "TRACE_AND_HOLD":
                    elapsed_trace = sim_time - trace_start_time

                    # HOLD CONSTANT COMMANDED HEIGHT (Hills exceed tolerance band)
                    cmd_tip_z = fixed_cmd_z

                    # Circular Trajectory Generation (XY Plane)
                    theta = 2.0 * np.pi * (elapsed_trace / CIRCLE_PERIOD_SEC)
                    target_tip_xy = np.array([
                        circle_center_xy[0] + CIRCLE_RADIUS * np.cos(theta),
                        circle_center_xy[1] + CIRCLE_RADIUS * np.sin(theta),
                    ])

                    if elapsed_trace >= TRACE_DURATION_SEC:
                        STATE = "RETRACT"
                        retract_start_time = sim_time
                        retract_tip_xy = target_tip_xy.copy()
                        tip_error_integral_xy = np.zeros(2, dtype=float)
                        logger.mark_event("Trace Complete / Retract", sim_time)
                        print(f"[{sim_time:.2f}s] Trace complete. Smoothly retracting...")

                elif STATE == "RETRACT":
                    target_tip_xy = retract_tip_xy.copy()
                    cmd_tip_z += RETRACT_SPEED_MPS * TIME_STEP

                    if cmd_tip_z >= hover_tip_xyz[2]:
                        cmd_tip_z = hover_tip_xyz[2]
                        STATE = "HOVER_AFTER_RETRACT"
                        retract_finish_time = sim_time
                        logger.mark_event("Retracted", sim_time)
                        print(f"[{sim_time:.2f}s] Retracted to hover height. Holding for 1.0s...")

                elif STATE == "HOVER_AFTER_RETRACT":
                    target_tip_xy = retract_tip_xy.copy()
                    cmd_tip_z = hover_tip_xyz[2]
                    if sim_time - retract_finish_time >= 1.0:
                        STATE = "DONE"
                        print(f"[{sim_time:.2f}s] Mission Complete!")

                # ==============================================================
                # C. Closed-Loop Cartesian Correction (XY Only)
                # ==============================================================
                tip_error_xy = target_tip_xy - actual_tip_xyz[:2]
                tip_error_integral_xy += tip_error_xy * TIME_STEP
                tip_error_integral_xy = np.clip(tip_error_integral_xy, -MAX_I_CLIP, MAX_I_CLIP)
                cartesian_corr_xy = (KP_CART * tip_error_xy) + (KI_CART * tip_error_integral_xy)

                # ==============================================================
                # D. Target Synthesis & IK Solve
                # ==============================================================
                nominal_wrist_x = target_tip_xy[0] - tip_offset_world[0]
                nominal_wrist_y = target_tip_xy[1] - tip_offset_world[1]
                nominal_wrist_z = cmd_tip_z - tip_offset_world[2]

                compensated_wrist_xyz = [
                    nominal_wrist_x + cartesian_corr_xy[0],
                    nominal_wrist_y + cartesian_corr_xy[1],
                    nominal_wrist_z,
                ]

                ik_position_target.set_target_position(compensated_wrist_xyz)
                for _ in range(15):
                    ik_solver.solve_ik()

                ik_actor.get_articulated_pose(ik_pose)
                for j in range(16):
                    ik_pose[7 + j] = HAND_POINTED_QPOS[j]

                pose_target.pose_dofs = ik_pose
                pose_controller.compute_output(pose_obsv, pose_target)
                scene.step(TIME_STEP)

                # ==============================================================
                # E. Telemetry & Console Diagnostics
                # ==============================================================
                step_count += 1
                full_target_xyz = np.array([target_tip_xy[0], target_tip_xy[1], current_surface_z])
                logger.record(
                    sim_time,
                    current_force_z,
                    actual_tip_xyz,
                    full_target_xyz,
                    current_surface_z,
                    STATE,
                )

                if step_count % 40 == 0:
                    xy_err_mm = np.linalg.norm(tip_error_xy) * 1000.0
                    exceeded_str = " (EXCEEDED!)" if (STATE == "TRACE_AND_HOLD" and abs(current_force_z - TARGET_FORCE_N) > FORCE_TOLERANCE_N) else ""
                    print(
                        f"t={sim_time:5.2f}s | [{STATE:14s}] | "
                        f"Fz: {current_force_z:4.2f} N (Tgt: {TARGET_FORCE_N:4.2f} ± {FORCE_TOLERANCE_N:.1f} N){exceeded_str:12s} | "
                        f"Tip Z: {actual_tip_z * 1000.0:5.1f} mm | "
                        f"Terrain Z: {current_surface_z * 1000.0:5.1f} mm | "
                        f"XY Err: {xy_err_mm:4.1f} mm"
                    )

    except KeyboardInterrupt:
        print("\n\n[Simulation] Interrupted by user.")

    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("[Teardown] Cleaning up resources...")

        if sensor is not None:
            sensor.teardown()

        if logger is not None:
            try:
                logger.generate_plots(
                    image_path="kinova_allegro_fixed_height_textured_plot.png",
                    landscape_fn=landscape_fn,
                    fixed_contact_z=fixed_contact_z,
                    h_touchdown=h_touchdown,
                )
            except Exception as e:
                print(f"[Teardown] Note during plot generation: {e}")

        try:
            if ik_solver is not None and ik_wrist_handle is not None:
                ik_solver.clear_position_target(ik_wrist_handle)
                ik_solver.clear_rotation_target(ik_wrist_handle)
            if ik_scene is not None and ik_bot is not None:
                robotics.destroy_bot(ik_scene, ik_bot)
            if ik_solver is not None:
                physics.experimental.destroy_ik_solver(ik_solver)
        except Exception as e:
            print(f"[Teardown] Note during IK teardown: {e}")

        try:
            if scene is not None and bot is not None:
                robotics.destroy_bot(scene, bot)
            physics.shutdown()
            print("[Teardown] Physics engine cleanly shutdown.")
        except Exception as e:
            print(f"[Teardown] Note during physics shutdown: {e}")


if __name__ == "__main__":
    main()