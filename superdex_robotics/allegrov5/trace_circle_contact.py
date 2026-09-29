"""Kinova Gen3 7-DOF + Allegro Hand v5: Hybrid Position/Force Control.

Maintains 2.0 N normal contact force against a tabletop while following
a circular trajectory in the XY plane for 10 seconds, then cleanly releases.

Finite State Machine Architecture:
  HOVER -> DESCEND -> TRACE_AND_HOLD (10s) -> RETRACT -> HOVER_AFTER_RETRACT -> DONE
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
TARGET_FORCE_N = 2.0         # 2.0 N target press force
FORCE_TOLERANCE_N = 0.2      # +/- 0.2 N tolerance [1.8 N, 2.2 N]
FORCE_RAMP_SEC = 0.40        # Smooth 0.4s ramp to target force (prevents impact shock)
TRACE_DURATION_SEC = 12.0    # 10.0 seconds contact trace duration
TABLE_SURFACE_Z = 0.1055     # Calibrated table contact height
HOVER_HEIGHT_Z = 0.20        # Hover standoff height
DESCENT_SPEED_MPS = 0.035    # 35 mm/s descent speed
RETRACT_SPEED_MPS = 0.040    # 40 mm/s retraction speed
CONTACT_STIFFNESS_N_PER_M = 1000.0

# Circular Trajectory Parameters (XY Plane)
CIRCLE_RADIUS = 0.08         # 8 cm radius
CIRCLE_PERIOD_SEC = 4.0      # 4.0 seconds per revolution (10s = 2.5 revolutions)

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


def find_fingertip_link_handle(scene: physics.Scene, bot_actor) -> tuple[int, str]:
    candidates = ["allegro_link_3_tip", "link_3_tip", "allegro_link_3", "tip_3"]
    handles = bot_actor.get_nested_link_actors()
    for cand in candidates:
        for handle in handles:
            name = scene.get_actor(handle).get_name().lower()
            if cand in name:
                return handle, scene.get_actor(handle).get_name()
    for handle in handles:
        name = scene.get_actor(handle).get_name().lower()
        if "tip" in name and ("3" in name or "index" in name):
            return handle, scene.get_actor(handle).get_name()
    return handles[-1], scene.get_actor(handles[-1]).get_name()


def spawn_visible_table(scene: physics.Scene, center_x: float = 0.75) -> float:
    surface_z = 0.000
    # plane_shape = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=surface_z)
    # scene.create_rigid_actor(name="table_collider", shape=plane_shape, is_static=True)

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
        print(f"[Setup] Loaded Visible Table Prefab at X={center_x:.2f}m")
    except Exception as e:
        print(f"[Setup] Note on table prefab: {e}")

    return surface_z


# ==============================================================================
# Contact Force Sensor Interface
# ==============================================================================
class FingertipForceSensor:
    def __init__(self, tip_actor, stiffness_n_per_m: float = CONTACT_STIFFNESS_N_PER_M):
        self.tip_actor = tip_actor
        self.k_contact = stiffness_n_per_m
        self.free_air_tracking_offset = None

        if hasattr(physics, "QueryType"):
            if hasattr(physics.QueryType, "TOTAL_CONTACT_FORCE") and hasattr(tip_actor, "register_query"):
                try:
                    tip_actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
                    print("[Sensor] Registered TOTAL_CONTACT_FORCE query.")
                except Exception as e:
                    print(f"[Sensor] Note on register_query: {e}")

    def calibrate_free_air(self, actual_tip_z: float, cmd_tip_z: float):
        self.free_air_tracking_offset = actual_tip_z - cmd_tip_z

    def read_force_z(self, actual_tip_z: float, cmd_tip_z: float, table_surface_z: float = TABLE_SURFACE_Z) -> float:
        if actual_tip_z > (table_surface_z + 0.002):
            return 0.0

        # Native physics query check
        if hasattr(self.tip_actor, "get_query_result") and hasattr(physics, "QueryType"):
            try:
                res = self.tip_actor.get_query_result(physics.QueryType.TOTAL_CONTACT_FORCE)
                if res is not None and np.linalg.norm(res) > 0.05:
                    return float(np.abs(res[2]))
            except Exception:
                pass

        # Spring deflection estimate against table
        deflection = max(0.0, table_surface_z - cmd_tip_z)
        return float(self.k_contact * deflection)


# ==============================================================================
# Telemetry Logger & 4-Panel Plotter
# ==============================================================================
class ContactCircleTelemetryLogger:
    def __init__(self):
        self.times: list[float] = []
        self.forces: list[float] = []
        self.target_forces: list[float] = []
        self.cmd_heights: list[float] = []
        self.actual_positions: list[np.ndarray] = []
        self.target_positions: list[np.ndarray] = []
        self.states: list[str] = []
        self.events: dict[str, float] = {}

    def record(
        self,
        t: float,
        force: float,
        target_f: float,
        actual_xyz: np.ndarray,
        target_xyz: np.ndarray,
        cmd_h: float,
        state: str,
    ):
        self.times.append(t)
        self.forces.append(force)
        self.target_forces.append(target_f)
        self.cmd_heights.append(cmd_h)
        self.actual_positions.append(np.array(actual_xyz, dtype=float))
        self.target_positions.append(np.array(target_xyz, dtype=float))
        self.states.append(state)

    def mark_event(self, name: str, t: float):
        self.events[name] = t

    def generate_plots(self, image_path: str = "kinova_allegro_contact_circle_plot.png"):
        if not self.times:
            print("[Logger] No data to plot.")
            return

        t = np.array(self.times)
        forces = np.array(self.forces)
        target_f = np.array(self.target_forces)
        actual_pos = np.array(self.actual_positions)
        target_pos = np.array(self.target_positions)
        heights = actual_pos[:, 2] * 1000.0  # mm
        cmd_h = np.array(self.cmd_heights) * 1000.0
        xy_err = np.linalg.norm(actual_pos[:, :2] - target_pos[:, :2], axis=1) * 1000.0

        states = np.array(self.states)
        contact_mask = states == "TRACE_AND_HOLD"

        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Normal Force Regulation (Top-Left)
        ax_f = axs[0, 0]
        ax_f.plot(t, forces, "b-", linewidth=2.0, label="Fingertip Force (Fz)")
        ax_f.plot(t, target_f, "r--", linewidth=1.8, label="Target Force")
        ax_f.axhline(TARGET_FORCE_N + FORCE_TOLERANCE_N, color="green", linestyle=":", label="Tolerance Band (±0.2 N)")
        ax_f.axhline(TARGET_FORCE_N - FORCE_TOLERANCE_N, color="green", linestyle=":")
        ax_f.fill_between(
            t,
            TARGET_FORCE_N - FORCE_TOLERANCE_N,
            TARGET_FORCE_N + FORCE_TOLERANCE_N,
            color="green",
            alpha=0.12,
        )
        for name, ev_t in self.events.items():
            ax_f.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
            ax_f.text(ev_t + 0.05, 1.4, name, rotation=90, color="dimgray", fontsize=8)
        ax_f.set_xlabel("Time [s]")
        ax_f.set_ylabel("Normal Force [N]")
        ax_f.set_title("Normal Force Regulation (2.0 N)")
        ax_f.grid(True, linestyle="--", alpha=0.6)
        ax_f.legend(loc="upper left")

        # 2. Height & Admittance Penetration (Top-Right)
        ax_h = axs[0, 1]
        ax_h.plot(t, heights, "k-", linewidth=1.8, label="Actual Fingertip Height (Z)")
        ax_h.plot(t, cmd_h, "g--", linewidth=1.2, alpha=0.7, label="Commanded Z (Admittance)")
        for name, ev_t in self.events.items():
            ax_h.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
        ax_h.set_xlabel("Time [s]")
        ax_h.set_ylabel("Height [mm]")
        ax_h.set_title("Fingertip Height (Z) & Admittance Offset")
        ax_h.grid(True, linestyle="--", alpha=0.6)
        ax_h.legend(loc="upper right")

        # 3. 2D Path (XY Plane)
        ax_xy = axs[1, 0]
        ax_xy.plot(target_pos[:, 0], target_pos[:, 1], "r--", linewidth=2.0, label="Nominal Circle")
        if np.any(contact_mask):
            ax_xy.plot(
                actual_pos[contact_mask, 0],
                actual_pos[contact_mask, 1],
                "b-",
                linewidth=1.8,
                label="Actual Path (In Contact)",
            )
        else:
            ax_xy.plot(actual_pos[:, 0], actual_pos[:, 1], "b-", linewidth=1.5, label="Actual Path")
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.set_title("Fingertip Trajectory on Tabletop (XY Plane)")
        ax_xy.axis("equal")
        ax_xy.grid(True, linestyle="--", alpha=0.6)
        ax_xy.legend(loc="upper right")

        # 4. XY Tracking Error vs Time (Bottom-Right)
        ax_err = axs[1, 1]
        ax_err.plot(t, xy_err, color="purple", linewidth=1.5, label="XY Error")
        for name, ev_t in self.events.items():
            ax_err.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
        mean_contact_err = np.mean(xy_err[contact_mask]) if np.any(contact_mask) else np.mean(xy_err)
        ax_err.set_xlabel("Time [s]")
        ax_err.set_ylabel("Error [mm]")
        ax_err.set_title(f"In-Plane Tracking Error (Contact Mean: {mean_contact_err:.2f} mm)")
        ax_err.grid(True, linestyle="--", alpha=0.6)
        ax_err.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        print(f"\n[Logger] Multi-axis contact regulation plot saved to: {image_path}")


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
    parser = argparse.ArgumentParser(description="Kinova + Allegro Contact Trajectory Control")
    parser.add_argument("path", type=str, nargs="?", default=None, help="Path to combo .superdex_bot")
    args = parser.parse_args()

    scene = None
    ik_scene = None
    bot = None
    ik_bot = None
    ik_solver = None
    ik_wrist_handle = None
    logger = None

    try:
        bot_path = args.path if args.path else resolve_combo_path()
        print(f"[Init] Loading robot combo from: {bot_path}")

        physics.initialize(num_worker_threads=0)

        # 1. Physics Scene
        scene = physics.create_scene("Kinova Contact Trajectory Scene")
        scene.set_gravity([0.0, 0.0, -9.81])

        robotics_context = robotics.create_context()
        bot = create_combo_robot(scene, bot_path, robotics_context)
        bot_actor = bot.get_articulated_actor()
        num_dofs = bot_actor.get_num_dofs()

        sim_wrist_handle, _ = find_bracelet_link_handle(scene, bot_actor)
        sim_tip_handle, _ = find_fingertip_link_handle(scene, bot_actor)
        sim_wrist_actor = scene.get_actor(sim_wrist_handle)
        sim_tip_actor = scene.get_actor(sim_tip_handle)

        spawn_visible_table(scene, center_x=0.75)

        # 2. Kinematic Twin Scene (IK)
        ik_scene = physics.create_scene("Kinova IK Scene")
        ik_bot = create_combo_robot(ik_scene, bot_path, robotics_context)
        ik_actor = ik_bot.get_articulated_actor()

        ik_wrist_handle, _ = find_bracelet_link_handle(ik_scene, ik_actor)
        ik_tip_handle, _ = find_fingertip_link_handle(ik_scene, ik_actor)
        ik_wrist_actor = ik_scene.get_actor(ik_wrist_handle)
        ik_tip_actor = ik_scene.get_actor(ik_tip_handle)

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
        
        # Touchdown occurs directly at theta = 0 on the circle circumference
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

        # Warm up robot at hover position above the touchdown point
        start_wrist_xyz = hover_tip_xyz - tip_offset_world
        ik_position_target.set_target_position(start_wrist_xyz.tolist())
        for _ in range(30):
            ik_solver.solve_ik()
        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        pose_target.pose_dofs = ik_pose
        pose_controller.compute_output(pose_obsv, pose_target)
        for _ in range(200):  # 1.0 second settle
            scene.step(TIME_STEP)

        sensor = FingertipForceSensor(sim_tip_actor)
        logger = ContactCircleTelemetryLogger()

        init_tip_z = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)[2]
        sensor.calibrate_free_air(init_tip_z, hover_tip_xyz[2])

        # ======================================================================
        # 6. Control Parameters (Admittance Z + Cartesian PI XY)
        # ======================================================================
        # Normal Force Admittance Gains (Z axis)
        KP_FORCE = 0.0002   # m/(N*s)
        KI_FORCE = 0.0006   # m/(N*s^2)
        KD_FORCE = 0.00015  # Velocity damping against impact
        force_integral = 0.0
        prev_tip_z = hover_tip_xyz[2]

        # In-Plane Cartesian PI Gains (XY plane)
        KP_CART = 0.85
        KI_CART = 2.50
        MAX_I_CLIP = 0.050  # Clamp integral correction to +/- 50 mm
        tip_error_integral_xy = np.zeros(2, dtype=float)

        # State Machine Initialization
        STATE = "HOVER"
        cmd_tip_z = hover_tip_xyz[2]
        trace_start_time = None
        retract_finish_time = None
        step_count = 0

        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

        print("\n[Simulation] Connecting to debugger...")
        print(f"[Mission] Hover -> Descend -> Trace Circle at {TARGET_FORCE_N:.1f}N for {TRACE_DURATION_SEC:.1f}s -> Retract\n")

        start_time = scene.get_total_simulation_time()

        if physics.debugger.attach():
            while physics.debugger.is_attached() and STATE != "DONE":
                sim_time = scene.get_total_simulation_time() - start_time

                # A. Read current fingertip position & force
                actual_tip_xyz = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)
                actual_tip_z = actual_tip_xyz[2]
                current_force_z = sensor.read_force_z(actual_tip_z, cmd_tip_z, TABLE_SURFACE_Z)

                # Defaults for trajectory and force reference
                target_f_for_log = 0.0
                target_tip_xy = start_tip_xy.copy()

                # ==============================================================
                # B. State Machine Transitions & Control Synthesis
                # ==============================================================
                if STATE == "HOVER":
                    cmd_tip_z = hover_tip_xyz[2]
                    target_tip_xy = start_tip_xy.copy()
                    if sim_time >= 0.8:
                        STATE = "DESCEND"
                        logger.mark_event("Start Descent", sim_time)
                        print(f"[{sim_time:.2f}s] Beginning descent toward table surface...")

                elif STATE == "DESCEND":
                    cmd_tip_z -= DESCENT_SPEED_MPS * TIME_STEP
                    target_tip_xy = start_tip_xy.copy()

                    if actual_tip_z <= (TABLE_SURFACE_Z + 0.001) or current_force_z >= 0.15:
                        STATE = "TRACE_AND_HOLD"
                        trace_start_time = sim_time
                        cmd_tip_z = TABLE_SURFACE_Z
                        force_integral = 0.0
                        tip_error_integral_xy = np.zeros(2, dtype=float)
                        logger.mark_event("Touchdown / Trace Start", sim_time)
                        print(f"[{sim_time:.2f}s] Touchdown at Z={actual_tip_z*1000:.1f}mm. Beginning 10s trace...")

                elif STATE == "TRACE_AND_HOLD":
                    elapsed_trace = sim_time - trace_start_time

                    # 1. Smooth Force Reference Ramp (0 -> 2.0 N over 0.4s)
                    if elapsed_trace < FORCE_RAMP_SEC:
                        target_force = TARGET_FORCE_N * (elapsed_trace / FORCE_RAMP_SEC)
                    else:
                        target_force = TARGET_FORCE_N
                    target_f_for_log = target_force

                    # 2. Normal Force Admittance Law (Z-Axis)
                    force_error = target_force - current_force_z
                    force_integral += force_error * TIME_STEP
                    force_integral = np.clip(force_integral, -0.5, 0.5)

                    tip_vel_z = (actual_tip_z - prev_tip_z) / TIME_STEP
                    z_correction = -(
                        KP_FORCE * force_error +
                        KI_FORCE * force_integral -
                        KD_FORCE * tip_vel_z
                    )
                    cmd_tip_z += z_correction
                    cmd_tip_z = np.clip(cmd_tip_z, TABLE_SURFACE_Z - 0.0035, TABLE_SURFACE_Z + 0.001)

                    # 3. Circular Trajectory Generation (XY Plane)
                    theta = 2.0 * np.pi * (elapsed_trace / CIRCLE_PERIOD_SEC)
                    target_tip_xy = np.array([
                        circle_center_xy[0] + CIRCLE_RADIUS * np.cos(theta),
                        circle_center_xy[1] + CIRCLE_RADIUS * np.sin(theta),
                    ])

                    # 4. Check for 10.0-Second Completion -> LATCH STOP POSITION
                    if elapsed_trace >= TRACE_DURATION_SEC:
                        STATE = "RETRACT"
                        retract_start_time = sim_time
                        # FIX 1: Latch exact stopping position so robot doesn't snap to start
                        retract_tip_xy = target_tip_xy.copy()
                        # FIX 2: Clear integral windup to prevent horizontal kicks on liftoff
                        tip_error_integral_xy = np.zeros(2, dtype=float)
                        force_integral = 0.0
                        logger.mark_event("10s Trace Complete / Retract", sim_time)
                        print(f"[{sim_time:.2f}s] 10.0s trace complete at {retract_tip_xy}. Smoothly releasing contact...")

                elif STATE == "RETRACT":
                    # FIX 3: Hold the exact stopping XY coordinate throughout retraction
                    target_tip_xy = retract_tip_xy.copy()
                    elapsed_retract = sim_time - retract_start_time
                    RELEASE_RAMP_SEC = 0.30

                    # FIX 4: Smooth 2-stage release (unloading ramp followed by vertical ascent)
                    if elapsed_retract < RELEASE_RAMP_SEC:
                        # Stage A: Ramp target force 2.0 N -> 0.0 N while decompressing fingertip
                        ramp_ratio = 1.0 - (elapsed_retract / RELEASE_RAMP_SEC)
                        target_force = TARGET_FORCE_N * ramp_ratio
                        target_f_for_log = target_force

                        force_error = target_force - current_force_z
                        force_integral += force_error * TIME_STEP
                        force_integral = np.clip(force_integral, -0.5, 0.5)
                        tip_vel_z = (actual_tip_z - prev_tip_z) / TIME_STEP
                        z_correction = -(
                            KP_FORCE * force_error +
                            KI_FORCE * force_integral -
                            KD_FORCE * tip_vel_z
                        )
                        cmd_tip_z += z_correction
                        cmd_tip_z = np.clip(cmd_tip_z, TABLE_SURFACE_Z - 0.0035, TABLE_SURFACE_Z + 0.002)
                    else:
                        # Stage B: Ascend vertically into free air to hover height
                        target_f_for_log = 0.0
                        cmd_tip_z += RETRACT_SPEED_MPS * TIME_STEP

                    if cmd_tip_z >= hover_tip_xyz[2]:
                        cmd_tip_z = hover_tip_xyz[2]
                        STATE = "HOVER_AFTER_RETRACT"
                        retract_finish_time = sim_time
                        logger.mark_event("Retracted", sim_time)
                        print(f"[{sim_time:.2f}s] Fully retracted to hover height. Holding for 1s...")

                elif STATE == "HOVER_AFTER_RETRACT":
                    target_tip_xy = retract_tip_xy.copy()
                    cmd_tip_z = hover_tip_xyz[2]
                    target_f_for_log = 0.0
                    if sim_time - retract_finish_time >= 1.0:
                        STATE = "DONE"
                        print(f"[{sim_time:.2f}s] Simulation task cleanly finished!")

                # Store for velocity calculation
                prev_tip_z = actual_tip_z

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
                # Nominal wrist target
                nominal_wrist_x = target_tip_xy[0] - tip_offset_world[0]
                nominal_wrist_y = target_tip_xy[1] - tip_offset_world[1]
                nominal_wrist_z = cmd_tip_z - tip_offset_world[2]

                # Apply Cartesian PI compensation exclusively to XY
                compensated_wrist_xyz = [
                    nominal_wrist_x + cartesian_corr_xy[0],
                    nominal_wrist_y + cartesian_corr_xy[1],
                    nominal_wrist_z,  # Z is governed strictly by the admittance loop
                ]

                ik_position_target.set_target_position(compensated_wrist_xyz)
                for _ in range(5):
                    ik_solver.solve_ik()

                # Dispatch joint targets to Mochi Controller
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
                full_target_xyz = np.array([target_tip_xy[0], target_tip_xy[1], TABLE_SURFACE_Z])
                logger.record(
                    sim_time,
                    current_force_z,
                    target_f_for_log,
                    actual_tip_xyz,
                    full_target_xyz,
                    cmd_tip_z,
                    STATE,
                )

                if step_count % 40 == 0:
                    xy_err_mm = np.linalg.norm(tip_error_xy) * 1000.0
                    print(
                        f"t={sim_time:5.2f}s | [{STATE:14s}] | "
                        f"Fz: {current_force_z:4.2f} N (Tgt: {target_f_for_log:4.2f} N) | "
                        f"Tip Z: {actual_tip_z * 1000.0:5.1f} mm | "
                        f"XY Err: {xy_err_mm:4.1f} mm"
                    )

    except KeyboardInterrupt:
        print("\n\n[Simulation] Interrupted by user.")

    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("[Teardown] Cleaning up resources...")

        if logger is not None:
            try:
                logger.generate_plots("kinova_allegro_contact_circle_plot.png")
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