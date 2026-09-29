"""Kinova Gen3 7-DOF + Allegro Hand v5: High-Precision Closed-Loop Fingertip Circular Tracking.

Fixes:
1. Workspace alignment with touch_table.py (Table at X=0.75m, circle center at X=root+0.48m).
2. Closed-Loop Cartesian PI error correction to eliminate the 30mm steady-state tracking offset.
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
# Configuration & Targets
# ==============================================================================
COMBO_BOT_PATH = (
    "assets/bots/arm_hand_combos/"
    "kinova_allegro_right/kinova_allegro_right.superdex_bot"
)

# Objective weights for IK solver [N/m] and [Nm/rad]
IK_POSITION_WEIGHT = 1.0e4
IK_ROTATION_WEIGHT = 1.0e2
EE_DOWN_ROTATION_VECTOR = [0.0, 0.0, 0.0]

# Control rate (200 Hz = 5 ms step)
CONTROL_RATE_HZ = 200.0
TIME_STEP = 1.0 / CONTROL_RATE_HZ

# Pointed finger configuration for the 16 Allegro hand joints
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
    """Resolves Kinova + Allegro combo .superdex_bot path."""
    if Path(COMBO_BOT_PATH).exists():
        return COMBO_BOT_PATH
    try:
        resolved = resolve_asset(
            "bots/arm_hand_combos/kinova_allegro_right/kinova_allegro_right.superdex_bot"
        )
        return str(resolved)
    except Exception:
        return COMBO_BOT_PATH


def create_combo_robot(
    scene: physics.Scene,
    bot_path: str,
    robotics_context: robotics.RoboticsContext,
) -> robotics.Bot:
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False
    return robotics.create_bot(scene, bot_prefab, robotics_context)


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
    """Spawns the exact table model and collider used in touch_table.py."""
    surface_z = 0.000
    plane_shape = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=surface_z)
    scene.create_rigid_actor(name="table_collider", shape=plane_shape, is_static=True)

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
# Telemetry Logger
# ==============================================================================
class FingertipTelemetryLogger:
    def __init__(self):
        self.times: list[float] = []
        self.targets: list[np.ndarray] = []
        self.actuals: list[np.ndarray] = []

    def record(self, t: float, target_pos: list[float] | np.ndarray, actual_pos: np.ndarray) -> None:
        self.times.append(t)
        self.targets.append(np.array(target_pos, dtype=float))
        self.actuals.append(np.array(actual_pos, dtype=float))

    def generate_plots(self, image_path: str = "kinova_allegro_tracking_plot.png") -> None:
        if not self.times:
            print("[Logger] No data recorded.")
            return

        t = np.array(self.times)
        target = np.array(self.targets)
        actual = np.array(self.actuals)

        error_3d = np.linalg.norm(actual - target, axis=1) * 1000.0
        error_xy = np.linalg.norm(actual[:, :2] - target[:, :2], axis=1) * 1000.0
        error_z = np.abs(actual[:, 2] - target[:, 2]) * 1000.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 2D Path (XY Plane)
        ax1.plot(target[:, 0], target[:, 1], "r--", label="Target Fingertip Path", linewidth=2.0)
        ax1.plot(actual[:, 0], actual[:, 1], "b-", label="Actual Fingertip", linewidth=1.5, alpha=0.85)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("Allegro Index Fingertip Trajectory (XY Plane)")
        ax1.axis("equal")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 2. Tracking Error vs Time
        ax2.plot(t, error_3d, label="3D Total Error", color="crimson", linewidth=1.5)
        ax2.plot(t, error_xy, label="XY Plane Error", color="orange", linestyle="--")
        ax2.plot(t, error_z, label="Z Height Error", color="purple", linestyle=":")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Error [mm]")
        ax2.set_title(f"Fingertip Tracking Error (Mean: {np.mean(error_3d):.2f} mm, Final: {error_3d[-1]:.2f} mm)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        print(f"\n[Logger] Tracking plot saved to: {image_path}")


# ==============================================================================
# Controller Parameter Synthesizer
# ==============================================================================
def get_or_create_combo_pose_params(
    bot_path: str, num_dofs: int
) -> robotics.ControllerMochiArticulatedPoseParams:
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
# Main Execution Loop
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Kinova + Allegro Closed-Loop Circle Tracking")
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
        print(f"[Init] Loading Kinova + Allegro combo from: {bot_path}")

        physics.initialize(num_worker_threads=0)

        # 1. Simulated Dynamic Scene
        scene = physics.create_scene("Kinova + Allegro Simulated Scene")
        scene.set_gravity([0.0, 0.0, -9.81])

        # Spawn Table at X=0.75m (matches touch_table.py)
        spawn_visible_table(scene, center_x=0.75)

        robotics_context = robotics.create_context()
        bot = create_combo_robot(scene, bot_path, robotics_context)
        bot_actor = bot.get_articulated_actor()
        num_dofs = bot_actor.get_num_dofs()

        sim_wrist_handle, _ = find_bracelet_link_handle(scene, bot_actor)
        sim_tip_handle, _ = find_fingertip_link_handle(scene, bot_actor)
        sim_wrist_actor = scene.get_actor(sim_wrist_handle)
        sim_tip_actor = scene.get_actor(sim_tip_handle)

        # 2. Kinematic Twin Scene (IK Solver)
        ik_scene = physics.create_scene("Kinova + Allegro IK Solver Scene")
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

        # 3. Mochi Controller Setup
        pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        pose_params = get_or_create_combo_pose_params(bot_path, num_dofs)
        pose_controller.set_params(pose_params)
        pose_controller.initialize(True)

        pose_obsv = robotics.ControllerMochiArticulatedPoseObsv()
        pose_target = robotics.ControllerMochiArticulatedPoseTarget()
        pose_target.world_from_root = bot_actor.get_root_transform()

        # ======================================================================
        # 4. Workspace & Circle Definition (Aligned with touch_table.py)
        # ======================================================================
        root_pos = np.asarray(bot_actor.get_root_transform().translation, dtype=float)
        TABLE_SURFACE_Z = 0.1055  # Physical tabletop height from touch_table.py

        circle_radius = 0.08      # 8 cm radius
        circle_period = 4.0       # 4 seconds per revolution
        circle_center_tip = np.array([
            root_pos[0] + 0.48,   # Exactly matches touch_table.py contact_xy[0]!
            root_pos[1],          # Centered along Y
            TABLE_SURFACE_Z + 0.04  # 40 mm standoff above table
        ])

        print(f"[Trajectory] Circle Center: {circle_center_tip}, Radius: {circle_radius*1000:.1f} mm")

        # ======================================================================
        # 5. Measure Kinematic Offset & Settle Arm at Circle Start
        # ======================================================================
        start_tip_xyz = np.array([
            circle_center_tip[0] + circle_radius,
            circle_center_tip[1],
            circle_center_tip[2],
        ])

        # Step A: Solve initial downward wrist posture in IK
        ik_position_target.set_target_position([start_tip_xyz[0], start_tip_xyz[1], start_tip_xyz[2] + 0.22])
        for _ in range(40):
            ik_solver.solve_ik()

        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        # Step B: Read wrist-to-tip kinematic offset directly
        p_wrist_ik = np.asarray(ik_wrist_actor.get_root_transform().translation, dtype=float)
        p_tip_ik = np.asarray(ik_tip_actor.get_root_transform().translation, dtype=float)
        tip_offset_world = p_tip_ik - p_wrist_ik
        print(f"[Geometry] Wrist-to-Tip Offset: {tip_offset_world} (norm: {np.linalg.norm(tip_offset_world)*1000:.1f} mm)")

        # Step C: Send simulated robot to start position and settle
        nominal_start_wrist = start_tip_xyz - tip_offset_world
        ik_position_target.set_target_position(nominal_start_wrist.tolist())
        for _ in range(30):
            ik_solver.solve_ik()
        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        pose_target.pose_dofs = ik_pose
        pose_controller.compute_output(pose_obsv, pose_target)
        for _ in range(200):  # 1.0 second settle
            scene.step(TIME_STEP)

        # ======================================================================
        # 6. Closed-Loop Tracking Controller Parameters
        # ======================================================================
        # Cartesian Proportional-Integral feedback gains to drive tracking error to zero
        KP_CART = 0.85          # Proportional tracking boost
        KI_CART = 2.50          # Integral action eliminates the ~30mm static offset
        MAX_I_CLIP = 0.050      # Clamp integral correction to +/- 50 mm

        tip_error_integral = np.zeros(3, dtype=float)
        logger = FingertipTelemetryLogger()
        step_count = 0

        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

        print("\n[Simulation] Connecting to debugger...")
        print("[Tracking] Starting Closed-Loop Fingertip Trajectory Control...\n")

        start_time = scene.get_total_simulation_time()

        if physics.debugger.attach():
            while physics.debugger.is_attached():
                sim_time = scene.get_total_simulation_time() - start_time

                # A. Desired fingertip coordinate on the circle
                theta = 2.0 * np.pi * sim_time / circle_period
                target_tip_xyz = np.array([
                    circle_center_tip[0] + circle_radius * np.cos(theta),
                    circle_center_tip[1] + circle_radius * np.sin(theta),
                    circle_center_tip[2],
                ])

                # B. Read current actual fingertip position
                actual_tip_xyz = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)

                # C. Closed-Loop Tip Tracking Error (PI Compensation)
                tip_error = target_tip_xyz - actual_tip_xyz
                tip_error_integral += tip_error * TIME_STEP
                tip_error_integral = np.clip(tip_error_integral, -MAX_I_CLIP, MAX_I_CLIP)

                # D. Compensated Wrist Target: Nominal + Closed-Loop Feedback
                cartesian_correction = (KP_CART * tip_error) + (KI_CART * tip_error_integral)
                nominal_wrist_xyz = target_tip_xyz - tip_offset_world
                compensated_wrist_xyz = nominal_wrist_xyz + cartesian_correction

                # E. Tightly converged IK solve (5 iterations per control step)
                ik_position_target.set_target_position(compensated_wrist_xyz.tolist())
                for _ in range(5):
                    ik_solver.solve_ik()

                # F. Dispatch joint targets to Mochi Controller
                ik_actor.get_articulated_pose(ik_pose)
                for j in range(16):
                    ik_pose[7 + j] = HAND_POINTED_QPOS[j]

                pose_target.pose_dofs = ik_pose
                pose_controller.compute_output(pose_obsv, pose_target)
                scene.step(TIME_STEP)

                # G. Telemetry & Live Diagnostics
                step_count += 1
                logger.record(sim_time, target_tip_xyz, actual_tip_xyz)

                if step_count % 50 == 0:
                    err_mm = np.linalg.norm(tip_error) * 1000.0
                    print(
                        f"Step {step_count:04d} | t={sim_time:5.2f}s | "
                        f"Target: [{target_tip_xyz[0]:.3f}, {target_tip_xyz[1]:.3f}] | "
                        f"Actual: [{actual_tip_xyz[0]:.3f}, {actual_tip_xyz[1]:.3f}] | "
                        f"Tip Error: {err_mm:4.2f} mm"
                    )

    except KeyboardInterrupt:
        print("\n\n[Simulation] Ctrl+C detected! Stopping simulation loop...")

    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("[Teardown] Cleaning up resources...")

        if logger is not None:
            try:
                logger.generate_plots("kinova_allegro_tracking_plot.png")
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