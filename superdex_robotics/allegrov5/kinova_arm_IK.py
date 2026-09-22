"""Kinova Gen3 7-DOF: IK + Mochi Articulated Pose Control (Circle Tracking)

Follows the Meta Project SuperDex dual-scene architecture:
1. Kinematic Twin (ik_scene): Solves quasistatic IK for position + downward rotation.
2. Simulated Robot (scene): Tracks solved joint angles using Mochi's native implicit
   pose controller.

Since no pre-tuned .superdex_controller file exists for the Kinova Gen3, this script
synthesizes per-joint stiffness and damping values tailored to the Kinova's link inertias.
"""

from __future__ import annotations

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
# Direct path or relative asset path for Kinova Gen3
KINOVA_BOT_PATH = "/home/radamsfr/HSL/project_superdex/assets/bots/arms/kinova_gen3/GEN3-7DOF-NOVISION_FOR_URDF_ARM_V12.superdex_bot"

# Objective weights for IK solver [N/m] and [Nm/rad]
IK_POSITION_WEIGHT = 1.0e4
IK_ROTATION_WEIGHT = 1.0e2

# Tool flange pointing straight down (axis-angle vector: half-turn about world X)
EE_DOWN_ROTATION_VECTOR = [np.pi, 0.0, 0.0]

# Control rate (200 Hz = 5 ms step)
CONTROL_RATE_HZ = 200.0
TIME_STEP = 1.0 / CONTROL_RATE_HZ


# ==============================================================================
# Helpers: Asset & Link Resolution
# ==============================================================================
def resolve_kinova_path() -> str:
    """Resolves Kinova .superdex_bot path via direct path or asset resolver."""
    if Path(KINOVA_BOT_PATH).exists():
        return KINOVA_BOT_PATH
    try:
        resolved = resolve_asset("bots/arms/kinova_gen3/GEN3-7DOF-NOVISION_FOR_URDF_ARM_V12.superdex_bot")
        return str(resolved)
    except Exception:
        return KINOVA_BOT_PATH


def create_arm(
    scene: physics.Scene,
    bot_path: str,
    robotics_context: robotics.RoboticsContext,
) -> robotics.Bot:
    """Loads Kinova arm with gravity disabled on links to eliminate sag."""
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
    for i in range(len(bot_prefab.links)):
        bot_prefab.links[i].has_gravity = False
    return robotics.create_bot(scene, bot_prefab, robotics_context)


def find_ee_link_handle(scene: physics.Scene, bot_actor) -> tuple[int, str]:
    """Finds the tool/flange link handle by searching candidate naming conventions."""
    candidates = ["tool_frame", "end_effector", "flange", "link_7", "arm_link_7"]
    handles = bot_actor.get_nested_link_actors()
    for cand in candidates:
        for handle in handles:
            name = scene.get_actor(handle).get_name().lower()
            if cand in name:
                return handle, scene.get_actor(handle).get_name()
    # Default to the final distal link
    last_handle = handles[-1]
    return last_handle, scene.get_actor(last_handle).get_name()


class KinovaTelemetryLogger:
    """Logs end-effector positions and errors, and exports plots."""

    def __init__(self):
        self.times: list[float] = []
        self.targets: list[np.ndarray] = []
        self.actuals: list[np.ndarray] = []

    def record(self, t: float, target_pos: list[float] | np.ndarray, actual_pos: np.ndarray) -> None:
        self.times.append(t)
        self.targets.append(np.array(target_pos, dtype=float))
        self.actuals.append(np.array(actual_pos, dtype=float))

    def generate_plots(self, image_path: str = "kinova_tracking_plot.png") -> None:
        if not self.times:
            print("[Logger] No data recorded.")
            return

        t = np.array(self.times)
        target = np.array(self.targets)
        actual = np.array(self.actuals)

        # Compute errors in millimeters
        error_3d = np.linalg.norm(actual - target, axis=1) * 1000.0
        error_xy = np.linalg.norm(actual[:, :2] - target[:, :2], axis=1) * 1000.0
        error_z = np.abs(actual[:, 2] - target[:, 2]) * 1000.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 2D Path (XY Plane)
        ax1.plot(target[:, 0], target[:, 1], "r--", label="Target Path", linewidth=2.0)
        ax1.plot(actual[:, 0], actual[:, 1], "b-", label="Actual EEF", linewidth=1.5, alpha=0.85)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_title("Kinova End-Effector Trajectory (XY Plane)")
        ax1.axis("equal")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # 2. Tracking Error vs Time
        ax2.plot(t, error_3d, label="3D Total Error", color="crimson", linewidth=1.5)
        ax2.plot(t, error_xy, label="XY Plane Error", color="orange", linestyle="--")
        ax2.plot(t, error_z, label="Z Height Error", color="purple", linestyle=":")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Error [mm]")
        ax2.set_title(f"End-Effector Tracking Error (Mean: {np.mean(error_3d):.2f} mm)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        print(f"\n[Logger] Tracking plot saved to: {image_path}")


# ==============================================================================
# Parameter Synthesizer for Kinova Gen3 (No Pre-trained Controller)
# ==============================================================================
def get_or_create_kinova_pose_params(
    bot_path: str, num_dofs: int
) -> robotics.ControllerMochiArticulatedPoseParams:
    """Dynamically creates a .superdex_controller file matching the Kinova link count."""
    bot_dir = Path(bot_path).parent
    out_dir = bot_dir / "control"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl_file = out_dir / f"{Path(bot_path).stem}_pose.superdex_controller"

    # 1. Inspect Kinova bot prefab to get exact link count
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)
    num_links = len(bot_prefab.links)
    print(f"[Controller] Kinova has {num_links} links and {num_dofs} DOFs.")

    # 2. Tuned gains for the 7 Kinova Gen3 joints
    kinova_joint_stiffness = [600.0, 600.0, 500.0, 450.0, 200.0, 150.0, 100.0]
    kinova_joint_damping = [50.0, 50.0, 40.0, 35.0, 20.0, 15.0, 10.0]

    # 3. Build jointTracking array matching num_links
    joint_tracking = []
    
    # Determine offset: base links with no DOFs get 0
    # For a typical arm: Link 0 is base (0 gain), links 1..7 are joints, remaining are tool/flange (0 gain)
    num_lead_in = 1 if num_links in [8, 9] else (num_links - num_dofs - 1)
    if num_lead_in < 0:
        num_lead_in = 0

    for i in range(num_links):
        if num_lead_in <= i < (num_lead_in + num_dofs):
            joint_idx = i - num_lead_in
            kp = kinova_joint_stiffness[joint_idx]
            kd = kinova_joint_damping[joint_idx]
        else:
            kp = 0.0
            kd = 0.0
            
        joint_tracking.append({
            "damping": float(kd),
            "saturation": -1.0,
            "stiffness": float(kp)
        })

    # 4. linkPosTracking and linkRotTracking (all zeros, matching FR3 template)
    zero_tracking = [
        {"damping": 0.0, "saturation": -1.0, "stiffness": 0.0}
        for _ in range(num_links)
    ]

    # 5. Assemble exact schema
    config = {
        "poseControllerParams": {
            "jointTracking": joint_tracking,
            "linkPosTracking": zero_tracking,
            "linkRotTracking": zero_tracking
        }
    }

    # 6. Save and load
    with open(ctrl_file, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[Controller] Wrote {num_links}-element controller to: {ctrl_file}")
    return robotics.ControllerMochiArticulatedPoseParams.load_from_file(str(ctrl_file))


# ==============================================================================
# Main Execution
# ==============================================================================
def main() -> None:
    scene = None
    ik_scene = None
    bot = None
    ik_bot = None
    ik_solver = None
    ik_ee_handle = None
    logger = None
    
    try:
    
        bot_path = resolve_kinova_path()
        print(f"[Init] Loading Kinova Gen3 from: {bot_path}")

        # 1. Physics Engine Setup
        physics.initialize(num_worker_threads=0)

        # 2. Simulated Dynamic Scene
        scene = physics.create_scene("Kinova Gen3 Simulated Scene")
        scene.set_gravity([0.0, 0.0, -9.81])

        plane_shape = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
        scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

        robotics_context = robotics.create_context()
        bot = create_arm(scene, bot_path, robotics_context)
        bot_actor = bot.get_articulated_actor()
        num_dofs = bot_actor.get_num_dofs()
        print(f"[Simulated Scene] Actor: {bot_actor.get_name()} ({num_dofs} DOFs)")
        
        # Add table
        table_prefab_path = str(resolve_asset("table/table.mochi_scene"))
        physics.prefab.add_to_scene(
            prefab_path=table_prefab_path,
            root_path=str(resolve_asset_root("table/table.mochi_scene")),
            scene=scene,
            params=physics.prefab.PrefabParams(
                name="tablePrefab",
                # Shifted forward to X = 0.85m so the arm base stands outside the table
                translation=[1.4, 0.0, -0.8],
            ),
        )
        table_actor = find_actor(scene, "tablePrefab/Table")
        if table_actor:
            print(f"[Setup] Loaded Table Prefab Actor: {table_actor.get_name()}")
        
        # Locate the end-effector link in the SIMULATED scene
        sim_ee_handle, sim_ee_name = find_ee_link_handle(scene, bot_actor)
        sim_ee_actor = scene.get_actor(sim_ee_handle)
        print(f"[Simulated Scene] Tracking Tool Link: {sim_ee_name}")

        # Instantiate the logger
        logger = KinovaTelemetryLogger()
        step_count = 0

        # 3. Kinematic Twin Scene (Dedicated to IK)
        ik_scene = physics.create_scene("Kinova IK Solver Scene")
        ik_bot = create_arm(ik_scene, bot_path, robotics_context)
        ik_actor = ik_bot.get_articulated_actor()

        ik_ee_handle, ee_link_name = find_ee_link_handle(ik_scene, ik_actor)
        print(f"[IK Scene] End-Effector Link: {ee_link_name}")

        ik_solver = physics.experimental.create_ik_solver(ik_scene)
        ik_position_target = ik_solver.create_position_target(
            ik_ee_handle,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            IK_POSITION_WEIGHT,
        )
        ik_solver.create_rotation_target(
            ik_ee_handle,
            [0.0, 0.0, 0.0],
            EE_DOWN_ROTATION_VECTOR,
            IK_ROTATION_WEIGHT,
        )

        ik_pose = physics.DynamicArrayReal(num_dofs)

        # 4. Native Implicit Pose Controller Setup
        pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        pose_params = get_or_create_kinova_pose_params(bot_path, num_dofs)
        pose_controller.set_params(pose_params)
        pose_controller.initialize(True)

        pose_obsv = robotics.ControllerMochiArticulatedPoseObsv()
        pose_target = robotics.ControllerMochiArticulatedPoseTarget()
        pose_target.world_from_root = bot_actor.get_root_transform()

        # 5. Cartesian Circle Trajectory Definition
        # Kinova Gen3 reach is ~0.90 m; 0.45 m forward, 0.35 m high is well within workspace
        root_pos = np.asarray(bot_actor.get_root_transform().translation, dtype=float)
        circle_center = np.array([root_pos[0] + 0.45, root_pos[1], 0.35])
        circle_radius = 0.10  # 10 cm radius
        circle_period = 4.0   # 4 seconds per revolution

        # 6. Coordinate Convention & Debugger Attachment
        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

        print("\n[Simulation] Connecting to debugger...")
        print(">> NOTE: If connecting with 'Start Paused', remember to press Play on BOTH scenes.")

        if physics.debugger.attach():
            while physics.debugger.is_attached():
                sim_time = scene.get_total_simulation_time()

                # Compute current position on horizontal circle
                theta = 2.0 * np.pi * sim_time / circle_period
                target_xyz = [
                    circle_center[0] + circle_radius * np.cos(theta),
                    circle_center[1] + circle_radius * np.sin(theta),
                    circle_center[2],
                ]

                # 1. Update target & solve IK quasistatically on the kinematic twin
                ik_position_target.set_target_position(target_xyz)
                ik_solver.solve_ik()

                # 2. Extract solved joint positions from twin
                ik_actor.get_articulated_pose(ik_pose)

                # 3. Update implicit pose controller target
                pose_target.pose_dofs = ik_pose
                pose_controller.compute_output(pose_obsv, pose_target)

                # 4. Step only the simulated physics scene
                scene.step(TIME_STEP)
                
                # 5. Log the current time, target, and actual EEF position
                step_count += 1

                # Read actual EEF position after the step
                actual_xyz = np.asarray(sim_ee_actor.get_root_transform().translation, dtype=float)
                logger.record(sim_time, target_xyz, actual_xyz)

                # Print live diagnostics every 50 steps (every 0.25 seconds)
                if step_count % 50 == 0:
                    err_mm = np.linalg.norm(actual_xyz - target_xyz) * 1000.0
                    print(
                        f"Step {step_count:04d} | t={sim_time:.2f}s | "
                        f"Target: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}] | "
                        f"Actual: [{actual_xyz[0]:.3f}, {actual_xyz[1]:.3f}, {actual_xyz[2]:.3f}] | "
                        f"Error: {err_mm:5.1f} mm"
                    )
                    
    except KeyboardInterrupt:
        print("\n\n[Simulation] Ctrl+C detected! Stopping simulation loop...")

    finally:
        # Temporarily ignore subsequent Ctrl+C signals so teardown can finish cleanly
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("[Teardown] Cleaning up resources...")

        # 1. Export plot if logger exists
        if logger is not None:
            try:
                logger.generate_plots("kinova_tracking_plot.png")
            except Exception as e:
                print(f"[Teardown] Note during plot generation: {e}")

        # 2. Release IK constraints and IK scene
        try:
            if ik_solver is not None and ik_ee_handle is not None:
                ik_solver.clear_position_target(ik_ee_handle)
                ik_solver.clear_rotation_target(ik_ee_handle)
            if ik_scene is not None and ik_bot is not None:
                robotics.destroy_bot(ik_scene, ik_bot)
            if ik_solver is not None:
                physics.experimental.destroy_ik_solver(ik_solver)
        except Exception as e:
            print(f"[Teardown] Note during IK teardown: {e}")

        # 3. Release simulated robot and shutdown physics engine
        try:
            if scene is not None and bot is not None:
                robotics.destroy_bot(scene, bot)
            physics.shutdown()
            print("[Teardown] Physics engine cleanly shutdown.")
        except Exception as e:
            print(f"[Teardown] Note during physics shutdown: {e}")


if __name__ == "__main__":
    main()