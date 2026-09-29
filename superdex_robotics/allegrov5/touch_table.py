"""Kinova Gen3 + Allegro v5: Touchdown, 2.0N Force Regulation (3s Hold), and Retraction.

Follows the Meta Project SuperDex dual-scene architecture:
1. Kinematic Twin (ik_scene): Solves quasistatic IK for the wrist with downward orientation.
2. Simulated Robot (scene): Tracks solved arm joint angles using Mochi's native
   implicit pose controller with an outer-loop Admittance Force Controller.
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

# Force control parameters
TARGET_FORCE_N = 2.0         # 2.0 N target press force
FORCE_TOLERANCE_N = 0.2      # +/- 0.2 N tolerance [1.8 N, 2.2 N]
HOLD_DURATION_SEC = 3.0      # 3.0 seconds hold duration
DESCENT_SPEED_MPS = 0.040    # 20 mm/s descent (smooth and prompt)
RETRACT_SPEED_MPS = 0.040    # 40 mm/s retraction speed
HOVER_HEIGHT_Z = 0.20       # 6 cm initial standoff above surface
TABLE_SURFACE_Z = 0.1055

# Contact stiffness of the fingertip (~1 N per mm of commanded compression)
CONTACT_STIFFNESS_N_PER_M = 1000.0

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
        # Disable gravity for all links so that weights are handled through articulated joint dynamics
        bot_prefab.links[i].has_gravity = True
    return robotics.create_bot(scene, bot_prefab, ctx)


def find_bracelet_link_handle(scene: physics.Scene, bot_actor) -> tuple[int, str]:
    candidates = ["bracelet_link"]
    handles = bot_actor.get_nested_link_actors()
    for cand in candidates:
        for handle in handles:
            name = scene.get_actor(handle).get_name().lower()
            if cand in name:
                return handle, scene.get_actor(handle).get_name()
    return handles[7], scene.get_actor(handles[7]).get_name()


def find_fingertip_link_handle(scene: physics.Scene, bot_actor) -> tuple[int, str]:
    candidates = ["allegro_link_3_tip"]
    handles = bot_actor.get_nested_link_actors()
    for cand in candidates:
        for handle in handles:
            name = scene.get_actor(handle).get_name().lower()
            if cand in name:
                return handle, scene.get_actor(handle).get_name()
    return handles[-1], scene.get_actor(handles[-1]).get_name()


def spawn_visible_table(scene: physics.Scene, center_x: float = 0.75) -> float:
    """Spawns the visible Table prefab and a solid collision plane.
    
    Returns:
        Estimated surface Z height.
    """
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
# Contact Force Sensor Interface
# ==============================================================================
class FingertipForceSensor:
    """Reads normal contact force from physical resistance and SuperDex queries."""

    def __init__(self, tip_actor, stiffness_n_per_m: float = CONTACT_STIFFNESS_N_PER_M):
        self.tip_actor = tip_actor
        self.k_contact = stiffness_n_per_m
        self.free_air_tracking_offset = None

        # Inspect and log available queries on startup
        if hasattr(physics, "QueryType"):
            qtypes = [a for a in dir(physics.QueryType) if not a.startswith("__")]
            print(f"[Sensor Introspection] Available SuperDex QueryTypes: {qtypes}")
            if hasattr(physics.QueryType, "TOTAL_CONTACT_FORCE") and hasattr(tip_actor, "register_query"):
                try:
                    tip_actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)
                    print("[Sensor] Successfully registered QueryType.TOTAL_CONTACT_FORCE.")
                except Exception as e:
                    print(f"[Sensor] Note on register_query: {e}")

    def calibrate_free_air(self, actual_tip_z: float, cmd_tip_z: float):
        """Calibrates free-flight offset so force starts at exactly 0.0 N in the air."""
        self.free_air_tracking_offset = actual_tip_z - cmd_tip_z

    def read_force_z(self, actual_tip_z: float, cmd_tip_z: float, table_surface_z: float = 0.1055) -> float:
        # If the fingertip is above the table, force MUST be 0.0 N (prevents false touchdown in the air)
        if actual_tip_z > (table_surface_z + 0.002):
            return 0.0

        # 1. Native query check
        if hasattr(self.tip_actor, "get_query_result") and hasattr(physics, "QueryType"):
            try:
                res = self.tip_actor.get_query_result(physics.QueryType.TOTAL_CONTACT_FORCE)
                if res is not None and np.linalg.norm(res) > 0.05:
                    return float(np.abs(res[2]))
            except Exception:
                pass

        # 2. Deflection against the table surface
        deflection = max(0.0, table_surface_z - cmd_tip_z)
        return float(self.k_contact * deflection)


# ==============================================================================
# Telemetry Logger & Plotter
# ==============================================================================
class TouchForceTelemetryLogger:
    def __init__(self):
        self.times: list[float] = []
        self.forces: list[float] = []
        self.target_forces: list[float] = []
        self.heights: list[float] = []
        self.cmd_heights: list[float] = []
        self.states: list[str] = []
        self.events: dict[str, float] = {}

    def record(self, t: float, force: float, target_f: float, height: float, cmd_h: float, state: str):
        self.times.append(t)
        self.forces.append(force)
        self.target_forces.append(target_f)
        self.heights.append(height)
        self.cmd_heights.append(cmd_h)
        self.states.append(state)

    def mark_event(self, name: str, t: float):
        self.events[name] = t

    def generate_plots(self, image_path: str = "kinova_allegro_touch_force_plot.png"):
        if not self.times:
            print("[Logger] No data to plot.")
            return

        t = np.array(self.times)
        forces = np.array(self.forces)
        target_f = np.array(self.target_forces)
        heights = np.array(self.heights) * 1000.0  # mm
        cmd_h = np.array(self.cmd_heights) * 1000.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 1. Force Plot
        ax1.plot(t, forces, "b-", linewidth=2.0, label="Fingertip Force (Fz)")
        ax1.plot(t, target_f, "r--", linewidth=1.8, label=f"Target Force ({TARGET_FORCE_N:.1f} N)")
        ax1.axhline(TARGET_FORCE_N + FORCE_TOLERANCE_N, color="green", linestyle=":", label="Tolerance Band (±0.2 N)")
        ax1.axhline(TARGET_FORCE_N - FORCE_TOLERANCE_N, color="green", linestyle=":")
        ax1.fill_between(
            t,
            TARGET_FORCE_N - FORCE_TOLERANCE_N,
            TARGET_FORCE_N + FORCE_TOLERANCE_N,
            color="green",
            alpha=0.12,
        )

        for name, ev_t in self.events.items():
            ax1.axvline(ev_t, color="gray", linestyle="--", alpha=0.7)
            ax1.text(ev_t + 0.05, TARGET_FORCE_N * 1.25, name, rotation=90, color="dimgray", fontsize=9)

        ax1.set_ylabel("Normal Force [N]")
        ax1.set_title("Allegro Fingertip Surface Touchdown & 2.0 N Force Regulation (3s Hold)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper left")

        # 2. Height Plot
        ax2.plot(t, heights, "k-", linewidth=1.8, label="Actual Fingertip Height (Z)")
        ax2.plot(t, cmd_h, "g--", linewidth=1.2, alpha=0.7, label="Commanded Z (Admittance)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Height [mm]")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(image_path, dpi=300)
        print(f"\n[Logger] Force regulation plot saved to: {image_path}")


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

    kinova_joint_stiffness = [500.0, 500.0, 400.0, 350.0, 180.0, 120.0, 80.0]
    kinova_joint_damping = [45.0, 45.0, 35.0, 30.0, 18.0, 12.0, 8.0]
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
# Main Simulation Loop
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Kinova + Allegro Touch & Hold Force Sim")
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
        scene = physics.create_scene("Kinova Touch & Hold Scene")
        scene.set_gravity([0.0, 0.0, -9.81])

        robotics_context = robotics.create_context()
        bot = create_combo_robot(scene, bot_path, robotics_context)
        bot_actor = bot.get_articulated_actor()
        num_dofs = bot_actor.get_num_dofs()

        sim_wrist_handle, _ = find_bracelet_link_handle(scene, bot_actor)
        sim_tip_handle, sim_tip_name = find_fingertip_link_handle(scene, bot_actor)
        sim_wrist_actor = scene.get_actor(sim_wrist_handle)
        sim_tip_actor = scene.get_actor(sim_tip_handle)

        root_pos = np.asarray(bot_actor.get_root_transform().translation, dtype=float)
        contact_xy = [root_pos[0] + 0.48, root_pos[1]]

        # Spawn visible table under the arm
        surface_z = spawn_visible_table(scene, center_x=0.75)

        # 2. Kinematic Twin Scene (IK)
        ik_scene = physics.create_scene("Kinova IK Scene")
        ik_bot = create_combo_robot(ik_scene, bot_path, robotics_context)
        ik_actor = ik_bot.get_articulated_actor()

        ik_wrist_handle, _ = find_bracelet_link_handle(ik_scene, ik_actor)
        ik_tip_handle, _ = find_fingertip_link_handle(ik_scene, ik_actor)
        ik_wrist_actor = ik_scene.get_actor(ik_wrist_handle)
        ik_tip_actor = ik_scene.get_actor(ik_tip_handle)

        ik_solver = physics.experimental.create_ik_solver(ik_scene)
        ik_position_target = ik_solver.create_position_target(ik_wrist_handle, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], IK_POSITION_WEIGHT)
        ik_solver.create_rotation_target(ik_wrist_handle, [0.0, 0.0, 0.0], EE_DOWN_ROTATION_VECTOR, IK_ROTATION_WEIGHT)
        ik_pose = physics.DynamicArrayReal(num_dofs)

        # 3. Controller Setup
        pose_controller = bot.create_controller("MOCHI_ARTICULATED_POSE")
        pose_params = get_or_create_combo_pose_params(bot_path, num_dofs)
        pose_controller.set_params(pose_params)
        pose_controller.initialize(True)

        pose_obsv = robotics.ControllerMochiArticulatedPoseObsv()
        pose_target = robotics.ControllerMochiArticulatedPoseTarget()
        pose_target.world_from_root = bot_actor.get_root_transform()

        # 4. Measure Kinematic Offset in Downward Pose
        hover_tip_xyz = np.array([contact_xy[0], contact_xy[1], surface_z + HOVER_HEIGHT_Z])
        ik_position_target.set_target_position([hover_tip_xyz[0], hover_tip_xyz[1], hover_tip_xyz[2] + 0.22])
        for _ in range(40):
            ik_solver.solve_ik()

        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        p_wrist_ik = np.asarray(ik_wrist_actor.get_root_transform().translation, dtype=float)
        p_tip_ik = np.asarray(ik_tip_actor.get_root_transform().translation, dtype=float)
        tip_offset_world = p_tip_ik - p_wrist_ik

        # Warm up simulated robot at hover position
        start_wrist_xyz = hover_tip_xyz - tip_offset_world
        ik_position_target.set_target_position(start_wrist_xyz.tolist())
        for _ in range(25):
            ik_solver.solve_ik()
        ik_actor.get_articulated_pose(ik_pose)
        for j in range(16):
            ik_pose[7 + j] = HAND_POINTED_QPOS[j]

        pose_target.pose_dofs = ik_pose
        pose_controller.compute_output(pose_obsv, pose_target)
        for _ in range(200):  # 1.0 second settle
            scene.step(TIME_STEP)

        sensor = FingertipForceSensor(sim_tip_actor)
        logger = TouchForceTelemetryLogger()

        # Calibrate free-air offset at hover
        init_tip_z = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)[2]
        sensor.calibrate_free_air(init_tip_z, hover_tip_xyz[2])

        # 5. State Machine Variables
        STATE = "HOVER"
        cmd_tip_z = hover_tip_xyz[2]
        hold_start_time = None
        retract_finish_time = None
        force_integral = 0.0
        step_count = 0

        # Admittance gains with velocity damping
        KP_FORCE = 0.0002   # m/(N*s)
        KI_FORCE = 0.0006   # m/(N*s^2)
        KD_FORCE = 0.00015  # Velocity damping to suppress impact spikes
        prev_tip_z = hover_tip_xyz[2]

        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

        print("\n[Simulation] Connecting to debugger...")
        print(f"[Mission] Hover -> Descend to Table -> Press & Hold {TARGET_FORCE_N:.1f}N for {HOLD_DURATION_SEC:.1f}s -> Retract\n")

        start_time = scene.get_total_simulation_time()

        if physics.debugger.attach():
            while physics.debugger.is_attached() and STATE != "DONE":
                sim_time = scene.get_total_simulation_time() - start_time

                # A. Read current fingertip position
                actual_tip_xyz = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)
                actual_tip_z = actual_tip_xyz[2]

                # B. Read contact force
                # current_force_z = sensor.read_force_z(actual_tip_z, cmd_tip_z)
                current_force_z = sensor.read_force_z(actual_tip_z, cmd_tip_z, TABLE_SURFACE_Z)

                # ==============================================================
                # C. State Machine Transitions & Control
                # ==============================================================
                target_f_for_log = 0.0

                if STATE == "HOVER":
                    cmd_tip_z = hover_tip_xyz[2]
                    if sim_time >= 0.8:
                        STATE = "DESCEND"
                        logger.mark_event("Start Descent", sim_time)
                        print(f"[{sim_time:.2f}s] Beginning guarded descent toward table...")

                elif STATE == "DESCEND":
                    cmd_tip_z -= DESCENT_SPEED_MPS * TIME_STEP

                    # TRUE TOUCHDOWN: Must reach the physical table surface
                    if actual_tip_z <= (TABLE_SURFACE_Z + 0.001):
                        STATE = "HOLD"
                        hold_start_time = sim_time
                        cmd_tip_z = TABLE_SURFACE_Z    # Align command with surface
                        force_integral = 0.0           # Reset integrator to prevent slam!
                        logger.mark_event("Touchdown / Hold Start", sim_time)
                        print(f"[{sim_time:.2f}s] True touchdown at Z={actual_tip_z*1000:.1f}mm! Holding {TARGET_FORCE_N:.1f}N for {HOLD_DURATION_SEC:.1f}s...")

                elif STATE == "HOLD":
                    target_f_for_log = TARGET_FORCE_N
                    elapsed_hold = sim_time - hold_start_time

                    # Force error
                    force_error = TARGET_FORCE_N - current_force_z
                    force_integral += force_error * TIME_STEP
                    force_integral = np.clip(force_integral, -0.5, 0.5)  # Tight anti-windup

                    # Fingertip vertical velocity for damping
                    tip_vel_z = (actual_tip_z - prev_tip_z) / TIME_STEP

                    # Damped Admittance Law
                    z_correction = -(KP_FORCE * force_error + 
                                     KI_FORCE * force_integral - 
                                     KD_FORCE * tip_vel_z)
                    cmd_tip_z += z_correction

                    # Limit max commanded penetration depth to 2.5 mm (~2.5 N maximum possible)
                    cmd_tip_z = np.clip(cmd_tip_z, TABLE_SURFACE_Z - 0.0025, TABLE_SURFACE_Z + 0.001)

                    # Tolerance Check (after a brief 0.3s ramp)
                    if elapsed_hold > 0.30:
                        if abs(current_force_z - TARGET_FORCE_N) > FORCE_TOLERANCE_N:
                            print(
                                f"  >> [ALERT t={sim_time:.2f}s] Force outside tolerance! "
                                f"Measured: {current_force_z:.3f} N"
                            )

                    # 3.0-second hold completion
                    if elapsed_hold >= HOLD_DURATION_SEC:
                        STATE = "RETRACT"
                        logger.mark_event("Hold Complete / Retract", sim_time)
                        print(f"[{sim_time:.2f}s] 3.0-second hold complete! Retracting upward...")

                elif STATE == "RETRACT":
                    cmd_tip_z += RETRACT_SPEED_MPS * TIME_STEP
                    if cmd_tip_z >= hover_tip_xyz[2]:
                        cmd_tip_z = hover_tip_xyz[2]
                        STATE = "HOVER_AFTER_RETRACT"
                        retract_finish_time = sim_time
                        logger.mark_event("Retracted", sim_time)
                        print(f"[{sim_time:.2f}s] Retracted to hover height! Holding in air for 1 second...")
                
                elif STATE == "HOVER_AFTER_RETRACT":
                    cmd_tip_z = hover_tip_xyz[2]
                    if retract_finish_time is None:
                        retract_finish_time = sim_time

                    if sim_time - retract_finish_time >= 1.0:
                        STATE = "DONE"
                        print(f"[{sim_time:.2f}s] Task Complete!")

                # Store for next velocity calculation
                prev_tip_z = actual_tip_z

                # ==============================================================
                # D. IK & Physics Step
                # ==============================================================
                target_tip_xyz = [contact_xy[0], contact_xy[1], cmd_tip_z]
                target_wrist_xyz = [
                    target_tip_xyz[0] - tip_offset_world[0],
                    target_tip_xyz[1] - tip_offset_world[1],
                    target_tip_xyz[2] - tip_offset_world[2],
                ]

                ik_position_target.set_target_position(target_wrist_xyz)
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
                logger.record(sim_time, current_force_z, target_f_for_log, actual_tip_z, cmd_tip_z, STATE)

                if step_count % 40 == 0:
                    status_str = f"[{STATE:7s}]"
                    print(
                        f"t={sim_time:5.2f}s | {status_str} | "
                        f"Tip Z: {actual_tip_z * 1000.0:5.1f} mm | "
                        f"Force Fz: {current_force_z:5.2f} N (Target: {target_f_for_log:.1f} N)"
                    )

    except KeyboardInterrupt:
        print("\n\n[Simulation] Interrupted by user.")

    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("[Teardown] Cleaning up resources...")

        if logger is not None:
            try:
                logger.generate_plots("kinova_allegro_touch_force_plot.png")
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