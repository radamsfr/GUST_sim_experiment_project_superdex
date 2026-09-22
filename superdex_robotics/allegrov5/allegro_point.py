# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simulating the Allegro v5 Right Hand in a Pointed Finger Position.

Loads allegro_v5_right.superdex_bot, applies the pointing finger joint targets
(Index extended, Middle/Ring/Thumb curled), and visualizes the hand in the
SuperDex Physics Debugger.
"""

import argparse
import numpy as np

import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset


def get_default_allegro_path() -> str:
    """Resolve path to the Allegro v5 right hand .superdex_bot asset."""
    return str(
        resolve_asset(
            "bots/hands/allegro_v5/right/allegro_v5_right.superdex_bot"
        )
    )


def get_pointed_finger_angles() -> list[float]:
    """Return the 16 target joint angles for the pointed finger gesture."""
    # Index finger (Joints 0-3): straight / pointing out
    q_index = [0.0, 0.05, 0.07, 0.02]

    # Middle finger (Joints 4-7): curled into the palm
    q_middle = [0.0, 1.55, 1.55, 1.30]

    # Ring finger (Joints 8-11): curled into the palm
    q_ring = [0.0, 1.55, 1.55, 1.30]

    # Thumb (Joints 12-15): rotated and tucked against palm
    q_thumb = [1.20, 0.60, 1.20, 0.80]

    return q_index + q_middle + q_ring + q_thumb


def main() -> None:
    parser = argparse.ArgumentParser(description="Allegro v5 Pointed Finger Sim")
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to allegro_v5_right.superdex_bot",
    )
    args = parser.parse_args()

    bot_path = args.path if args.path else get_default_allegro_path()

    # 1. Initialize the physics engine (single-threaded for determinism)
    physics.initialize(num_worker_threads=0)

    # 2. Create scene
    scene = physics.create_scene("Allegro Hand Pointing Simulation")

    # Set zero-gravity so the floating hand does not fall out of view
    scene.set_gravity([0, 0, 0])

    # 3. Load the bot prefab
    bot_prefab = robotics.load_bot_prefab_from_file(bot_path)

    # Pointing finger angles (16 DOFs)
    pointed_qpos = get_pointed_finger_angles()

    # Override the prefab default pose if supported so it spawns directly into the pose
    if hasattr(bot_prefab, "default_pose"):
        bot_prefab.default_pose = pointed_qpos
    elif hasattr(bot_prefab, "defaultPose"):
        bot_prefab.defaultPose = pointed_qpos

    # 4. Instantiate the Bot into the scene
    robotics_context = robotics.create_context()
    bot = robotics.create_bot(scene, bot_prefab, robotics_context)
    bot_actor = bot.get_articulated_actor()

    num_dofs = bot_actor.get_num_dofs()
    print(f"Robot: {bot_prefab.name}")
    print(f"  Links: {len(bot_prefab.links)}")
    print(f"  Joints: {len(bot_prefab.joints)}")
    print(f"  Articulated DOFs: {num_dofs}")

    # Build full qpos vector (handling 16 DOFs or floating-base + 16 DOFs)
    target_qpos = np.array(pointed_qpos, dtype=np.float32)
    if num_dofs > len(pointed_qpos):
        # In case root has 6 floating base DOFs prepended: [pos(3), rot(3 or 4), joints(16)]
        extra_dofs = num_dofs - len(pointed_qpos)
        target_qpos = np.concatenate([np.zeros(extra_dofs, dtype=np.float32), target_qpos])

    # Set initial joint positions
    if hasattr(bot_actor, "set_joint_positions"):
        bot_actor.set_joint_positions(target_qpos)
    elif hasattr(bot_actor, "set_positions"):
        bot_actor.set_positions(target_qpos)

    # Optional: static ground plane reference
    plane_shape = physics.create_plane_shape(normal=[0, 0, 1], distance=-0.2)
    scene.create_rigid_actor(name="ground", shape=plane_shape, is_static=True)

    # 5. Set coordinate space for SuperDex Physics Debugger (FLU: Forward, Left, Up)
    physics.get_debug_server().set_coordinate_space(
        physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
    )

    time_step = 1.0 / 60.0

    # 6. Run simulation loop and attach debugger
    print("Connecting to SuperDex Physics Debugger...")
    if physics.debugger.attach():
        print("Connected! Simulating pointed finger position. Close debugger to exit.")
        while physics.debugger.is_attached():
            # Continuously enforce joint position targets to hold pose
            if hasattr(bot_actor, "set_joint_position_targets"):
                bot_actor.set_joint_position_targets(target_qpos)
            elif hasattr(bot_actor, "set_joint_positions"):
                bot_actor.set_joint_positions(target_qpos)

            scene.step(time_step)

    # 7. Clean up
    robotics.destroy_bot(scene, bot)
    physics.shutdown()
    print("Simulation complete.")


if __name__ == "__main__":
    main()