"""Kinova Gen3 7-DOF + Allegro Hand v5: GUST-Filtered Surface Tracing.

GUST Safety Filter Integration:
1. Physical Task: Traces an 8 cm radius circle on a textured plate while regulating
   normal contact force F_z to TARGET_FORCE_N = 2.0 N within [1.70 N, 2.30 N].
2. Disturbance Landscape: The Lipschitz-continuous textured surface acts as a spatial
   disturbance d_i = |F_z - F_target| with known Lipschitz constant L_D.
3. Trust Cones & Event-Triggered Sampling: GUST maintains the upper-bound envelope:
   D_combined(p) = min_i (d_i + L_D * ||p - p_i||_2).
   A new tactile sample is acquired when D_combined reaches the trust boundary.
4. GUST Velocity Governor: Lookahead evaluation of D_combined proactively scales down
   tangential tracing speed, giving the vertical admittance controller ample time to
   conform to steep hills and prevent contact force violations.
5. Focused Visuals: 2-panel figure:
   - Panel 1: Contact Force vs [1.70 N, 2.30 N] Tolerance Band + Governor Scaling s(t).
   - Panel 2: 2D Topographic Map with Trajectory, GUST Samples, and Trust Cones.
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

import superdex.physics as physics
import superdex.robotics as robotics
from superdex.physics.paths import resolve_asset, resolve_asset_root

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
FORCE_TOLERANCE_N = 0.25      # +/- 0.25 N tolerance band [1.75 N, 2.25 N]
HOVER_HEIGHT_Z  = 0.2000     # 20 cm standoff height
DESCENT_SPEED_MPS = 0.035    # 35 mm/s descent speed
RETRACT_SPEED_MPS = 0.040    # 40 mm/s retraction speed
CONTACT_STIFFNESS_N_PER_M = 1000.0

# Circular Trajectory Parameters (XY Plane)
CIRCLE_RADIUS = 0.08         # 8 cm radius
CIRCLE_PERIOD_SEC = 6.0      # Nominal 6.0 seconds per revolution
TOTAL_REVOLUTIONS = 2.0      # Trace 2 full revolutions under GUST

# ==============================================================================
# GUST Safety Filter Configuration
# ==============================================================================
GUST_L_D = 8.0               # Known a-priori Lipschitz constant of force disturbance [N/m]
GUST_D_MAX = FORCE_TOLERANCE_N  # Maximum allowable disturbance before failure (0.25 N)
GUST_EPSILON = 0.03          # Buffer margin below D_max to trigger new tactile sample (N)
GUST_MIN_STEP_DIST = 0.015   # Minimum spatial distance between samples (15 mm)
GUST_LOOKAHEAD_DIST = 0.018  # Lookahead distance along circular trajectory (18 mm)
GUST_MIN_SPEED_SCALE = 0.15  # Minimum governor speed scaling factor (15% of nominal)

# Pointed finger configuration for 16 Allegro hand joints
HAND_POINTED_QPOS = [
    0.0, 0.05, 0.07, 0.02,   # Index (straight)
    0.0, 1.55, 1.55, 1.30,   # Middle (curled)
    0.0, 1.55, 1.55, 1.30,   # Ring (curled)
    1.20, 0.60, 1.20, 0.80,  # Thumb (tucked)
]

BRACELET_LINK_NAME = "kinova_gen3_allegro_v5_right/bracelet_link"
INDEX_TIP_LINK_NAME = "kinova_gen3_allegro_v5_right/allegro_link_3_tip"
INDEX_COLLISION_LINK_NAME = "kinova_gen3_allegro_v5_right/allegro_link_3"

# ==============================================================================
# GUST Filter Class: Disturbance Cones & Velocity Governor
# ==============================================================================
class GUSTSafetyFilter:
    """Implements GUST Disturbance Cones and Velocity Governor with known L_D."""
    def __init__(
        self,
        L_D: float = GUST_L_D,
        D_max: float = GUST_D_MAX,
        epsilon: float = GUST_EPSILON,
        min_step_dist: float = GUST_MIN_STEP_DIST,
        lookahead_dist: float = GUST_LOOKAHEAD_DIST,
        min_speed_scale: float = GUST_MIN_SPEED_SCALE,
    ):
        self.L_D = L_D
        self.D_max = D_max
        self.trigger_threshold = D_max - epsilon
        self.min_step_dist = min_step_dist
        self.lookahead_dist = lookahead_dist
        self.min_speed_scale = min_speed_scale

        self.sampled_points: list[np.ndarray] = []  # Sample positions p_i in R^2
        self.sampled_disturbances: list[float] = [] # Tactile disturbance d_i = |Fz - F_target|
        self.sample_times: list[float] = []

    def add_sample(self, p: np.ndarray, d: float, t: float):
        """Records a new ground-truth tactile disturbance sample."""
        self.sampled_points.append(np.array(p[:2], dtype=float))
        self.sampled_disturbances.append(float(d))
        self.sample_times.append(float(t))

    def evaluate_combined_d(self, p: np.ndarray) -> float:
        """Computes D_combined(p) = min_i (d_i + L_D * ||p - p_i||_2)."""
        if len(self.sampled_points) == 0:
            return float("inf")
        pts = np.asarray(self.sampled_points)
        dists = np.linalg.norm(pts - p[:2], axis=1)
        cone_vals = np.asarray(self.sampled_disturbances) + self.L_D * dists
        return float(np.min(cone_vals))

    def should_sample(self, p_curr: np.ndarray) -> bool:
        """Event-trigger: Checks if current state reaches the trust boundary."""
        if len(self.sampled_points) == 0:
            return True
        d_combined = self.evaluate_combined_d(p_curr)
        if d_combined >= self.trigger_threshold:
            dist_from_last = np.linalg.norm(p_curr[:2] - self.sampled_points[-1])
            return dist_from_last >= self.min_step_dist
        return False

    def compute_governor_speed_scale(self, p_lookahead: np.ndarray) -> float:
        """
        GUST Velocity Governor:
        Proactively slows down tracking when predicted lookahead disturbance approaches D_max.
        """
        d_ahead = self.evaluate_combined_d(p_lookahead)
        if np.isinf(d_ahead):
            return self.min_speed_scale
        
        # Linear scale: 1.0 when d_ahead=0, min_speed_scale when d_ahead >= D_max
        scale = 1.0 - (d_ahead / self.D_max)
        return float(np.clip(scale, self.min_speed_scale, 1.0))


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


def get_robot_links_by_name(scene: physics.Scene, bot_actor) -> dict[str, tuple[any, any]]:
    return {
        scene.get_actor(h).get_name(): (h, scene.get_actor(h))
        for h in bot_actor.get_nested_link_actors()
    }


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
    seed=102,
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
# Robust Fingertip Force Sensor Interface
# ==============================================================================
class FingertipForceSensor:
    def __init__(self, contact_actor, stiffness_n_per_m: float = CONTACT_STIFFNESS_N_PER_M):
        self.contact_actor = contact_actor
        self.k_contact = stiffness_n_per_m
        self.free_air_offset = 0.0
        self.target_qt = physics.QueryType.TOTAL_CONTACT_FORCE
        
        try:
            self.query_handle = self.contact_actor.register_query(self.target_qt)
        except Exception as e:
            print(f"[Sensor] Note on register_query: {e}")
            self.query_handle = None

    def calibrate_free_air(self, actual_tip_z: float, cmd_tip_z: float):
        self.free_air_offset = actual_tip_z - cmd_tip_z

    def read_force_z(self, actual_tip_z: float, cmd_tip_z: float) -> float:
        if self.query_handle is not None:
            try:
                res = self.contact_actor.get_query_result(self.query_handle)
                if res is not None:
                    res_arr = np.asarray(res, dtype=float)
                    if res_arr.size >= 3 and abs(res_arr[2]) > 0.05:
                        return float(abs(res_arr[2]))
            except Exception:
                pass

        try:
            res = self.contact_actor.get_contact_force_world()
            if res is not None and len(res) >= 3 and abs(res[2]) > 0.05:
                return float(abs(res[2]))
        except Exception:
            pass

        raw_deflection = (actual_tip_z - cmd_tip_z) - self.free_air_offset
        return float(self.k_contact * max(0.0, raw_deflection))

    def teardown(self):
        if self.query_handle is not None:
            try:
                self.contact_actor.cancel_query(self.query_handle)
            except Exception:
                pass


# ==============================================================================
# Dedicated 2-Panel GUST Telemetry Logger & Visualizer
# ==============================================================================
class GUSTTelemetryLogger:
    def __init__(self):
        self.times: list[float] = []
        self.forces: list[float] = []
        self.actual_positions: list[np.ndarray] = []
        self.target_positions: list[np.ndarray] = []
        self.governor_scales: list[float] = []
        self.states: list[str] = []
        self.events: dict[str, float] = {}

    def record(
        self,
        t: float,
        force: float,
        actual_xyz: np.ndarray,
        target_xyz: np.ndarray,
        governor_scale: float,
        state: str,
    ):
        self.times.append(t)
        self.forces.append(force)
        self.actual_positions.append(np.array(actual_xyz, dtype=float))
        self.target_positions.append(np.array(target_xyz, dtype=float))
        self.governor_scales.append(governor_scale)
        self.states.append(state)

    def mark_event(self, name: str, t: float):
        self.events[name] = t

    def generate_two_panel_plot(
        self,
        image_path: str = "gust_tactile_tracing_results.png",
        landscape_fn: callable = None,
        gust_filter: GUSTSafetyFilter = None,
        is_gust_enabled: bool = True,
    ):
        if not self.times:
            print("[Logger] No telemetry data to plot.")
            return

        t = np.array(self.times)
        forces = np.array(self.forces)
        actual_pos = np.array(self.actual_positions)
        target_pos = np.array(self.target_positions)
        gov_scales = np.array(self.governor_scales)
        states = np.array(self.states)
        contact_mask = states == "TRACE_AND_HOLD"

        # Publication-ready aspect ratio and typography styling
        fig, (ax_f, ax_map) = plt.subplots(
            1, 2, 
            figsize=(18, 7.0),
            gridspec_kw={"wspace": 0.48}   # <-- Adds clearance between the two panels
        )

        # ==============================================================
        # Panel 1: Normal Contact Force vs Tolerance Band & Governor
        # ==============================================================
        ax_f.plot(
            t,
            forces,
            color="#0055d4",
            linestyle="-",
            linewidth=2.4,
            label="Contact Force $F_z(t)$",
            zorder=3,
        )
        ax_f.axhline(
            TARGET_FORCE_N,
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            label=f"Target Force ({TARGET_FORCE_N:.1f} N)",
            zorder=2,
        )

        # Tolerance band
        upper_bound = TARGET_FORCE_N + FORCE_TOLERANCE_N
        lower_bound = TARGET_FORCE_N - FORCE_TOLERANCE_N
        ax_f.axhline(
            upper_bound,
            color="#2ca02c",
            linestyle=":",
            linewidth=1.8,
            label=f"Tolerance Band (±{FORCE_TOLERANCE_N:.2f} N)",
            zorder=2,
        )
        ax_f.axhline(lower_bound, color="#2ca02c", linestyle=":", linewidth=1.8, zorder=2)
        ax_f.fill_between(
            t,
            lower_bound,
            upper_bound,
            color="#2ca02c",
            alpha=0.18,
            zorder=1,
        )

        # Secondary Y axis for Governor Speed Scale (SOLID LINE)
        ax_gov = ax_f.twinx()
        ax_gov.plot(
            t,
            gov_scales,
            color="#e66101",
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
            label="Governor Scale $s(t)$",
            zorder=4,
        )
        ax_gov.set_ylabel(
            "Governor Speed Scaling $s(t)$",
            color="#b34700",
            fontsize=18,
            fontweight="bold",
        )
        ax_gov.set_ylim([0.0, 1.15])
        ax_gov.tick_params(axis="y", labelcolor="#b34700", labelsize=15)

        ax_f.set_xlabel("Time [s]", fontsize=18, fontweight="bold")
        ax_f.set_ylabel("Normal Force $F_z$ [N]", fontsize=18, fontweight="bold")
        ax_f.tick_params(axis="both", labelsize=15)

        title_f = (
            "Contact Force Regulation with GUST Governor"
            if is_gust_enabled
            else "Contact Force Regulation (Unfiltered Baseline)"
        )
        ax_f.set_title(title_f, fontweight="bold", fontsize=20, pad=12)
        ax_f.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)

        # Place legend below axis so data curves are never hidden
        lines_f, labels_f = ax_f.get_legend_handles_labels()
        lines_g, labels_g = ax_gov.get_legend_handles_labels()
        ax_f.legend(
            lines_f + lines_g,
            labels_f + labels_g,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=18,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.95,
        )

        # ==============================================================
        # Panel 2: 2D Topographic Map with Trajectory & GUST Cones
        # ==============================================================
        if landscape_fn is not None:
            cx = float(np.mean(target_pos[:, 0]))
            cy = float(np.mean(target_pos[:, 1]))
            margin = 1.35 * CIRCLE_RADIUS
            gx = np.linspace(cx - margin, cx + margin, 140)
            gy = np.linspace(cy - margin, cy + margin, 140)
            GX, GY = np.meshgrid(gx, gy)
            coords_flat = np.column_stack([GX.ravel(), GY.ravel()])

            h_grid_mm = landscape_fn(coords_flat).reshape(GX.shape) * 1000.0  # mm
            cf = ax_map.contourf(GX, GY, h_grid_mm, levels=25, cmap="terrain", alpha=0.55)
            cs = ax_map.contour(
                GX,
                GY,
                h_grid_mm,
                levels=10,
                colors="dimgray",
                alpha=0.45,
                linewidths=0.7,
            )
            ax_map.clabel(cs, inline=True, fontsize=15, fmt="%.1f")

            cbar = fig.colorbar(cf, ax=ax_map, fraction=0.046, pad=0.04)
            cbar.set_label("Surface Height $h(x, y)$ [mm]", fontsize=18, fontweight="bold")
            cbar.ax.tick_params(labelsize=15)

        # Nominal Target Circle
        ax_map.plot(
            target_pos[:, 0],
            target_pos[:, 1],
            color="#d62728",
            linestyle="--",
            linewidth=2.2,
            label="Nominal Target Circle",
            zorder=3,
        )

        # Actual Path (In Contact)
        if np.any(contact_mask):
            ax_map.plot(
                actual_pos[contact_mask, 0],
                actual_pos[contact_mask, 1],
                color="#0055d4",
                linestyle="-",
                linewidth=2.4,
                label="Actual Path (In Contact)",
                zorder=4,
            )
        else:
            ax_map.plot(
                actual_pos[:, 0],
                actual_pos[:, 1],
                color="#0055d4",
                linestyle="-",
                linewidth=2.0,
                label="Actual Path",
                zorder=4,
            )

        # Overlay GUST Samples & Trust Radius Cones
        if gust_filter is not None and len(gust_filter.sampled_points) > 0:
            s_pts = np.asarray(gust_filter.sampled_points)
            s_dists = np.asarray(gust_filter.sampled_disturbances)

            # Star markers for sample locations
            ax_map.scatter(
                s_pts[:, 0],
                s_pts[:, 1],
                color="#e41a1c",
                edgecolors="black",
                linewidths=0.6,
                marker="*",
                s=140,
                zorder=6,
                label="Tactile Sample $p_i$",
            )

            # Draw trust radius circle for each cone: r_trust = (D_max - d_i) / L_D
            for i in range(len(s_pts)):
                r_trust = max(0.002, (gust_filter.D_max - s_dists[i]) / gust_filter.L_D)
                circle_patch = plt.Circle(
                    (s_pts[i, 0], s_pts[i, 1]),
                    r_trust,
                    color="#e41a1c",
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.75,
                )
                ax_map.add_patch(circle_patch)

            # Dummy line for legend
            ax_map.plot([], [], color="#e41a1c", linestyle="--", linewidth=1.2, label="Trust Cone ($r_{\mathrm{trust}}$)")

        ax_map.set_xlabel("Position X [m]", fontsize=18, fontweight="bold")
        ax_map.set_ylabel("Position Y [m]", fontsize=18, fontweight="bold")
        ax_map.tick_params(axis="both", labelsize=15)

        title_m = (
            "Trajectory & GUST Trust Cones"
            if is_gust_enabled
            else "Trajectory on Textured Surface (No GUST)"
        )
        ax_map.set_title(title_m, fontweight="bold", fontsize=20, pad=12)
        ax_map.axis("equal")
        ax_map.grid(True, linestyle="--", alpha=0.4, linewidth=0.8)

        # Place legend below axis so surface contours, circle, and cones are completely unobstructed
        ax_map.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=18,
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.95,
        )

        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        print(f"\n[Logger] 2-panel GUST tactile tracing visual saved to: {image_path}")


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
# Main Simulation Execution with GUST
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Kinova + Allegro GUST Textured Trajectory")
    parser.add_argument("path", type=str, nargs="?", default=None, help="Path to combo .superdex_bot")
    parser.add_argument(
        "--no-gust",
        action="store_true",
        help="Disable GUST safety filter (runs unfiltered baseline at constant speed)",
    )
    parser.add_argument(
        "--constant-speed",
        type=float,
        default=None,
        help="Disable adaptive governor and use a fixed speed scale."
    )
    args = parser.parse_args()

    # enable_gust = not args.no_gust
    if args.constant_speed is not None:
        if not (0.0 < args.constant_speed <= 1.0):
            parser.error("--constant-speed must be in (0, 1].")
        if args.no_gust:
            parser.error("Use either --no-gust or --constant-speed.")

    enable_gust = not args.no_gust and args.constant_speed is None
    print(f"[Init] GUST Safety Filter: {'ENABLED' if enable_gust else 'DISABLED (Baseline Mode)'}")

    scene = None
    ik_scene = None
    bot = None
    ik_bot = None
    ik_solver = None
    ik_wrist_handle = None
    sensor = None
    logger = None
    landscape_fn = None

    try:
        bot_path = args.path if args.path else resolve_combo_path()
        print(f"[Init] Loading robot combo from: {bot_path}")

        physics.initialize(num_worker_threads=0)

        # 1. Physics Scene
        scene = physics.create_scene("Kinova Textured Surface Scene")
        scene.set_gravity([0.0, 0.0, -9.81])

        robotics_context = robotics.create_context()
        bot = create_combo_robot(scene, bot_path, robotics_context)
        bot_actor = bot.get_articulated_actor()
        num_dofs = bot_actor.get_num_dofs()
        
        sim_links = get_robot_links_by_name(scene, bot_actor)
        sim_wrist_handle, sim_wrist_actor = sim_links[BRACELET_LINK_NAME]
        sim_tip_handle, sim_tip_actor     = sim_links[INDEX_TIP_LINK_NAME]
        _, sim_contact_actor              = sim_links[INDEX_COLLISION_LINK_NAME]
        
        # Spawn solid watertight textured surface block
        landscape_fn = spawn_visible_table(scene, center_x=0.75)

        # 2. Kinematic Twin Scene (IK)
        ik_scene = physics.create_scene("Kinova IK Scene")
        ik_bot = create_combo_robot(ik_scene, bot_path, robotics_context)
        ik_actor = ik_bot.get_articulated_actor()

        ik_links = get_robot_links_by_name(ik_scene, ik_actor)
        ik_wrist_handle, ik_wrist_actor = ik_links[BRACELET_LINK_NAME]
        ik_tip_handle, ik_tip_actor     = ik_links[INDEX_TIP_LINK_NAME]

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

        sensor = FingertipForceSensor(sim_contact_actor)
        logger = GUSTTelemetryLogger()
        gust = GUSTSafetyFilter()

        init_tip_z = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)[2]
        sensor.calibrate_free_air(init_tip_z, hover_tip_xyz[2])

        # In-Plane Cartesian Tracking PI Gains (XY plane)
        KP_CART = 0.30
        KI_CART = 0.50
        MAX_I_CLIP = 0.015
        tip_error_integral_xy = np.zeros(2, dtype=float)

        # Active Vertical Force Regulation PI Gains (Z axis)
        KP_FORCE_Z = 0.0035       # Proportional compliance: moves Z up/down based on force error
        KI_FORCE_Z = 0.0080       # Integral compliance for steady-state 2.0 N tracking
        MAX_FORCE_I = 0.015       # Integral windup clip
        MAX_Z_ADAPT_MPS = 0.020   # Actuator adaptation speed clamp (20 mm/s)
        force_error_integral_z = 0.0

        # State Machine Initialization
        STATE = "HOVER"
        cmd_tip_z = hover_tip_xyz[2]
        theta_current = 0.0       # Dynamic angle advanced via GUST Velocity Governor
        governor_scale = 1.0
        retract_finish_time = None
        step_count = 0

        physics.get_debug_server().set_coordinate_space(
            physics.CoordinateSpace(axes=physics.CoordinateSpaceAxes.FLU)
        )

        print("\n[Simulation] Connecting to debugger...")
        print(f"[Mission] Hover -> Descend to {TARGET_FORCE_N:.1f}N -> GUST Governed Contact Circle -> Retract\n")

        start_time = scene.get_total_simulation_time()

        if physics.debugger.attach():
            while physics.debugger.is_attached() and STATE != "DONE":
                sim_time = scene.get_total_simulation_time() - start_time

                # A. Read current fingertip position & force
                actual_tip_xyz = np.asarray(sim_tip_actor.get_root_transform().translation, dtype=float)
                actual_tip_z = actual_tip_xyz[2]
                current_force_z = sensor.read_force_z(actual_tip_z, cmd_tip_z)

                target_tip_xy = start_tip_xy.copy()

                # ==============================================================
                # B. State Machine Transitions with GUST
                # ==============================================================
                if STATE == "HOVER":
                    cmd_tip_z = hover_tip_xyz[2]
                    target_tip_xy = start_tip_xy.copy()
                    governor_scale = 1.0
                    if sim_time >= 0.8:
                        STATE = "DESCEND"
                        logger.mark_event("Start Descent", sim_time)
                        print(f"[{sim_time:.2f}s] Descending toward textured surface...")

                elif STATE == "DESCEND":
                    governor_scale = 1.0
                    # Decelerate descent as contact approaches
                    if current_force_z < 0.30:
                        speed = DESCENT_SPEED_MPS
                    else:
                        speed = DESCENT_SPEED_MPS * max(0.20, 1.0 - (current_force_z / TARGET_FORCE_N))

                    cmd_tip_z -= speed * TIME_STEP
                    target_tip_xy = start_tip_xy.copy()

                    # TOUCHDOWN TRIGGER at 2.0 N
                    if current_force_z >= TARGET_FORCE_N:
                        STATE = "TRACE_AND_HOLD"
                        theta_current = 0.0

                        # Register first GUST cone sample at touchdown
                        d_touchdown = abs(current_force_z - TARGET_FORCE_N)
                        gust.add_sample(actual_tip_xyz[:2], d_touchdown, sim_time)

                        tip_error_integral_xy = np.zeros(2, dtype=float)
                        force_error_integral_z = 0.0
                        logger.mark_event("Touchdown / GUST Engaged", sim_time)
                        print(f"[{sim_time:.2f}s] Touchdown at Fz = {current_force_z:.2f} N!")
                        print(f"[{sim_time:.2f}s] GUST Safety Filter Engaged. Tracing circle with active velocity governor...")

                elif STATE == "TRACE_AND_HOLD":
                    if enable_gust:
                        # 1. GUST Event-Triggered Boundary Sampling
                        if gust.should_sample(actual_tip_xyz[:2]):
                            d_meas = abs(current_force_z - TARGET_FORCE_N)
                            gust.add_sample(actual_tip_xyz[:2], d_meas, sim_time)
                            logger.mark_event(f"Cone {len(gust.sampled_points)}", sim_time)

                        # 2. Lookahead Evaluation for the Velocity Governor
                        d_theta_lookahead = GUST_LOOKAHEAD_DIST / CIRCLE_RADIUS
                        theta_lookahead = theta_current + d_theta_lookahead
                        p_lookahead = np.array([
                            circle_center_xy[0] + CIRCLE_RADIUS * np.cos(theta_lookahead),
                            circle_center_xy[1] + CIRCLE_RADIUS * np.sin(theta_lookahead),
                        ])

                        # 3. Compute Governor Speed Scaling Factor
                        governor_scale = gust.compute_governor_speed_scale(p_lookahead)
                        
                    elif args.constant_speed is not None:
                        governor_scale = args.constant_speed
                    else:
                        governor_scale = 1.0

                    # 4. Advance Trajectory Angle
                    omega_nom = 2.0 * np.pi / CIRCLE_PERIOD_SEC
                    d_theta = governor_scale * omega_nom * TIME_STEP
                    theta_current += d_theta

                    target_tip_xy = np.array([
                        circle_center_xy[0] + CIRCLE_RADIUS * np.cos(theta_current),
                        circle_center_xy[1] + CIRCLE_RADIUS * np.sin(theta_current),
                    ])

                    # 5. Active Vertical Force Regulation (Admittance/Compliance in Z)
                    force_error_z = current_force_z - TARGET_FORCE_N
                    force_error_integral_z += force_error_z * TIME_STEP
                    force_error_integral_z = np.clip(force_error_integral_z, -MAX_FORCE_I, MAX_FORCE_I)

                    # If force is too high, move tip UP (+Z); if too low, push DOWN (-Z)
                    z_adapt_speed = (KP_FORCE_Z * force_error_z) + (KI_FORCE_Z * force_error_integral_z)
                    z_adapt_speed = np.clip(z_adapt_speed, -MAX_Z_ADAPT_MPS, MAX_Z_ADAPT_MPS)
                    cmd_tip_z += z_adapt_speed * TIME_STEP

                    # Termination: Complete requested revolutions
                    if theta_current >= (TOTAL_REVOLUTIONS * 2.0 * np.pi):
                        STATE = "RETRACT"
                        retract_tip_xy = target_tip_xy.copy()
                        tip_error_integral_xy = np.zeros(2, dtype=float)
                        logger.mark_event("Trace Complete / Retract", sim_time)
                        print(f"[{sim_time:.2f}s] Completed {TOTAL_REVOLUTIONS:.0f} revolutions! Smoothly retracting...")

                elif STATE == "RETRACT":
                    governor_scale = 1.0
                    target_tip_xy = retract_tip_xy.copy()
                    cmd_tip_z += RETRACT_SPEED_MPS * TIME_STEP

                    if cmd_tip_z >= hover_tip_xyz[2]:
                        cmd_tip_z = hover_tip_xyz[2]
                        STATE = "HOVER_AFTER_RETRACT"
                        retract_finish_time = sim_time
                        logger.mark_event("Retracted", sim_time)
                        print(f"[{sim_time:.2f}s] Retracted to hover height. Holding for 1.0s...")

                elif STATE == "HOVER_AFTER_RETRACT":
                    governor_scale = 1.0
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
                full_target_xyz = np.array([target_tip_xy[0], target_tip_xy[1], cmd_tip_z])
                logger.record(
                    sim_time,
                    current_force_z,
                    actual_tip_xyz,
                    full_target_xyz,
                    governor_scale,
                    STATE,
                )

                if step_count % 40 == 0:
                    xy_err_mm = np.linalg.norm(tip_error_xy) * 1000.0
                    in_band = abs(current_force_z - TARGET_FORCE_N) <= FORCE_TOLERANCE_N
                    band_str = "[IN BAND]" if in_band else "[VIOLATION!]"
                    print(
                        f"t={sim_time:5.2f}s | [{STATE:14s}] | "
                        f"Fz: {current_force_z:4.2f} N {band_str} | "
                        f"Gov: {governor_scale:4.2f}x | "
                        f"Cones: {len(gust.sampled_points):2d} | "
                        f"Tip Z: {actual_tip_z * 1000.0:5.1f} mm | "
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
            out_filename = (
                "tactile_tracing_WITH_GUST.png" if enable_gust 
                else "tactile_tracing_WITHOUT_GUST_baseline.png"
            )
            try:
                logger.generate_two_panel_plot(
                    image_path=out_filename,
                    landscape_fn=landscape_fn,
                    gust_filter=gust if enable_gust else None,
                    is_gust_enabled=enable_gust,
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