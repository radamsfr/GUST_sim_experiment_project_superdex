"""Standalone 2D Top-Down Contour Map of the Lipschitz Surface (Viridis).

Plots the surface height perturbation h(x, y) with the robot's circular
fingertip trajectory overlaid.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class ContactSurfaceLandscape:
    def __init__(
        self,
        center_xy: tuple[float, float] = (0.48, 0.0),
        size_xy: tuple[float, float] = (1.2, 0.8),
        amplitude_mm: float = 10.0,
        sigma_mm: float = 35.0,
        num_centers: int = 32,
        seed: int = 101,
    ):
        self.center_xy = np.array(center_xy, dtype=float)
        self.size_xy = np.array(size_xy, dtype=float)
        self.amplitude_mm = amplitude_mm
        self.sigma = sigma_mm / 1000.0

        rng = np.random.default_rng(seed)
        cx, cy = self.center_xy
        sx, sy = self.size_xy
        pad = 0.02

        self.centers = np.column_stack([
            rng.uniform(cx - sx / 2 - pad, cx + sx / 2 + pad, num_centers),
            rng.uniform(cy - sy / 2 - pad, cy + sy / 2 + pad, num_centers),
        ])
        self.weights = rng.uniform(-1.0, 1.0, num_centers)

    def evaluate_h_mm(self, coords_m: np.ndarray) -> np.ndarray:
        coords_m = np.atleast_2d(coords_m)
        diffs = coords_m[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        dist_sq = np.sum(diffs**2, axis=-1)
        raw = np.sum(self.weights * np.exp(-dist_sq / (2.0 * self.sigma**2)), axis=-1)
        raw -= np.mean(raw)

        max_raw = max(np.max(np.abs(raw)), 1e-6)
        h = self.amplitude_mm * (raw / max_raw)

        # Smooth window to blend into table edges
        dist_center = np.linalg.norm(coords_m - self.center_xy, axis=-1)
        max_r = 0.45 * max(self.size_xy)
        window = np.clip(1.0 - (dist_center / max_r) ** 2, 0.0, 1.0) ** 2
        return h * window


def plot_2d_contour_map(
    output_image: str = "lipschitz_surface_2d.png",
    grid_resolution: int = 200,
    circle_radius_m: float = 0.08,
):
    landscape = ContactSurfaceLandscape()
    cx, cy = landscape.center_xy
    sx, sy = landscape.size_xy

    # 1. High-Density Spatial Grid
    xs = np.linspace(cx - sx / 2, cx + sx / 2, grid_resolution)
    ys = np.linspace(cy - sy / 2, cy + sy / 2, grid_resolution)
    GX, GY = np.meshgrid(xs, ys)
    grid_points = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    H_mm = landscape.evaluate_h_mm(grid_points).reshape(GX.shape)

    # 2. Circular Trajectory Points
    thetas = np.linspace(0, 2 * np.pi, 250)
    traj_x = cx + circle_radius_m * np.cos(thetas)
    traj_y = cy + circle_radius_m * np.sin(thetas)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    # Filled contour map (Viridis)
    cf = ax.contourf(
        GX * 1000.0,
        GY * 1000.0,
        H_mm,
        levels=50,
        cmap="viridis",
    )

    # Contour line overlays with elevation labels
    cs = ax.contour(
        GX * 1000.0,
        GY * 1000.0,
        H_mm,
        levels=10,
        colors="white",
        linewidths=0.6,
        alpha=0.45,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f mm")

    # Fingertip Circular Path
    ax.plot(
        traj_x * 1000.0,
        traj_y * 1000.0,
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        label=f"Fingertip Path (R={circle_radius_m*1000:.0f} mm)",
    )

    # Touchdown / Start Point (theta = 0)
    ax.plot(
        traj_x[0] * 1000.0,
        traj_y[0] * 1000.0,
        "o",
        color="gold",
        markersize=10,
        markeredgecolor="black",
        label="Touchdown (Start)",
        zorder=10,
    )

    # Direction Arrow (at theta = pi/2)
    arrow_idx = len(thetas) // 4
    ax.annotate(
        "",
        xy=(traj_x[arrow_idx + 4] * 1000.0, traj_y[arrow_idx + 4] * 1000.0),
        xytext=(traj_x[arrow_idx] * 1000.0, traj_y[arrow_idx] * 1000.0),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=2.5, mutation_scale=20),
    )

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Surface Height Perturbation $h(x, y)$ [mm]", rotation=270, labelpad=20, fontsize=11)

    # Labels and Formatting
    ax.set_title("Top-Down Contact Surface Topography (Viridis)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Table X [mm]", fontsize=11)
    ax.set_ylabel("Table Y [mm]", fontsize=11)
    ax.axis("equal")
    ax.grid(True, linestyle=":", alpha=0.5, color="white")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"[Done] 2D Viridis contour map saved to: {output_image}")
    plt.show()


if __name__ == "__main__":
    plot_2d_contour_map()