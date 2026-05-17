"""Differentiable parametric geometry primitives.

Implements a minimal domain-specific language (DSL) of CAD-style geometry
operators whose parameters are continuous and differentiable, following the
framework described in:

    "Differentiable Parametric Geometry Generation via Quality-Diversity Search"
    MaxOSL AI Research, 2026

Each primitive is a function from parameters → geometric descriptor dict.
The descriptor includes axis-aligned bounding box (AABB), volume, surface area,
cross-sectional moments of inertia, and a sampled point cloud — all computed
from the continuous parameters so that gradient-based parameter tuning is
possible when the caller uses a differentiable runtime (JAX, PyTorch).

Here the primitives are implemented in plain NumPy so they work everywhere.
To enable autodiff: wrap parameters in jax.numpy arrays and replace np with jnp.

Primitive hierarchy
-------------------
LinearExtrusion    — rectangular cross-section swept along Z
Revolution         — rectangle rotated around Z axis (annular solid)
Sweep              — circular cross-section along a quadratic spine
BooleanUnion       — merge two primitives (bounding box approximation)
BooleanDiff        — subtract one primitive from another (volume approx)
FilletBox          — rounded-corner rectangular extrusion
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Descriptor returned by every primitive
# ---------------------------------------------------------------------------

@dataclass
class GeomDescriptor:
    """Axis-aligned geometric descriptor for a parametric solid."""
    primitive: str
    parameters: dict[str, float]
    # Bounding box (half-extents from origin)
    aabb_min: np.ndarray          # shape (3,)
    aabb_max: np.ndarray          # shape (3,)
    volume_m3: float
    surface_area_m2: float
    # Cross-section at z=0
    cross_section_area_m2: float
    second_moment_Ixx: float      # bending stiffness proxy (m⁴)
    second_moment_Iyy: float
    # Derived scalars used as MAP-Elites behavioral dimensions
    aspect_ratio: float           # max_extent / min_extent
    wall_thickness_min_m: float   # smallest wall (for castability)
    # Sampled point cloud (n×3, object coordinates)
    point_cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "primitive": self.primitive,
            "parameters": self.parameters,
            "aabb_min": self.aabb_min.tolist(),
            "aabb_max": self.aabb_max.tolist(),
            "volume_m3": self.volume_m3,
            "surface_area_m2": self.surface_area_m2,
            "cross_section_area_m2": self.cross_section_area_m2,
            "second_moment_Ixx_m4": self.second_moment_Ixx,
            "second_moment_Iyy_m4": self.second_moment_Iyy,
            "aspect_ratio": self.aspect_ratio,
            "wall_thickness_min_m": self.wall_thickness_min_m,
            "n_points": len(self.point_cloud),
        }
        return d


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def linear_extrusion(
    width: float,
    depth: float,
    height: float,
    taper_angle_deg: float = 0.0,
    n_points: int = 200,
) -> GeomDescriptor:
    """Rectangular cross-section extruded along Z.

    Parameters
    ----------
    width, depth, height : float
        Dimensions in metres.
    taper_angle_deg : float
        Draft angle — cross-section linearly shrinks from base to top.
        0° = prismatic; positive = tapers inward (positive draft for casting).
    """
    w, d, h = abs(float(width)), abs(float(depth)), abs(float(height))
    alpha = math.radians(abs(float(taper_angle_deg)))

    # Width and depth at top after taper
    w_top = max(1e-9, w - 2.0 * h * math.tan(alpha))
    d_top = max(1e-9, d - 2.0 * h * math.tan(alpha))

    # Volume = frustum formula for trapezoidal cross-section
    A_bot = w * d
    A_top = w_top * d_top
    volume = h / 3.0 * (A_bot + A_top + math.sqrt(A_bot * A_top))

    # Surface area (4 side faces + 2 caps, approximate for small taper)
    side_l1 = h / math.cos(alpha)  # slant height width face
    side_l2 = h / math.cos(alpha)  # slant height depth face
    surface = A_bot + A_top + (w + w_top) * side_l1 + (d + d_top) * side_l2

    # Cross section at mid-height
    w_mid = (w + w_top) / 2.0
    d_mid = (d + d_top) / 2.0
    Ixx = w_mid * d_mid ** 3 / 12.0
    Iyy = d_mid * w_mid ** 3 / 12.0

    # Point cloud: sample uniformly inside the frustum
    rng = np.random.default_rng(0)
    z_pts = rng.uniform(0, h, n_points)
    frac = z_pts / h
    w_z = w - 2.0 * z_pts * math.tan(alpha)
    d_z = d - 2.0 * z_pts * math.tan(alpha)
    x_pts = rng.uniform(-0.5, 0.5, n_points) * np.maximum(w_z, 1e-9)
    y_pts = rng.uniform(-0.5, 0.5, n_points) * np.maximum(d_z, 1e-9)
    pts = np.stack([x_pts, y_pts, z_pts], axis=1)

    extents = np.array([w, d, h])
    aspect = max(w, d, h) / max(min(w, d, h), 1e-9)
    wall_min = min(w, d, h)

    return GeomDescriptor(
        primitive="linear_extrusion",
        parameters={"width": w, "depth": d, "height": h, "taper_angle_deg": float(taper_angle_deg)},
        aabb_min=np.array([-w/2, -d/2, 0.0]),
        aabb_max=np.array([w/2, d/2, h]),
        volume_m3=volume,
        surface_area_m2=surface,
        cross_section_area_m2=w_mid * d_mid,
        second_moment_Ixx=Ixx,
        second_moment_Iyy=Iyy,
        aspect_ratio=aspect,
        wall_thickness_min_m=wall_min,
        point_cloud=pts,
    )


def revolution(
    outer_radius: float,
    inner_radius: float,
    height: float,
    n_points: int = 200,
) -> GeomDescriptor:
    """Annular solid of revolution (hollow cylinder).

    Parameters
    ----------
    outer_radius, inner_radius : float
        Outer and inner radii in metres.  inner_radius = 0 → solid cylinder.
    height : float
        Height in metres.
    """
    R = abs(float(outer_radius))
    r = min(abs(float(inner_radius)), R - 1e-9)
    h = abs(float(height))

    volume = math.pi * (R**2 - r**2) * h
    lateral_area = 2.0 * math.pi * (R + r) * h
    cap_area = 2.0 * math.pi * (R**2 - r**2)
    surface = lateral_area + cap_area

    # Cross-section (annular ring) second moments
    Ixx = math.pi / 4.0 * (R**4 - r**4)
    Iyy = Ixx

    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * math.pi, n_points)
    rho = np.sqrt(rng.uniform(r**2, R**2, n_points))
    z_pts = rng.uniform(0, h, n_points)
    pts = np.stack([rho * np.cos(theta), rho * np.sin(theta), z_pts], axis=1)

    aspect = h / max(2 * R, 1e-9)
    wall = R - r

    return GeomDescriptor(
        primitive="revolution",
        parameters={"outer_radius": R, "inner_radius": r, "height": h},
        aabb_min=np.array([-R, -R, 0.0]),
        aabb_max=np.array([R, R, h]),
        volume_m3=volume,
        surface_area_m2=surface,
        cross_section_area_m2=math.pi * (R**2 - r**2),
        second_moment_Ixx=Ixx,
        second_moment_Iyy=Iyy,
        aspect_ratio=aspect,
        wall_thickness_min_m=wall,
        point_cloud=pts,
    )


def sweep(
    radius: float,
    spine_length: float,
    spine_curve: float = 0.0,
    n_points: int = 200,
) -> GeomDescriptor:
    """Circular cross-section swept along a quadratic spine curve.

    Parameters
    ----------
    radius : float
        Cross-section radius in metres.
    spine_length : float
        Arc length of the sweep path in metres.
    spine_curve : float
        Quadratic curvature parameter.  0 = straight pipe.
        Positive = bends upward (lateral deflection = curve * L²).
    """
    R = abs(float(radius))
    L = abs(float(spine_length))
    c = float(spine_curve)

    arc_length = L * math.sqrt(1.0 + (2.0 * c * L) ** 2) if abs(c) > 1e-9 else L
    volume = math.pi * R**2 * arc_length
    surface = 2.0 * math.pi * R * arc_length + 2.0 * math.pi * R**2

    Ixx = math.pi * R**4 / 4.0
    Iyy = Ixx

    rng = np.random.default_rng(0)
    t = rng.uniform(0, 1, n_points)
    z_spine = t * L
    y_spine = c * z_spine**2
    theta = rng.uniform(0, 2 * math.pi, n_points)
    rho = np.sqrt(rng.uniform(0, R**2, n_points))
    pts = np.stack([rho * np.cos(theta), rho * np.sin(theta) + y_spine, z_spine], axis=1)

    max_lateral = abs(c) * L**2
    aabb_y_max = R + max_lateral
    aspect = max(L, 2*R) / max(min(L, 2*R), 1e-9)

    return GeomDescriptor(
        primitive="sweep",
        parameters={"radius": R, "spine_length": L, "spine_curve": c},
        aabb_min=np.array([-R, -R, 0.0]),
        aabb_max=np.array([R, aabb_y_max + R, L]),
        volume_m3=volume,
        surface_area_m2=surface,
        cross_section_area_m2=math.pi * R**2,
        second_moment_Ixx=Ixx,
        second_moment_Iyy=Iyy,
        aspect_ratio=aspect,
        wall_thickness_min_m=R,
        point_cloud=pts,
    )


def fillet_box(
    width: float,
    depth: float,
    height: float,
    fillet_radius: float,
    n_points: int = 200,
) -> GeomDescriptor:
    """Rectangular extrusion with rounded vertical edges (fillet).

    Fillets reduce stress concentrations and are required for forged/cast parts.

    Parameters
    ----------
    fillet_radius : float
        Edge rounding radius in metres.  Clamped to min(width, depth)/2.
    """
    w, d, h = abs(float(width)), abs(float(depth)), abs(float(height))
    fr = min(abs(float(fillet_radius)), min(w, d) / 2.0)

    # Volume: box minus 4 corner cylinders plus 4 quarter-cylinders
    corner_loss = (1.0 - math.pi / 4.0) * fr**2 * h
    volume = w * d * h - 4.0 * corner_loss

    surface = 2.0 * (w * d - (4.0 - math.pi) * fr**2) + 2.0 * h * (w + d - (4.0 - math.pi) * fr)
    cs_area = w * d - (4.0 - math.pi) * fr**2
    Ixx = (w * d**3 / 12.0) - (4.0 - math.pi) * fr**2 * (d/2.0)**2 / 4.0
    Iyy = (d * w**3 / 12.0) - (4.0 - math.pi) * fr**2 * (w/2.0)**2 / 4.0

    rng = np.random.default_rng(0)
    x_pts = rng.uniform(-w/2, w/2, n_points)
    y_pts = rng.uniform(-d/2, d/2, n_points)
    z_pts = rng.uniform(0, h, n_points)
    # Rough in-bounds check for corners
    in_box = (
        (np.abs(x_pts) <= w/2 - fr) | (np.abs(y_pts) <= d/2 - fr) |
        ((np.abs(x_pts) - (w/2 - fr))**2 + (np.abs(y_pts) - (d/2 - fr))**2 <= fr**2)
    )
    pts = np.stack([x_pts[in_box], y_pts[in_box], z_pts[in_box]], axis=1)

    aspect = max(w, d, h) / max(min(w, d, h), 1e-9)

    return GeomDescriptor(
        primitive="fillet_box",
        parameters={"width": w, "depth": d, "height": h, "fillet_radius": fr},
        aabb_min=np.array([-w/2, -d/2, 0.0]),
        aabb_max=np.array([w/2, d/2, h]),
        volume_m3=volume,
        surface_area_m2=surface,
        cross_section_area_m2=cs_area,
        second_moment_Ixx=max(Ixx, 1e-30),
        second_moment_Iyy=max(Iyy, 1e-30),
        aspect_ratio=aspect,
        wall_thickness_min_m=min(w, d, h),
        point_cloud=pts,
    )


def boolean_union(a: GeomDescriptor, b: GeomDescriptor) -> GeomDescriptor:
    """Bounding-box approximation of the boolean union of two solids."""
    bb_min = np.minimum(a.aabb_min, b.aabb_min)
    bb_max = np.maximum(a.aabb_max, b.aabb_max)
    extents = bb_max - bb_min
    volume = float(a.volume_m3 + b.volume_m3)   # approximate (ignores overlap)
    surface = float(a.surface_area_m2 + b.surface_area_m2) * 0.85  # overlap factor
    aspect = float(np.max(extents) / max(np.min(extents), 1e-9))
    pts = np.concatenate([a.point_cloud, b.point_cloud], axis=0)
    return GeomDescriptor(
        primitive="boolean_union",
        parameters={"child_a": a.primitive, "child_b": b.primitive},
        aabb_min=bb_min, aabb_max=bb_max,
        volume_m3=volume, surface_area_m2=surface,
        cross_section_area_m2=a.cross_section_area_m2 + b.cross_section_area_m2,
        second_moment_Ixx=a.second_moment_Ixx + b.second_moment_Ixx,
        second_moment_Iyy=a.second_moment_Iyy + b.second_moment_Iyy,
        aspect_ratio=aspect,
        wall_thickness_min_m=min(a.wall_thickness_min_m, b.wall_thickness_min_m),
        point_cloud=pts,
    )


def boolean_diff(base: GeomDescriptor, subtract: GeomDescriptor) -> GeomDescriptor:
    """Approximate boolean difference: subtract volume from base."""
    volume = max(1e-30, base.volume_m3 - subtract.volume_m3 * 0.5)
    surface = base.surface_area_m2 + subtract.surface_area_m2 * 0.5
    aspect = float(np.max(base.aabb_max - base.aabb_min) /
                   max(np.min(base.aabb_max - base.aabb_min), 1e-9))
    return GeomDescriptor(
        primitive="boolean_diff",
        parameters={"base": base.primitive, "subtract": subtract.primitive},
        aabb_min=base.aabb_min.copy(), aabb_max=base.aabb_max.copy(),
        volume_m3=volume, surface_area_m2=surface,
        cross_section_area_m2=max(1e-30, base.cross_section_area_m2 - subtract.cross_section_area_m2),
        second_moment_Ixx=max(1e-30, base.second_moment_Ixx - subtract.second_moment_Ixx),
        second_moment_Iyy=max(1e-30, base.second_moment_Iyy - subtract.second_moment_Iyy),
        aspect_ratio=aspect,
        wall_thickness_min_m=base.wall_thickness_min_m,
        point_cloud=base.point_cloud,
    )


# ---------------------------------------------------------------------------
# Constraint loss functions (composable, differentiable in JAX)
# ---------------------------------------------------------------------------

def loss_target_volume(geom: GeomDescriptor, target_m3: float, weight: float = 1.0) -> float:
    """Squared relative error between geometry volume and target."""
    return weight * ((geom.volume_m3 - target_m3) / max(target_m3, 1e-9)) ** 2


def loss_aspect_ratio(geom: GeomDescriptor, target_ar: float, weight: float = 1.0) -> float:
    """Squared error on aspect ratio."""
    return weight * ((geom.aspect_ratio - target_ar) / max(target_ar, 1e-9)) ** 2


def loss_min_wall(geom: GeomDescriptor, min_wall_m: float, weight: float = 1.0) -> float:
    """Hinge loss penalising walls thinner than min_wall_m (manufacturability)."""
    violation = max(0.0, min_wall_m - geom.wall_thickness_min_m)
    return weight * (violation / max(min_wall_m, 1e-9)) ** 2


def loss_draft_angle(taper_angle_deg: float, min_draft_deg: float = 1.0,
                     weight: float = 1.0) -> float:
    """Hinge loss for insufficient draft angle (castability)."""
    violation = max(0.0, min_draft_deg - taper_angle_deg)
    return weight * (violation / max(min_draft_deg, 1e-9)) ** 2


def total_constraint_loss(
    geom: GeomDescriptor,
    target_volume_m3: float | None = None,
    target_aspect_ratio: float | None = None,
    min_wall_m: float | None = None,
    taper_angle_deg: float | None = None,
    min_draft_deg: float = 1.0,
) -> dict[str, float]:
    """Sum all enabled constraint losses and return a breakdown."""
    losses: dict[str, float] = {}
    if target_volume_m3 is not None:
        losses["volume"] = loss_target_volume(geom, target_volume_m3)
    if target_aspect_ratio is not None:
        losses["aspect_ratio"] = loss_aspect_ratio(geom, target_aspect_ratio)
    if min_wall_m is not None:
        losses["min_wall"] = loss_min_wall(geom, min_wall_m)
    if taper_angle_deg is not None:
        losses["draft_angle"] = loss_draft_angle(taper_angle_deg, min_draft_deg)
    losses["total"] = sum(losses.values())
    return losses
