"""Scene sampler.

Public API:
- :class:`SceneConfig` — sampling knobs (the things you might want to
  tune).
- :class:`SceneSampler` — call ``.sample()`` for one scene or
  ``.sample_batch(n=N)`` for N. Failures raise :class:`SamplingError`
  with a descriptive message.

Algorithm: structured hard-side sampling (every variant has one close
obstacle and one far obstacle; coin-flip picks which side is hard).
See docs/experiments/tomato_reach_scene.md §3.2 for the design rationale.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import constants as C
from .geometry import box_aabb_extent, box_circle_distance
from .grasp import canonical_top_down_grasp
from .scene import ObjectPlacement, Scene, SceneParams


class SamplingError(RuntimeError):
    """Raised when the sampler can't find a feasible scene within retries."""


@dataclass(frozen=True)
class SceneConfig:
    """Knobs that control scene generation.

    Default values are calibrated for the chocopie-replaced cocopops
    scene on the EKENÄBBEN 2nd shelf with a Franka Panda. Adjust as
    needed for harder / easier scenes.

    Attributes
    ----------
    k_mm
        Edge-to-edge gap between A (rice noodle) and D (soup can), in mm.
        Range ``[K_MIN_MM, K_MAX_MM]``. Small → easier (smaller roundabout
        needed); large → harder.
    r_swing_mm
        Estimated clearance threshold around the central column (A + D),
        in mm. Derived from the Panda URDF: ``r_A + hand_half_thickness
        + safety`` (≈ 125 mm with the standard hand at tangent
        orientation). An obstacle's centre at less than ``r_swing_mm``
        from the composite origin is treated as "blocks the swing on
        that side"; greater, as "clear."

        Note this is a *soft* clearance estimate for scene labelling,
        not a hard reachability rule — the true reachability check
        happens in the downstream planner against the URDF.
    clearance_buffer_mm
        Buffer around ``r_swing_mm`` within which no obstacle centre is
        allowed to land. Hard-side obstacles are sampled from
        ``[d_min, r_swing - buffer]``; easy-side from
        ``[r_swing + buffer, d_max]``. The forbidden zone
        ``(r_swing - buffer, r_swing + buffer)`` excludes "borderline"
        obstacles.

        Why this matters (pedagogical, not just geometric): the
        structured hard-side design's whole point is to teach the
        downstream policy that one side is *clearly* harder than the
        other. Without a buffer, the close band's upper edge and the
        far band's lower edge meet at ``r_swing``, and the sampler can
        produce scenes where both obstacles land within a millimetre
        of the threshold — ambiguous on both sides. Such scenes
        produce a noisy training signal and undermine the experiment's
        ability to test "policy picks the side with more clearance."

        Default 15 mm matches the safety margin already baked into
        ``r_swing_mm``'s derivation, so the epistemic uncertainty in
        the clearance threshold is symmetrically respected on both
        sides. Set to 0 to recover the legacy (uniform-touching-bands)
        sampler.
    d_min_mm
        Lower bound on E / C centre distance from composite origin, in mm.
        Must be ≤ ``r_swing_mm - clearance_buffer_mm`` (otherwise close band
        is empty).
    d_max_mm
        Upper bound on E / C centre distance from composite origin, in mm.
        Must be ≥ ``r_swing_mm + clearance_buffer_mm`` (otherwise far band
        is empty). Capped by slab extent in practice.
    both_easy_probability
        Probability ``∈ [0, 1]`` that a sampled scene is "both easy" —
        i.e., **no hard side**: *both* obstacles are drawn from the easy
        band ``[r_swing + clearance_buffer, d_max]``. The remaining
        ``1 − p`` of scenes go through the usual structured hard-side
        coin flip (left vs right) where exactly one obstacle is close.

        Use this when the downstream policy needs training data
        covering the "both swing paths are clear" case — otherwise the
        policy can overfit to "always find the blocked side" and lose
        the ability to take whichever path is shortest in trivially
        easy scenes.

        Default ``0.33`` — gives a roughly even distribution across the
        three scene classes (``left`` ≈ 33%, ``right`` ≈ 33%,
        ``neither`` ≈ 33%) so the policy sees all cases in similar
        proportion during training. Set to ``0.0`` to recover the
        original two-case behaviour (only structured hard-side coin
        flips). The accepted scene's :attr:`SceneParams.hard_side` is
        recorded as ``"neither"`` when both-easy is sampled, so the
        manifest can bucket scenes for analysis.
    e_yaw_mode
        ``"random"`` — sample E's yaw uniformly from [0°, 180°) per scene.
        ``"fixed"`` — use ``e_yaw_deg`` every time.
    e_yaw_deg
        Used only when ``e_yaw_mode == "fixed"``.
    seed
        RNG seed. Same seed + same other knobs ⇒ identical scene sequence.
    max_tries_per_scene
        Rejection-sampling retry budget per call to ``sample()``.
    grasp_approach_offset_m
        Distance above B's top face where the gripper TCP stops, metres.
    """
    k_mm: float = 60.0
    r_swing_mm: float = C.R_SWING_DEFAULT_MM
    clearance_buffer_mm: float = 15.0
    d_min_mm: float = 100.0
    d_max_mm: float = 300.0
    both_easy_probability: float = 0.33
    e_yaw_mode: Literal["random", "fixed"] = "random"
    e_yaw_deg: float = 0.0
    seed: int = 0
    max_tries_per_scene: int = 500
    grasp_approach_offset_m: float = 0.005

    def __post_init__(self) -> None:
        if not (C.K_MIN_MM <= self.k_mm <= C.K_MAX_MM):
            raise ValueError(
                f"k_mm={self.k_mm} out of geometric bound "
                f"[{C.K_MIN_MM}, {C.K_MAX_MM}] — A would overhang slab."
            )
        if self.clearance_buffer_mm < 0:
            raise ValueError(
                f"clearance_buffer_mm={self.clearance_buffer_mm} must be ≥ 0"
            )
        close_upper = self.r_swing_mm - self.clearance_buffer_mm
        far_lower = self.r_swing_mm + self.clearance_buffer_mm
        if self.d_min_mm > close_upper:
            raise ValueError(
                f"d_min_mm ({self.d_min_mm}) must be ≤ "
                f"r_swing_mm - clearance_buffer_mm "
                f"({self.r_swing_mm} - {self.clearance_buffer_mm} = "
                f"{close_upper}); otherwise close band is empty."
            )
        if self.d_max_mm < far_lower:
            raise ValueError(
                f"d_max_mm ({self.d_max_mm}) must be ≥ "
                f"r_swing_mm + clearance_buffer_mm "
                f"({self.r_swing_mm} + {self.clearance_buffer_mm} = "
                f"{far_lower}); otherwise far band is empty."
            )
        if self.e_yaw_mode not in ("random", "fixed"):
            raise ValueError(f"e_yaw_mode={self.e_yaw_mode!r}; expected 'random' or 'fixed'")
        if not (0.0 <= self.both_easy_probability <= 1.0):
            raise ValueError(
                f"both_easy_probability={self.both_easy_probability}; must be ∈ [0, 1]"
            )


class SceneSampler:
    """Sample scenes governed by a :class:`SceneConfig`.

    Stateless apart from the RNG. ``sample()`` advances the RNG per
    call so successive samples differ.
    """

    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def sample(self) -> Scene:
        """Draw one feasible scene. Raises :class:`SamplingError` on failure."""
        params = self._sample_params()
        return self._assemble_scene(params)

    def sample_batch(self, n: int) -> list[Scene]:
        """Draw ``n`` feasible scenes. Raises :class:`SamplingError` if any
        single sample call exhausts its retry budget."""
        return [self.sample() for _ in range(n)]

    # ----- internals -----

    def _sample_params(self) -> SceneParams:
        cfg = self.config
        rng = self._rng
        k_m = cfg.k_mm / 1000.0
        d_min_m = cfg.d_min_mm / 1000.0
        d_max_m = cfg.d_max_mm / 1000.0
        # The clearance buffer carves a "no-sample" zone of width
        # 2*clearance_buffer_mm around r_swing_mm. Hard-side obstacles
        # are drawn from [d_min, r_swing - buffer]; easy-side from
        # [r_swing + buffer, d_max]. This guarantees that every
        # accepted scene has a clearly-blocking obstacle on one side
        # and a clearly-clear obstacle on the other, instead of two
        # ambiguous obstacles straddling r_swing. See the SceneConfig
        # docstring for the rationale.
        hard_upper_m = (cfg.r_swing_mm - cfg.clearance_buffer_mm) / 1000.0
        easy_lower_m = (cfg.r_swing_mm + cfg.clearance_buffer_mm) / 1000.0

        # A and D positions are deterministic from k.
        x_a = +(k_m / 2 + C.R_A)
        x_d = -(k_m / 2 + C.R_D)

        for _ in range(cfg.max_tries_per_scene):
            # 1. Decide scene type — "neither" (both easy) with probability
            #    cfg.both_easy_probability, else fair coin flip for which
            #    side is hard.
            hard_side: Literal["left", "right", "neither"]
            if (
                cfg.both_easy_probability > 0.0
                and rng.random() < cfg.both_easy_probability
            ):
                hard_side = "neither"
                dist_e = rng.uniform(easy_lower_m, d_max_m)
                dist_c = rng.uniform(easy_lower_m, d_max_m)
            else:
                hard_side = rng.choice(("right", "left"))
                if hard_side == "right":
                    dist_e = rng.uniform(d_min_m, hard_upper_m)
                    dist_c = rng.uniform(easy_lower_m, d_max_m)
                else:
                    dist_e = rng.uniform(easy_lower_m, d_max_m)
                    dist_c = rng.uniform(d_min_m, hard_upper_m)

            # 2. Random angle inside each obstacle's half-plane.
            theta_e = rng.uniform(-math.pi, 0.0)   # E in -y half-plane
            theta_c = rng.uniform(0.0, math.pi)    # C in +y half-plane
            x_e = dist_e * math.cos(theta_e)
            y_e = dist_e * math.sin(theta_e)
            x_c = dist_c * math.cos(theta_c)
            y_c = dist_c * math.sin(theta_c)

            # 3. E yaw.
            if cfg.e_yaw_mode == "random":
                e_yaw = rng.uniform(0.0, math.pi)   # 2-fold symmetry
            else:
                e_yaw = math.radians(cfg.e_yaw_deg)

            # 4. Rejection: slab + overlap.
            e_aabb_x, e_aabb_y = box_aabb_extent(
                C.E_HALF_EXTENTS_M[0], C.E_HALF_EXTENTS_M[1], e_yaw,
            )
            # Slab bounds: x along slab long axis, y along slab short axis.
            if abs(x_e) + e_aabb_x > C.SLAB_X_HALF_M: continue
            if abs(y_e) + e_aabb_y > C.SLAB_Y_HALF_M: continue
            if abs(x_c) + C.R_C > C.SLAB_X_HALF_M: continue
            if abs(y_c) + C.R_C > C.SLAB_Y_HALF_M: continue
            # E vs A / D.
            if box_circle_distance(
                (x_e, y_e), C.E_HALF_EXTENTS_M[0], C.E_HALF_EXTENTS_M[1],
                e_yaw, (x_a, 0.0), C.R_A,
            ) <= 0: continue
            if box_circle_distance(
                (x_e, y_e), C.E_HALF_EXTENTS_M[0], C.E_HALF_EXTENTS_M[1],
                e_yaw, (x_d, 0.0), C.R_D,
            ) <= 0: continue
            # C vs A / D.
            if math.hypot(x_c - x_a, y_c) <= C.R_C + C.R_A: continue
            if math.hypot(x_c - x_d, y_c) <= C.R_C + C.R_D: continue
            # E vs C.
            if box_circle_distance(
                (x_e, y_e), C.E_HALF_EXTENTS_M[0], C.E_HALF_EXTENTS_M[1],
                e_yaw, (x_c, y_c), C.R_C,
            ) <= 0: continue

            # Accept.
            return SceneParams(
                k_mm=cfg.k_mm,
                hard_side=hard_side,
                e_xy_mm=(x_e * 1000.0, y_e * 1000.0),
                c_xy_mm=(x_c * 1000.0, y_c * 1000.0),
                e_yaw_deg=math.degrees(e_yaw),
                dist_e_mm=math.hypot(x_e, y_e) * 1000.0,
                dist_c_mm=math.hypot(x_c, y_c) * 1000.0,
            )

        raise SamplingError(
            f"could not find a feasible scene in {cfg.max_tries_per_scene} tries; "
            f"check that d_min < r_swing < d_max and k_mm leaves room for E/C."
        )

    def _assemble_scene(self, params: SceneParams) -> Scene:
        """Materialise the Scene from sampling params."""
        k_m = params.k_mm / 1000.0
        x_a = +(k_m / 2 + C.R_A)
        x_d = -(k_m / 2 + C.R_D)
        x_e_m = params.e_xy_mm[0] / 1000.0
        y_e_m = params.e_xy_mm[1] / 1000.0
        x_c_m = params.c_xy_mm[0] / 1000.0
        y_c_m = params.c_xy_mm[1] / 1000.0
        e_yaw_rad = math.radians(params.e_yaw_deg)

        B_T_S = C.composite_pose_in_base()
        primitives: dict[str, ObjectPlacement] = {}

        # Shelf primitives (frozen, in base frame directly).
        for sp in C.SHELF_PRIMITIVES_BASE:
            primitives[sp["name"]] = ObjectPlacement(
                name=sp["name"], type=sp["type"], dims_m=tuple(sp["dims_m"]),
                pose_xyz_m=tuple(sp["pose_xyz_m"]),
                pose_quat_xyzw=tuple(sp["pose_quat_xyzw"]),
            )

        # Scene primitives — compose composite-frame placements into base frame.
        # Composite R = identity, so the rotation column is just the primitive's own
        # rotation in composite; translation is shifted by composite origin.

        # A — cylinder
        primitives["A_rice_noodle"] = _cylinder_in_base(
            name="A_rice_noodle",
            radius_m=C.R_A,
            height_m=C.A_HEIGHT_M,
            center_xy_composite_m=(x_a, 0.0),
            B_T_S=B_T_S,
        )

        # B — cylinder concentric on A's top
        primitives["B_tomato"] = _cylinder_in_base(
            name="B_tomato",
            radius_m=C.R_B,
            height_m=C.B_HEIGHT_M,
            center_xy_composite_m=(x_a, 0.0),
            B_T_S=B_T_S,
            z_bottom_composite_m=C.A_HEIGHT_M,    # sits on A's top
        )

        # D — cylinder
        primitives["D_soup_can"] = _cylinder_in_base(
            name="D_soup_can",
            radius_m=C.R_D,
            height_m=C.D_HEIGHT_M,
            center_xy_composite_m=(x_d, 0.0),
            B_T_S=B_T_S,
        )

        # E — box with yaw
        primitives["E_cocopops"] = _box_in_base(
            name="E_cocopops",
            half_extents_m=C.E_HALF_EXTENTS_M,
            center_xy_composite_m=(x_e_m, y_e_m),
            yaw_rad=e_yaw_rad,
            B_T_S=B_T_S,
        )

        # C — cylinder
        primitives["C_aqua_bottle"] = _cylinder_in_base(
            name="C_aqua_bottle",
            radius_m=C.R_C,
            height_m=C.C_HEIGHT_M,
            center_xy_composite_m=(x_c_m, y_c_m),
            B_T_S=B_T_S,
        )

        # Grasp pose: canonical top-down at B's top.
        grasp_xyz, grasp_quat = canonical_top_down_grasp(
            b_center_xyz_composite_m=(x_a, 0.0, C.A_HEIGHT_M + C.B_HEIGHT_M / 2),
            b_top_z_composite_m=C.A_HEIGHT_M + C.B_HEIGHT_M,
            approach_offset_m=self.config.grasp_approach_offset_m,
        )

        return Scene(
            primitives=primitives,
            grasp_pose_xyz_m=grasp_xyz,
            grasp_pose_quat_xyzw=grasp_quat,
            params=params,
        )


# ----- private helpers (build ObjectPlacement entries in base frame) -----


def _cylinder_in_base(
    *, name: str, radius_m: float, height_m: float,
    center_xy_composite_m: tuple[float, float],
    B_T_S: np.ndarray,
    z_bottom_composite_m: float = 0.0,
) -> ObjectPlacement:
    """Build an axis-aligned upright cylinder placement in base frame."""
    centre_composite = np.array([
        center_xy_composite_m[0],
        center_xy_composite_m[1],
        z_bottom_composite_m + height_m / 2,
    ], dtype=np.float64)
    centre_base = B_T_S[:3, :3] @ centre_composite + B_T_S[:3, 3]
    # Composite R is identity in base, so the cylinder's own R = identity too.
    return ObjectPlacement(
        name=name, type="cylinder",
        dims_m=(radius_m, height_m, 0.0),
        pose_xyz_m=(float(centre_base[0]), float(centre_base[1]), float(centre_base[2])),
        pose_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def _box_in_base(
    *, name: str, half_extents_m: tuple[float, float, float],
    center_xy_composite_m: tuple[float, float],
    yaw_rad: float,
    B_T_S: np.ndarray,
) -> ObjectPlacement:
    """Build a box placement at z = half-z above composite floor, yawed in xy."""
    hx, hy, hz = half_extents_m
    centre_composite = np.array([
        center_xy_composite_m[0],
        center_xy_composite_m[1],
        hz,
    ], dtype=np.float64)
    centre_base = B_T_S[:3, :3] @ centre_composite + B_T_S[:3, 3]
    # Yaw quaternion (rotation about composite +z = base +z).
    qw = math.cos(yaw_rad / 2)
    qz = math.sin(yaw_rad / 2)
    return ObjectPlacement(
        name=name, type="box",
        dims_m=(hx * 2, hy * 2, hz * 2),
        pose_xyz_m=(float(centre_base[0]), float(centre_base[1]), float(centre_base[2])),
        pose_quat_xyzw=(0.0, 0.0, float(qz), float(qw)),
    )
