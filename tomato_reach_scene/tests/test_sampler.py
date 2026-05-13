"""Smoke tests for the sampler + exporter."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from tomato_reach_scene import Scene, SceneConfig, SceneSampler
from tomato_reach_scene.exporters import (
    to_curobo_v2_yaml,
    to_storm_yaml,
    write_batch,
    write_batch_v2,
)
from tomato_reach_scene.sampler import SamplingError


def test_default_config_accepts():
    sampler = SceneSampler(SceneConfig(seed=0))
    scene = sampler.sample()
    assert isinstance(scene, Scene)
    assert "A_rice_noodle" in scene.primitives
    assert "B_tomato" in scene.primitives
    assert "C_aqua_bottle" in scene.primitives
    assert "D_soup_can" in scene.primitives
    assert "E_cocopops" in scene.primitives
    assert "shelf_slab_middle" in scene.primitives


def test_feasibility_disjunction_holds():
    """Every sample's max(dist_E, dist_C) >= r_swing — the algorithmic guarantee."""
    cfg = SceneConfig(seed=0, r_swing_mm=125.0)
    sampler = SceneSampler(cfg)
    for _ in range(200):
        s = sampler.sample()
        assert max(s.params.dist_e_mm, s.params.dist_c_mm) >= cfg.r_swing_mm - 1e-6


def test_clearance_buffer_eliminates_borderline_scenes():
    """With clearance_buffer_mm > 0, no obstacle's centre lands within
    ±buffer of r_swing in *hard-sided* scenes. This is the property that
    prevents the pedagogically-ambiguous "both borderline" case.

    For ``hard_side == "neither"`` (both-easy) scenes both obstacles are
    in the easy band by construction; this test pins down only the
    structured hard-side cases by setting ``both_easy_probability=0``."""
    cfg = SceneConfig(
        seed=0, r_swing_mm=125.0, clearance_buffer_mm=15.0,
        both_easy_probability=0.0,
    )
    sampler = SceneSampler(cfg)
    for _ in range(500):
        s = sampler.sample()
        assert s.params.hard_side in ("left", "right")
        hard = min(s.params.dist_e_mm, s.params.dist_c_mm)
        easy = max(s.params.dist_e_mm, s.params.dist_c_mm)
        # Hard must be at most r_swing - buffer; easy at least r_swing + buffer.
        assert hard <= cfg.r_swing_mm - cfg.clearance_buffer_mm + 1e-6
        assert easy >= cfg.r_swing_mm + cfg.clearance_buffer_mm - 1e-6
        # Minimum contrast between hard and easy is 2 * buffer.
        assert easy - hard >= 2 * cfg.clearance_buffer_mm - 1e-6


def test_both_easy_probability_disables_when_zero():
    """Setting both_easy_probability=0 forces structured hard-side
    sampling only; hard_side ∈ {left, right} for every scene."""
    cfg = SceneConfig(seed=0, both_easy_probability=0.0)
    sampler = SceneSampler(cfg)
    sides = {sampler.sample().params.hard_side for _ in range(200)}
    assert sides == {"left", "right"}, f"unexpected sides: {sides}"


def test_default_both_easy_probability_is_balanced():
    """Default both_easy_probability ≈ 0.33 → all three scene types appear
    in ~equal proportion."""
    cfg = SceneConfig(seed=0)  # uses default both_easy_probability
    sampler = SceneSampler(cfg)
    sides = [sampler.sample().params.hard_side for _ in range(2000)]
    counts = {label: sides.count(label) for label in ("left", "right", "neither")}
    # Every label should appear, and none should be wildly far from 1/3.
    for label, n in counts.items():
        assert n > 0, f"label {label!r} never appeared"
        assert abs(n / 2000 - 1 / 3) < 0.05, (
            f"label {label!r} proportion {n/2000:.3f} too far from 1/3"
        )


def test_both_easy_probability_one_always_neither():
    """At p=1 every scene must be 'both easy', with both obstacles in
    the easy band [r_swing + buffer, d_max]."""
    cfg = SceneConfig(seed=0, both_easy_probability=1.0)
    sampler = SceneSampler(cfg)
    for _ in range(200):
        s = sampler.sample()
        assert s.params.hard_side == "neither"
        assert s.params.dist_e_mm >= cfg.r_swing_mm + cfg.clearance_buffer_mm - 1e-6
        assert s.params.dist_c_mm >= cfg.r_swing_mm + cfg.clearance_buffer_mm - 1e-6


def test_both_easy_probability_mid_value_mixes():
    """At p=0.5 we expect roughly half the scenes to be 'neither' and
    the rest split between 'left' and 'right'. Loose tolerance: just
    check all three labels appear in ≥1000 samples."""
    cfg = SceneConfig(seed=0, both_easy_probability=0.5)
    sampler = SceneSampler(cfg)
    sides = [sampler.sample().params.hard_side for _ in range(1000)]
    seen = set(sides)
    assert seen == {"left", "right", "neither"}, f"missing labels: {seen}"
    # Sanity on rough proportion — within 10 percentage points of expectation.
    n_neither = sides.count("neither")
    assert abs(n_neither / 1000 - 0.5) < 0.10, (
        f"neither ratio {n_neither/1000:.2f} too far from 0.5"
    )


def test_both_easy_probability_validation():
    """Out-of-range probabilities are rejected at config time."""
    with pytest.raises(ValueError, match="both_easy_probability"):
        SceneConfig(both_easy_probability=-0.1)
    with pytest.raises(ValueError, match="both_easy_probability"):
        SceneConfig(both_easy_probability=1.5)


def test_zero_buffer_recovers_legacy_behaviour():
    """clearance_buffer_mm == 0 means the close and far bands touch at r_swing,
    same as the original uniform-bands sampler."""
    cfg = SceneConfig(seed=0, r_swing_mm=125.0, clearance_buffer_mm=0.0)
    sampler = SceneSampler(cfg)
    # Distribution should still produce feasible scenes.
    for _ in range(50):
        s = sampler.sample()
        assert max(s.params.dist_e_mm, s.params.dist_c_mm) >= cfg.r_swing_mm - 1e-6


def test_seed_determinism():
    a = SceneSampler(SceneConfig(seed=42)).sample()
    b = SceneSampler(SceneConfig(seed=42)).sample()
    assert a.params.k_mm == b.params.k_mm
    assert a.params.hard_side == b.params.hard_side
    assert a.params.e_xy_mm == b.params.e_xy_mm
    assert a.params.c_xy_mm == b.params.c_xy_mm
    assert a.params.e_yaw_deg == b.params.e_yaw_deg


def test_k_out_of_range_raises():
    with pytest.raises(ValueError, match="out of geometric bound"):
        SceneConfig(k_mm=300.0)  # > K_MAX_MM ≈ 119


def test_d_min_must_be_less_than_r_swing():
    with pytest.raises(ValueError, match="d_min_mm"):
        SceneConfig(d_min_mm=200.0, r_swing_mm=125.0)


def test_e_yaw_fixed_mode():
    cfg = SceneConfig(seed=1, e_yaw_mode="fixed", e_yaw_deg=42.0)
    sampler = SceneSampler(cfg)
    for _ in range(20):
        s = sampler.sample()
        assert math.isclose(s.params.e_yaw_deg, 42.0)


def test_storm_yaml_roundtrip(tmp_path: Path):
    sampler = SceneSampler(SceneConfig(seed=7))
    scene = sampler.sample()
    out = tmp_path / "world.yaml"
    to_storm_yaml(scene, out)
    data = yaml.safe_load(out.read_text())
    assert "world_model" in data
    coll = data["world_model"]["coll_objs"]
    assert set(coll.keys()) == {"cube", "cylinder", "sphere"}
    # Shelf slabs + columns + E = 8 cubes; cylinders = A, B, C, D = 4.
    assert len(coll["cube"]) == 8
    assert len(coll["cylinder"]) == 4
    assert coll["sphere"] == {}


def test_batch_writer(tmp_path: Path):
    sampler = SceneSampler(SceneConfig(seed=11))
    scenes = sampler.sample_batch(n=5)
    out_dir = write_batch(scenes, tmp_path / "scenes")
    assert out_dir.exists()
    assert (out_dir / "manifest.yaml").exists()
    for i in range(5):
        assert (out_dir / f"variant_{i:02d}.yaml").exists()


def test_curobo_v2_yaml_schema(tmp_path: Path):
    """V2 schema: root-level cuboid:/cylinder:, no world_model wrapper,
    pose quaternion in wxyz order."""
    sampler = SceneSampler(SceneConfig(seed=7))
    scene = sampler.sample()
    out = tmp_path / "world_v2.yaml"
    to_curobo_v2_yaml(scene, out)
    data = yaml.safe_load(out.read_text())

    # 1. No world_model wrapper.
    assert "world_model" not in data
    # 2. Boxes are under `cuboid:`, not `cube:`.
    assert "cuboid" in data
    assert "cube" not in data
    # 3. Cylinders at root too.
    assert "cylinder" in data
    # 4. Counts match: 7 shelf cubes + 1 E box; 4 cylinders.
    assert len(data["cuboid"]) == 8
    assert len(data["cylinder"]) == 4
    # 5. Pose quaternion is wxyz: for the E box, our generator emits
    #    yaw rotation about z only, so qx == qy == 0, and the wxyz form
    #    starts with [pos, qw, 0, 0, qz]. Verify the qx/qy slots are zero.
    e_pose = data["cuboid"]["E_cocopops"]["pose"]
    assert len(e_pose) == 7
    qx_slot, qy_slot = e_pose[4], e_pose[5]
    assert abs(qx_slot) < 1e-9 and abs(qy_slot) < 1e-9, (
        f"expected qx,qy ≈ 0 (yaw-only rotation), got qx={qx_slot}, qy={qy_slot}"
    )


def test_curobo_v2_batch_writer(tmp_path: Path):
    sampler = SceneSampler(SceneConfig(seed=13))
    scenes = sampler.sample_batch(n=3)
    out_dir = write_batch_v2(scenes, tmp_path / "scenes_v2")
    assert (out_dir / "manifest.yaml").exists()
    manifest = yaml.safe_load((out_dir / "manifest.yaml").read_text())
    assert manifest["format"] == "curobo_v2"
    for i in range(3):
        f = out_dir / f"variant_{i:02d}.yaml"
        assert f.exists()
        data = yaml.safe_load(f.read_text())
        # Verify V2 shape on every variant.
        assert "world_model" not in data
        assert "cuboid" in data
        assert "cylinder" in data


def test_grasp_pose_above_b():
    """Grasp position should be above B's geometric centre in z."""
    from tomato_reach_scene.constants import (
        A_HEIGHT_M, B_HEIGHT_M, COMPOSITE_ORIGIN_IN_BASE_M,
    )
    sampler = SceneSampler(SceneConfig(seed=3, grasp_approach_offset_m=0.005))
    scene = sampler.sample()
    expected_z = (
        COMPOSITE_ORIGIN_IN_BASE_M[2]    # slab top in base z
        + A_HEIGHT_M                      # top of A
        + B_HEIGHT_M                      # top of B
        + 0.005                           # approach offset
    )
    assert math.isclose(scene.grasp_pose_xyz_m[2], expected_z, abs_tol=1e-9)


def test_storm_yaml_contains_grasp_target_link_hint():
    """Manifest carries the TCP link name so the collaborator knows which
    frame the grasp pose refers to."""
    import tempfile
    sampler = SceneSampler(SceneConfig(seed=4))
    scenes = sampler.sample_batch(n=3)
    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path
        out_dir = write_batch(scenes, Path(tmp) / "scenes")
        manifest = yaml.safe_load((out_dir / "manifest.yaml").read_text())
    assert manifest["count"] == 3
    for v in manifest["variants"]:
        assert v["grasp_pose_base"]["reference_link"] == "panda_hand_tcp"
        assert len(v["grasp_pose_base"]["translation_m"]) == 3
        assert len(v["grasp_pose_base"]["quaternion_xyzw"]) == 4
