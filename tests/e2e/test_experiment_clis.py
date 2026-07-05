"""End-to-end experiment CLI tests (full mission through the CLI entry points)."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_single_mission_cli(config_path, tmp_path):
    from uav_swarm_sim.experiments import run_single_mission
    # write a tiny complete config the CLI can load
    import yaml
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["fleet"]["n_drones"] = 3
    raw["fleet"]["battery_capacity_wh"] = 400.0
    raw["failure"]["hazard_rate_per_hour"] = 0.0
    raw["env"]["geojson_path"] = "data/areas/smoke_area.geojson"
    raw["env"]["obstacle_density_per_km2"] = 4.0
    raw["sim"]["dt_s"] = 1.0
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(raw))
    rc = run_single_mission.main([
        "--config", str(cfgp), "--base", str(tmp_path / "runs"),
        "--run-name", "run-test", "--name", "weighted",
    ])
    assert rc == 0
    sim_dir = tmp_path / "runs" / "run-test" / "simulation-weighted"
    # structured run/simulation layout: artifacts + both JSON logs + manifest
    assert (sim_dir / "partition.png").exists()
    assert (sim_dir / "plan.json").exists()
    assert (sim_dir / "results.json").exists()
    assert (tmp_path / "runs" / "run-test" / "run.json").exists()


def test_launch_site_cli(config_path, tmp_path):
    from uav_swarm_sim.experiments import run_launch_site_study
    import yaml
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["fleet"]["n_drones"] = 3
    raw["env"]["geojson_path"] = "data/areas/smoke_area.geojson"
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(raw))
    rc = run_launch_site_study.main(["--config", str(cfgp)])
    assert rc == 0
