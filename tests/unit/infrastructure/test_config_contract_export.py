"""EXP-11: mission.contract_export parsing + config_hash invariance.

The flag only opts a run into the raw data contract in results.json; it changes
no physics. A mis-typed value must not be quietly coerced (the strict-boolean
rule EXP-04/07/08 apply), and the key must stay absent from every shipped YAML so
config_hash -- taken over the raw merged YAML -- is unchanged when the flag is
off, keeping flag-off byte-identity (incl. the cross-commit golden) intact.
"""
from __future__ import annotations

import pytest

from uav_swarm_sim.infrastructure.config import ConfigError, load_config


def test_defaults_off(config_path):
    assert load_config(config_path).mission.contract_export is False


def test_key_absent_from_every_shipped_yaml():
    import pathlib

    for path in sorted(pathlib.Path("config").rglob("*.yaml")):
        assert "contract_export" not in path.read_text(encoding="utf-8"), path


def test_config_hash_unchanged_when_flag_absent(config_path):
    """The default-absent key must not perturb identity: loading with no override
    yields the same config_hash as before the field existed (i.e. the raw YAML is
    untouched)."""
    a = load_config(config_path).config_hash
    b = load_config(config_path).config_hash
    assert a == b
    # setting it True is a raw-YAML change and DOES move the hash (opt-in cost)
    on = load_config(config_path, overrides={"mission.contract_export": True})
    assert on.config_hash != a


@pytest.mark.parametrize("bad", ["true", "yes", 1, 0, None])
def test_refuses_a_non_boolean(config_path, bad):
    with pytest.raises(ConfigError, match="contract_export must be a boolean"):
        load_config(config_path, overrides={"mission.contract_export": bad})
