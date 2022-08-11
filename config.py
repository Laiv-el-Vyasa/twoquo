"""General configuration handling.

Includes functions to load and parse a JSON configuration and to load solvers
using a plugin system.
"""
import copy
import json
import importlib
from numpy import set_printoptions

from paths import PATH_CFG


def _update_cfg(base_cfg, new_cfg):
    # We support one level for now.
    for k in new_cfg.keys():
        if k == 'base_cfg' or k == 'desc' or k == 'dataset_id':
            continue
        base_cfg[k].update(new_cfg[k])
    base_cfg['dataset_id'] = new_cfg['dataset_id']


def load_cfg(fname=PATH_CFG, cfg_id="1"):
    """Load the JSON configuration found in `tooquo/config.json`.

    Args:
        fname: Path to the configuration. Optional, default is config.json in
            the current directory.
        cfg_id: Configuration identifier of the desired pipeline.
            Optional, default is "1". This allows configuring and loading
            separate pipelines and models.

    Returns:
        A dictionary resulting from parsing the JSON.
    """
    with open(fname, 'r') as f:
        full_cfg = json.load(f)

    if cfg_id not in full_cfg["pipeline"]:
        raise Exception(
            'Error: Pipeline key %s does not exist in config.' % cfg_id
        )

    # set numpy print options if they've been set in the config
    if full_cfg['set_numpy_printoptions']:
        set_numpy_printoptions()

    cfg = full_cfg["pipeline"]

    initial_base_cfg = cfg.get("1", {})
    base_cfg = cfg[cfg[cfg_id].get('base_cfg', cfg_id)]
    # All base configs are based on config "1".
    # This enables backwards compatibility when new options are added.
    _update_cfg(initial_base_cfg, base_cfg)

    new_cfg = copy.deepcopy(initial_base_cfg)
    _update_cfg(new_cfg, cfg[cfg_id])

    full_cfg["pipeline"] = new_cfg
    full_cfg["pipeline"]["cfg_id"] = cfg_id

    # print(json.dumps(full_cfg, sort_keys=True, indent=4))
    return full_cfg


def print_pipeline_cfg_ids():
    with open(PATH_CFG, 'r') as f:
        full_cfg = json.load(f)

    for cfg_id in full_cfg["pipeline"]:
        print(cfg_id)


def set_numpy_printoptions() -> None:
    """Sets global numpy print options so our matrices look nice"""

    set_printoptions(
        linewidth=100,  # ensures the matrix doesn't wrap, 100 is arbitrary
        suppress=True,  # removes scientific notation
        precision=4  # sets the precision
    )


if __name__ == "__main__":
    print_pipeline_cfg_ids()
    cfg = load_cfg()
    print(len(cfg["solvers"]))
