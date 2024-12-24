from os import listdir

from ab.nn.util.Util import *


def extract_all_configs(config) -> list[str]:
    """
    Collect models matching the given configuration prefix
    """
    l = [c for c in listdir(Const.stat_dir_global) if c.startswith(config)]
    if not l and is_full_config(config):
        makedirs(join(Const.stat_dir_global, config))
        l = [config]
    return l


# todo: Request information from database
def provide_all_configs(config) -> tuple[str]:
    if not isinstance(config, tuple):
        config = (config,)
    all_configs = []
    for c in config:
        all_configs = all_configs + extract_all_configs(c)
    all_configs: list = list(set(all_configs))
    all_configs.sort()
    return tuple(all_configs)
