import json
import logging
import os
import pathlib
import random
import sys
import time

import hydra
import numpy as np
from omegaconf import OmegaConf

from vtamp.environments.utils import Environment, Task, Updater
from vtamp.policies.utils import Policy
from vtamp.utils import get_log_dir

import pickle as pkl
log = logging.getLogger(__name__)


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logger():
    log_level = logging.DEBUG

    # Get the Hydra log directory
    log_dir = get_log_dir()
    log_file = os.path.join(log_dir, f"output.log")

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Add FileHandler to logger to output logs to a file
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Add StreamHandler to logger to output logs to stdout
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logger, log_level)
    sys.stderr = StreamToLogger(logger, logging.ERROR)



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("vtamp", "config")),
)
def main(cfg: OmegaConf):

    log.info(" ".join(sys.argv))

    setup_logger()

    if cfg.get("seed") is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    log.info("Setting up environment and policy...")
    task: Task = hydra.utils.instantiate(cfg.task)
    updater: Updater = hydra.utils.instantiate(cfg.updater)
    env: Environment = hydra.utils.instantiate(
        cfg.env, task=task, render=cfg.render and not cfg.vis_debug
    )
    with open("last_ground_plan.plan", "rb") as f:
        plan = pkl.load(f)
    obs = env.reset()
    for action in plan:
        obs, reward, done, info = env.step(action)
        if cfg.render:
            env.render()

    env.close()

if __name__ == "__main__":
    main()