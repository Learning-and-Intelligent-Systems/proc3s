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
    obs = env.reset()

    belief = updater.update(obs)

    twin_env: Environment = hydra.utils.get_class(cfg.env._target_).sample_twin(
        env, belief, task, render=cfg.vis_debug
    )
    policy: Policy = hydra.utils.instantiate(
        cfg.policy, twin=twin_env, seed=cfg["seed"]
    )

    statistics = {"execution_time": 0, "planning_time": 0}
    for i in range(cfg.get("max_env_steps")):
        log.info("Step " + str(i))
        goal = env.task.get_goal()
        log.info("Goal: " + str(goal))
        belief = updater.update(obs)
        log.info("Scene: " + str(belief))
        st = time.time()
        action, step_statistics = policy.get_action(belief, goal)
        for k, v in step_statistics.items():
            statistics["step_{}_{}".format(i, k)] = v
        statistics["planning_time"] += time.time() - st
        log.info("Action: " + str(action))
        if action is None:
            break

        st = time.time()
        obs, reward, done, info = env.step(action)
        for k, v in info.items():
            statistics["step_{}_{}".format(i, k)] = v
        statistics["execution_time"] += time.time() - st

        if cfg.render:
            env.render()

        log.info("Reward: " + str(reward))
        log.info("Done: " + str(done))
        log.info("Info: " + str(info))

    twin_env.close()
    env.close()
    log.info("Statistics: " + str(json.dumps(statistics)))


if __name__ == "__main__":
    main()
