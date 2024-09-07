from __future__ import annotations

import importlib
import logging
import math
import os
import pathlib
import random
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Union
import PIL
from PIL import Image

import numpy as np

from vtamp.environments.utils import Action, Environment, State
from vtamp.policies.utils import (
    ContinuousSampler,
    DiscreteSampler,
    Policy,
    Sampler,
    parse_code,
    query_llm,
)
from vtamp.utils import (
    are_files_identical,
    get_log_dir,
    get_previous_log_folder,
    parse_text_prompt,
    read_file,
    save_log,
    write_prompt,
)

_, _ = Action(), State()
log = logging.getLogger(__name__)


FUNC_NAME = "gen_plan"
FUNC_DOMAIN = "gen_domain"

def get_rendering_from_curr_env(
        env: Environment
) -> PIL.Image.Image:
    """Simply grabs a rendering from the current environment."""
    if "Raven" in env.__str__():
        return Image.fromarray(env.get_camera_image_side(
                    image_size=(460 * 2, 640 * 2)
                )[0])

    else:
        raise NotImplementedError(f"Need to implement rendering function(s) for env {env}!")


def rejection_sample_csp(
    env: Environment,
    initial_state: State,
    plan_gen: Callable[[List[Union[int, float]]], List[Action]],
    domains_gen: List[Sampler],
    max_attempts: int = 10000,
) -> Union[List[Action], str, int, List[PIL.Image.Image]]:
    """A constraint satisfaction strategy that randomly samples input vectors
    until it finds one that satisfies the constraints.

    If none are found, it returns the most common mode of failure.
    """
    violation_modes = Counter()
    img_renderings_from_most_progressed_plan = []
    for i in range(max_attempts):
        curr_attempt_img_renderings = []
        log.info(f"CSP Sampling iter {i}")
        domains = domains_gen(initial_state)
        input_vec = {name: domain.sample() for name, domain in domains.items()}
        _ = env.reset()
        ground_plan = plan_gen(initial_state, **input_vec)
        constraint_violated = False
        curr_attempt_img_renderings.append(get_rendering_from_curr_env(env))
        for ai, action in enumerate(ground_plan):
            _, _, _, info = env.step(action)
            curr_attempt_img_renderings.append(get_rendering_from_curr_env(env))
            if len(info["constraint_violations"]) > 0:
                violation_str = [
                    "Step {}, Action {}, Violation: {}".format(
                        ai, action.name, violation
                    )
                    for violation in info["constraint_violations"]
                ]
                violation_modes.update(violation_str)
                constraint_violated = True
                log.info(f"Constraint violation " + str(info["constraint_violations"]))
                if len(img_renderings_from_most_progressed_plan) < len(curr_attempt_img_renderings):
                    img_renderings_from_most_progressed_plan = curr_attempt_img_renderings[:]
                break
        if not constraint_violated:
            return ground_plan, None, i, curr_attempt_img_renderings        
    return ground_plan[:i+1], violation_modes, i, img_renderings_from_most_progressed_plan


def import_constants_from_class(cls):
    # Get the module name from the class
    module_name = cls.__module__

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Import all uppercase attributes (assuming these are constants)
    for attribute_name in module.__all__:
        # Importing the attribute into the global namespace
        globals()[attribute_name] = getattr(module, attribute_name)
        print(f"Imported {attribute_name}: {globals()[attribute_name]}")


class Ours(Policy):
    def __init__(
        self,
        twin=None,
        max_feedbacks=0,
        max_csp_samples=10000,
        seed=0,
        use_cache=False,
        gaussian_blur=False,
        **kwargs,
    ):
        self.twin = twin
        self.use_cache = use_cache
        self.gaussian_blur = gaussian_blur
        self.seed = seed
        self.max_csp_samples = max_csp_samples
        self.max_feedbacks = max_feedbacks

        # Get environment specific prompt
        import_constants_from_class(twin.__class__)
        prompt_fn = "prompt_{}".format(twin.__class__.__name__)

        prompt_path = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn)
        )
        self.prompt = parse_text_prompt(prompt_path)

        self.plan = None

    def get_action(self, belief, goal: str):
        statistics = {}
        if self.plan is None:
            # No plan yet, we need to come up with one
            ground_plan, statistics = self.full_query_csp(belief, goal)
            if ground_plan is None:
                return None, statistics
            else:
                log.info("Found plan: {}".format(ground_plan))
                self.plan = ground_plan[1:]
                return ground_plan[0], statistics
        elif len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action, statistics
        else:
            return None, statistics

    def full_query_csp(self, belief, task):
        _ = self.twin.reset()
        content = "Goal: {}".format(task)
        content = "State: {}\n".format(str(belief)) + content
        chat_history = self.prompt + [{"role": "user", "content": content}]
        statistics = {}
        statistics["csp_samples"] = 0
        statistics["csp_solve_time"] = 0
        statistics["llm_query_time"] = 0
        for iter in range(self.max_feedbacks + 1):
            statistics["num_feedbacks"] = iter
            st = time.time()
            input_fn = f"llm_input_{iter}.txt"
            output_fn = f"llm_output_{iter}.txt"
            write_prompt(input_fn, chat_history)

            # Check if the inputs match
            parent_log_folder = os.path.join(get_log_dir(), "..")
            previous_folder = get_previous_log_folder(parent_log_folder)
            llm_query_time = 0
            if (
                self.use_cache
                and os.path.isfile(os.path.join(previous_folder, output_fn))
                and are_files_identical(
                    os.path.join(previous_folder, input_fn),
                    os.path.join(get_log_dir(), input_fn),
                )
            ):
                log.info("Loading cached LLM response")
                llm_response = read_file(os.path.join(previous_folder, output_fn))
            else:
                log.info("Querying LLM")
                llm_response, llm_query_time = query_llm(chat_history, seed=self.seed)

            statistics["llm_query_time"] += llm_query_time

            chat_history.append({"role": "assistant", "content": llm_response})
            save_log(output_fn, llm_response)

            error_message = None
            ground_plan = None

            try:
                llm_code = parse_code(llm_response)
                exec(llm_code, globals())
                func = globals()[FUNC_NAME]
                domain = globals()[FUNC_DOMAIN]
                st = time.time()
                ground_plan, failure_message, csp_samples, imgs_list = rejection_sample_csp(
                    self.twin,
                    belief,
                    func,
                    domain,
                    max_attempts=self.max_csp_samples,
                )
                statistics["csp_samples"] += csp_samples
                statistics["csp_solve_time"] += time.time() - st

            except Exception as e:
                # Get the traceback as a string
                error_message = traceback.format_exc()
                log.info("Code error: " + str(error_message))

            # HACK! For now enable manual oracle input of a failure reason!
            error_message = ""
            for img_i, img in enumerate(imgs_list):
                img.save(f"test_{img_i}.png")
            failure_string = input("human-provided reason for failure")
            if len(failure_string) > 0:
                error_message += failure_string + "\n"
            # END HACK

            if ground_plan is not None and len(error_message) == 0:
                return ground_plan, statistics
            else:
                if error_message is not None:
                    failure_response = error_message
                else:
                    failure_response = ""
                    for fm, count in failure_message.most_common(2):
                        failure_response += f"{count} occurences: {fm}\n"

                save_log(f"feedback_output_{iter}.txt", failure_response)
                chat_history.append({"role": "user", "content": failure_response})

        import ipdb; ipdb.set_trace()

        return ground_plan, statistics
