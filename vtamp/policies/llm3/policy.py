from __future__ import annotations

import importlib
import logging
import os
import pathlib
import time
import traceback

from vtamp.environments.utils import Action, State
from vtamp.policies.utils import (
    Policy,
    guassian_rejection_sample,
    parse_code,
    query_llm,
)
from vtamp.utils import parse_text_prompt, save_log, write_prompt

_, _ = Action(), State()
log = logging.getLogger(__name__)

FUNC_NAME = "gen_plan"
FUNC_DOMAIN = "gen_domain"
ENGINE = "gpt-4-0125-preview"  # "gpt-4-turbo-2024-04-09" #"gpt-4-0125-preview"  #'gpt-3.5-turbo-instruct'


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


class LLM3(Policy):
    def __init__(
        self,
        twin=None,
        max_feedbacks=0,
        gaussian_blur=False,
        seed=0,
        max_csp_samples=0,
        **kwargs,
    ):
        self.twin = twin
        self.max_csp_samples = max_csp_samples
        self.gaussian_blur = gaussian_blur
        self.seed = seed
        self.max_feedbacks = max_feedbacks

        import_constants_from_class(twin.__class__)

        # Get environment specific prompt
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
            content = "Goal: {}".format(goal)
            content = "State: {}\n".format(str(belief)) + content

            chat_history = self.prompt + [{"role": "user", "content": content}]
            statistics["csp_samples"] = 0
            statistics["csp_solve_time"] = 0
            statistics["llm_query_time"] = 0
            for iter in range(self.max_feedbacks + 1):
                statistics["num_feedbacks"] = iter
                llm_response, llm_query_time = query_llm(chat_history, seed=0)
                statistics["llm_query_time"] += llm_query_time
                write_prompt(f"llm_input_{iter}.txt", chat_history)
                chat_history.append({"role": "assistant", "content": llm_response})
                save_log(f"llm_output_{iter}.txt", llm_response)
                error_message = None
                try:
                    llm_code = parse_code(llm_response)
                    exec(llm_code, globals())
                    ground_plan = globals()["gen_plan"]

                    violation_str = ""
                    _ = self.twin.reset()
                    for ai, action in enumerate(ground_plan):
                        _, _, _, info = self.twin.step(action)
                        if len(info["constraint_violations"]) > 0:
                            violation_str += (
                                "Step {}, Action {}, Violations: {}\n".format(
                                    ai, action.name, info["constraint_violations"]
                                )
                            )
                            break
                except Exception as e:
                    # Get the traceback as a string
                    error_message = traceback.format_exc()
                    log.info("Code error: " + str(error_message))

                if len(violation_str) > 0 and self.max_feedbacks > iter:
                    if error_message is not None:
                        # There was an error in code execution
                        failure_response = error_message
                    else:
                        if self.gaussian_blur:
                            st = time.time()
                            blurred_plan, csp_samples = guassian_rejection_sample(
                                self.twin,
                                ground_plan,
                                max_attempts=self.max_csp_samples,
                            )
                            statistics["csp_samples"] += csp_samples
                            statistics["csp_solve_time"] += time.time() - st
                            if blurred_plan is not None:
                                self.plan = blurred_plan[1:]
                                return blurred_plan[0], statistics
                        failure_response = "Failure: {}".format(str(violation_str))
                    save_log(f"feedback_output_{iter}.txt", failure_response)
                    chat_history.append({"role": "user", "content": failure_response})
                else:
                    if (
                        ground_plan is not None
                        and isinstance(ground_plan, list)
                        and len(ground_plan) > 0
                    ):
                        self.plan = ground_plan[1:]
                        return ground_plan[0], statistics
        elif len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action, statistics

        return None, statistics
