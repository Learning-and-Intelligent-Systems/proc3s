import logging
import os
import pathlib
from heapq import nlargest

import clip
import flax
import hydra
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow.compat.v1 as tf
import torch
from flax import linen as nn
from jax.lib import xla_bridge

from vtamp.environments.utils import Action, Environment
from vtamp.policies.utils import Policy, openai_client

log = logging.getLogger(__name__)
ENGINE = "davinci-002"  # "text-ada-001"

PICK_TARGETS = {
    "blue block": None,
    "red block": None,
    "green block": None,
    "yellow block": None,
}

PLACE_TARGETS = {
    #   "blue block": None,
    #   "red block": None,
    #   "green block": None,
    #   "yellow block": None,
    "blue bowl": None,
    "red bowl": None,
    "green bowl": None,
    "yellow bowl": None,
    # "purple bowl": None,
    # "pink bowl": None,
    # "cyan bowl": None,
    # "brown bowl": None,
    #   "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
    #   "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
    #   "middle":              (0,           -0.5,        0),
    #   "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
    #   "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}

CONTEXT = """
  objects = [red block, yellow block, blue block, green bowl]
  # move all the blocks to the top left corner.
  robot.pick_and_place(blue block, top left corner)
  robot.pick_and_place(red block, top left corner)
  robot.pick_and_place(yellow block, top left corner)
  done()

  objects = [red block, yellow block, blue block, green bowl]
  # put the yellow one the green thing.
  robot.pick_and_place(yellow block, green bowl)
  done()

  objects = [yellow block, blue block, red block]
  # move the light colored block to the middle.
  robot.pick_and_place(yellow block, middle)
  done()

  objects = [blue block, green bowl, red block, yellow bowl, green block]
  # stack the blocks.
  robot.pick_and_place(green block, blue block)
  robot.pick_and_place(red block, green block)
  done()

  objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
  # group the blue objects together.
  robot.pick_and_place(blue block, blue bowl)
  done()

  objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
  # sort all the blocks into their matching color bowls.
  robot.pick_and_place(green block, green bowl)
  robot.pick_and_place(red block, red bowl)
  robot.pick_and_place(yellow block, yellow bowl)
  done()
  """
TERMINATION_STRING = "done()"

# @title LLM Cache
overwrite_cache = True
if overwrite_cache:
    LLM_CACHE = {}


####### Clipport ##########


class ResNetBlock(nn.Module):
    """ResNet pre-Activation block.

    https://arxiv.org/pdf/1603.05027.pdf
    """

    features: int
    stride: int = 1

    def setup(self):
        self.conv0 = nn.Conv(self.features // 4, (1, 1), (self.stride, self.stride))
        self.conv1 = nn.Conv(self.features // 4, (3, 3))
        self.conv2 = nn.Conv(self.features, (1, 1))
        self.conv3 = nn.Conv(self.features, (1, 1), (self.stride, self.stride))

    def __call__(self, x):
        y = self.conv0(nn.relu(x))
        y = self.conv1(nn.relu(y))
        y = self.conv2(nn.relu(y))
        if x.shape != y.shape:
            x = self.conv3(nn.relu(x))
        return x + y


class UpSample(nn.Module):
    """Simple 2D 2x bilinear upsample."""

    def __call__(self, x):
        B, H, W, C = x.shape
        new_shape = (B, H * 2, W * 2, C)
        return jax.image.resize(x, new_shape, "bilinear")


class ResNet(nn.Module):
    """Hourglass 53-layer ResNet with 8-stride."""

    out_dim: int

    def setup(self):
        self.dense0 = nn.Dense(8)

        self.conv0 = nn.Conv(64, (3, 3), (1, 1))
        self.block0 = ResNetBlock(64)
        self.block1 = ResNetBlock(64)
        self.block2 = ResNetBlock(128, stride=2)
        self.block3 = ResNetBlock(128)
        self.block4 = ResNetBlock(256, stride=2)
        self.block5 = ResNetBlock(256)
        self.block6 = ResNetBlock(512, stride=2)
        self.block7 = ResNetBlock(512)

        self.block8 = ResNetBlock(256)
        self.block9 = ResNetBlock(256)
        self.upsample0 = UpSample()
        self.block10 = ResNetBlock(128)
        self.block11 = ResNetBlock(128)
        self.upsample1 = UpSample()
        self.block12 = ResNetBlock(64)
        self.block13 = ResNetBlock(64)
        self.upsample2 = UpSample()
        self.block14 = ResNetBlock(16)
        self.block15 = ResNetBlock(16)
        self.conv1 = nn.Conv(self.out_dim, (3, 3), (1, 1))

    def __call__(self, x, text):
        # # Project and concatenate CLIP features (early fusion).
        # text = self.dense0(text)
        # text = jnp.expand_dims(text, axis=(1, 2))
        # text = jnp.broadcast_to(text, x.shape[:3] + (8,))
        # x = jnp.concatenate((x, text), axis=-1)

        x = self.conv0(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # Concatenate CLIP features (mid-fusion).
        text = jnp.expand_dims(text, axis=(1, 2))
        text = jnp.broadcast_to(text, x.shape)
        x = jnp.concatenate((x, text), axis=-1)

        x = self.block8(x)
        x = self.block9(x)
        x = self.upsample0(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.upsample1(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.upsample2(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.conv1(x)
        return x


class TransporterNets(nn.Module):
    """TransporterNet with 3 ResNets (translation only)."""

    def setup(self):
        # Picking affordances.
        self.pick_net = ResNet(1)

        # Pick-conditioned placing affordances.
        self.q_net = ResNet(3)  # Query (crop around pick location).
        self.k_net = ResNet(3)  # Key (place features).
        self.crop_size = 64
        self.crop_conv = nn.Conv(
            features=1,
            kernel_size=(self.crop_size, self.crop_size),
            use_bias=False,
            dtype=jnp.float32,
            padding="SAME",
        )

    def __call__(self, x, text, p=None, train=True):
        B, H, W, C = x.shape
        pick_out = self.pick_net(x, text)  # (B, H, W, 1)

        # Get key features.
        k = self.k_net(x, text)

        # Add 0-padding before cropping.
        h = self.crop_size // 2
        x_crop = jnp.pad(x, [(0, 0), (h, h), (h, h), (0, 0)], "maximum")

        # Get query features and convolve them over key features.
        place_out = jnp.zeros((0, H, W, 1), jnp.float32)
        for b in range(B):
            # Get coordinates at center of crop.
            if p is None:
                pick_out_b = pick_out[b, ...]  # (H, W, 1)
                pick_out_b = pick_out_b.flatten()  # (H * W,)
                amax_i = jnp.argmax(pick_out_b)
                v, u = jnp.unravel_index(amax_i, (H, W))
            else:
                v, u = p[b, :]

            # Get query crop.
            x_crop_b = jax.lax.dynamic_slice(
                x_crop,
                (b, v, u, 0),
                (1, self.crop_size, self.crop_size, x_crop.shape[3]),
            )
            # x_crop_b = x_crop[b:b+1, v:(v + self.crop_size), u:(u + self.crop_size), ...]

            # Convolve q (query) across k (key).
            q = self.q_net(x_crop_b, text[b : b + 1, :])  # (1, H, W, 3)
            q = jnp.transpose(q, (1, 2, 3, 0))  # (H, W, 3, 1)
            place_out_b = self.crop_conv.apply(
                {"params": {"kernel": q}}, k[b : b + 1, ...]
            )  # (1, H, W, 1)
            scale = 1 / (
                self.crop_size * self.crop_size
            )  # For higher softmax temperatures.
            place_out_b *= scale
            place_out = jnp.concatenate((place_out, place_out_b), axis=0)

        return pick_out, place_out


def n_params(params):
    return jnp.sum(
        jnp.int32(
            [
                (
                    n_params(v)
                    if isinstance(v, dict)
                    or isinstance(v, flax.core.frozen_dict.FrozenDict)
                    else np.prod(v.shape)
                )
                for v in params.values()
            ]
        )
    )


# Train with InfoNCE loss over pick and place positions.
@jax.jit
def train_step(optimizer, batch):
    def loss_fn(params):
        batch_size = batch["img"].shape[0]
        pick_logits, place_logits = TransporterNets().apply(
            {"params": params}, batch["img"], batch["text"], batch["pick_yx"]
        )

        # InfoNCE pick loss.
        pick_logits = pick_logits.reshape(batch_size, -1)
        pick_onehot = batch["pick_onehot"].reshape(batch_size, -1)
        pick_loss = jnp.mean(
            optax.softmax_cross_entropy(logits=pick_logits, labels=pick_onehot), axis=0
        )

        # InfoNCE place loss.
        place_logits = place_logits.reshape(batch_size, -1)
        place_onehot = batch["place_onehot"].reshape(batch_size, -1)
        place_loss = jnp.mean(
            optax.softmax_cross_entropy(logits=place_logits, labels=place_onehot),
            axis=0,
        )

        loss = pick_loss + place_loss
        return loss, (pick_logits, place_logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss, grad, logits


@jax.jit
def eval_step(params, batch):
    pick_logits, place_logits = TransporterNets().apply(
        {"params": params}, batch["img"], batch["text"]
    )
    return pick_logits, place_logits


def run_cliport(params, clip_model, obs, text):
    # Tokenize text and get CLIP features.

    if torch.cuda.is_available():
        text_tokens = clip.tokenize(text).cuda()
    else:
        text_tokens = clip.tokenize(text)

    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens).float()
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    text_feats = np.float32(text_feats.cpu())

    coord_x, coord_y = np.meshgrid(
        np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing="ij"
    )
    coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)

    # Normalize image and add batch dimension.
    img = obs["image"][None, ...] / 255
    img = np.concatenate((img, coords[None, ...]), axis=3)

    # Run Transporter Nets to get pick and place heatmaps.
    batch = {"img": jnp.float32(img), "text": jnp.float32(text_feats)}
    pick_map, place_map = eval_step(params, batch)
    pick_map, place_map = np.float32(pick_map), np.float32(place_map)

    # Get pick position.
    pick_max = np.argmax(np.float32(pick_map)).squeeze()
    pick_yx = (pick_max // 224, pick_max % 224)
    pick_yx = np.clip(pick_yx, 20, 204)
    pick_xyz = obs["xyzmap"][pick_yx[0], pick_yx[1]]

    # Get place position.
    place_max = np.argmax(np.float32(place_map)).squeeze()
    place_yx = (place_max // 224, place_max % 224)
    place_yx = np.clip(place_yx, 20, 204)
    place_xyz = obs["xyzmap"][place_yx[0], place_yx[1]]

    return pick_xyz, place_xyz


################


def gpt3_call(
    prompt="",
    max_tokens=128,
    temperature=0,
    logprobs=1,
    echo=False,
):
    full_query = ""
    for p in prompt:
        full_query += p

    id = tuple((ENGINE, full_query, max_tokens, temperature, logprobs, echo))
    if id in LLM_CACHE.keys():
        log.info("cache hit, returning")
        response = LLM_CACHE[id]
    else:
        response = openai_client.completions.create(
            model=ENGINE,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=logprobs,
            echo=echo,
        )
        LLM_CACHE[id] = response

    return response


def gpt3_scoring(
    query,
    options,
    limit_num_options=None,
    option_start="\n",
    verbose=False,
    print_tokens=False,
):
    if limit_num_options:
        options = options[:limit_num_options]
    verbose and log.info("Scoring {} options".format(len(options)))
    gpt3_prompt_options = [query + option for option in options]
    response = gpt3_call(
        prompt=gpt3_prompt_options,
        max_tokens=0,
        logprobs=1,
        temperature=0,
        echo=True,
    )

    scores = {}
    for option, choice in zip(options, response.choices):
        tokens = choice.logprobs.tokens
        token_logprobs = choice.logprobs.token_logprobs

        total_logprob = 0
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
            print_tokens and log.info(str(token) + str(token_logprob))
            if option_start is None and not token in option:
                break
            if token == option_start:
                break
            total_logprob += token_logprob
        scores[option] = total_logprob

    for i, option in enumerate(sorted(scores.items(), key=lambda x: -x[1])):
        verbose and log.info("{}\t{}".format(option[1], option[0]))
        if i >= 10:
            break

    return scores, response


def make_options(
    pick_targets=None,
    place_targets=None,
    options_in_api_form=True,
    termination_string="done()",
):
    if not pick_targets:
        pick_targets = PICK_TARGETS
    if not place_targets:
        place_targets = PLACE_TARGETS
    options = []
    for pick in pick_targets:
        for place in place_targets:
            if options_in_api_form:
                option = "robot.pick_and_place({}, {})".format(pick, place)
            else:
                option = "Pick the {} and place it on the {}.".format(pick, place)
            options.append(option)

    options.append(termination_string)
    log.info("Considering {} options".format(len(options)))
    return options


# query = "To pick the blue block and put it on the red block, I should:\n"
# options = make_options(PICK_TARGETS, PLACE_TARGETS)
# scores, response = gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)


def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
    scene_description = f"objects = {found_objects}"
    scene_description = scene_description.replace(block_name, "block")
    scene_description = scene_description.replace(bowl_name, "bowl")
    scene_description = scene_description.replace("'", "")
    return scene_description


def step_to_nlp(step):
    step = step.replace("robot.pick_and_place(", "")
    step = step.replace(")", "")
    pick, place = step.split(", ")
    return "Pick the " + pick + " and place it on the " + place + "."


def normalize_scores(scores):
    max_score = max(scores.values())
    normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
    return normed_scores


def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
    if show_top:
        top_options = nlargest(show_top, combined_scores, key=combined_scores.get)
        # add a few top llm options in if not already shown
        top_llm_options = nlargest(show_top // 2, llm_scores, key=llm_scores.get)
        for llm_option in top_llm_options:
            if not llm_option in top_options:
                top_options.append(llm_option)
        llm_scores = {option: llm_scores[option] for option in top_options}
        vfs = {option: vfs[option] for option in top_options}
        combined_scores = {option: combined_scores[option] for option in top_options}

    sorted_keys = dict(sorted(combined_scores.items()))
    keys = [key for key in sorted_keys]
    positions = np.arange(len(combined_scores.items()))
    width = 0.3

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 1, 1)

    plot_llm_scores = normalize_scores(
        {key: np.exp(llm_scores[key]) for key in sorted_keys}
    )
    plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
    plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
    plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])

    ax1.bar(
        positions,
        plot_combined_scores,
        3 * width,
        alpha=0.6,
        color="#93CE8E",
        label="combined",
    )

    score_colors = ["#ea9999ff" for score in plot_affordance_scores]
    ax1.bar(
        positions + width / 2,
        0 * plot_combined_scores,
        width,
        color="#ea9999ff",
        label="vfs",
    )
    ax1.bar(
        positions + width / 2,
        0 * plot_combined_scores,
        width,
        color="#a4c2f4ff",
        label="language",
    )
    ax1.bar(
        positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors
    )

    plt.xticks(rotation="vertical")
    ax1.set_ylim(0.0, 1.0)

    ax1.grid(True, which="both")
    ax1.axis("on")

    ax1_llm = ax1.twinx()
    ax1_llm.bar(
        positions + width / 2,
        plot_llm_scores,
        width,
        color="#a4c2f4ff",
        label="language",
    )
    ax1_llm.set_ylim(0.01, 1.0)
    plt.yscale("log")

    font = {"fontname": "Arial", "size": "16", "color": "k" if correct else "r"}
    plt.title(task, **font)
    key_strings = [
        key.replace("robot.pick_and_place", "")
        .replace(", ", " to ")
        .replace("(", "")
        .replace(")", "")
        for key in keys
    ]
    plt.xticks(positions, key_strings, **font)
    ax1.legend()
    plt.show()


def affordance_scoring(
    options,
    found_objects,
    verbose=False,
    block_name="box",
    bowl_name="circle",
    termination_string="done()",
):
    affordance_scores = {}
    found_objects = [
        found_object.replace(block_name, "block").replace(bowl_name, "bowl")
        for found_object in found_objects
    ]
    for option in options:
        if option == termination_string:
            affordance_scores[option] = 0.2
            continue
        pick, place = (
            option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
        )
        affordance = 0
        found_objects_copy = found_objects.copy()
        if pick in found_objects_copy:
            found_objects_copy.remove(pick)
            if place in found_objects_copy:
                affordance = 1
        affordance_scores[option] = affordance
        verbose and log.info("{} \t {}".format(affordance, option))
    return affordance_scores


class SayCanPolicy(Policy):
    def __init__(self, twin: Environment, debug: bool = False):
        self.debug = debug
        self.steps_text = None
        self.twin = twin
        self.plan = None

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        if torch.cuda.is_available():
            self.clip_model.cuda().eval()
        else:
            self.clip_model.eval()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.session = tf.Session(
            graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)
        )

        saved_model_dir = "./checkpoints/image_path_v2"
        _ = tf.saved_model.loader.load(self.session, ["serve"], saved_model_dir)

        # Load in a transporter net trained specifically for picking and placing
        CKPT_DIR = os.path.join(
            pathlib.Path(__file__).parent.parent.parent.parent, "checkpoints/clipport"
        )
        mgr_options = ocp.CheckpointManagerOptions(
            create=True, max_to_keep=3, keep_period=2, step_prefix="test"
        )

        if xla_bridge.get_backend().platform == "cpu":
            ckpt_mgr = ocp.CheckpointManager(
                CKPT_DIR, ocp.PyTreeCheckpointer(), mgr_options
            )
            step = ckpt_mgr.latest_step()
            sharding = os.path.join(CKPT_DIR, "test_{}/default/_sharding".format(step))

            if os.path.exists(sharding):
                os.remove(sharding)

            structure = ckpt_mgr.item_metadata(step)
            self.params = ckpt_mgr.restore(
                step,
                restore_kwargs={
                    "restore_args": jax.tree_map(
                        lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
                        structure,
                    )
                },
            )
        else:
            ckpt_mgr = ocp.CheckpointManager(
                CKPT_DIR,
                ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                mgr_options,
            )
            self.params = ckpt_mgr.restore(ckpt_mgr.latest_step())

    def get_action(self, belief, goal: str):
        if self.plan is not None and len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action
        else:
            if self.steps_text is not None and len(self.steps_text) == 0:
                return None

            if self.steps_text is None:
                self.steps_text = []
                found_objects = [
                    f"{color} {cat}"
                    for color, cat in zip(belief.colors, belief.categories)
                ]
                scene_description = build_scene_description(found_objects)
                env_description = scene_description

                use_environment_description = False
                gpt3_context_lines = CONTEXT.split("\n")
                gpt3_context_lines_keep = []
                for gpt3_context_line in gpt3_context_lines:
                    if (
                        "objects =" in gpt3_context_line
                        and not use_environment_description
                    ):
                        continue
                    gpt3_context_lines_keep.append(gpt3_context_line)

                gpt3_prompt = CONTEXT
                if use_environment_description:
                    gpt3_prompt += "\n" + env_description
                gpt3_prompt += "\n# " + goal + "\n"

                all_llm_scores = []
                all_affordance_scores = []
                all_combined_scores = []
                options = make_options(
                    PICK_TARGETS, PLACE_TARGETS, termination_string=TERMINATION_STRING
                )

                affordance_scores = affordance_scoring(
                    options,
                    found_objects,
                    block_name="box",
                    bowl_name="circle",
                    verbose=False,
                )
                log.info("Affordance scores: " + str(affordance_scores))
                num_tasks = 0
                selected_task = ""
                self.steps_text = []
                while not selected_task == TERMINATION_STRING:
                    num_tasks += 1
                    max_tasks = 5
                    if num_tasks > max_tasks:
                        break

                    llm_scores, _ = gpt3_scoring(
                        gpt3_prompt,
                        options,
                        verbose=True,
                        print_tokens=False,
                    )
                    combined_scores = {
                        option: np.exp(llm_scores[option]) * affordance_scores[option]
                        for option in options
                    }
                    combined_scores = normalize_scores(combined_scores)
                    selected_task = max(combined_scores, key=combined_scores.get)
                    self.steps_text.append(selected_task)
                    log.info("Step {} Selecting: {}".format(num_tasks, selected_task))
                    gpt3_prompt += selected_task + "\n"

                    all_llm_scores.append(llm_scores)
                    all_affordance_scores.append(affordance_scores)
                    all_combined_scores.append(combined_scores)

            step = self.steps_text[0]
            self.steps_text = self.steps_text[1:]
            print("Step: " + str(step))
            try:
                nlp_step = step_to_nlp(step)
            except:
                return None

            # Aidan: I think saycan would use the first observation image instead of the most recent one?
            pick_xyz, place_xyz = run_cliport(
                self.params, self.clip_model, belief.observations[0], nlp_step
            )
            self.plan = [Action("place", place_xyz)]
            return Action("pick", pick_xyz)
