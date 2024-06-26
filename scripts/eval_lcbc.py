import datetime
import os
import time
from functools import partial

import cv2
import imageio
import jax
import numpy as np
import orbax.checkpoint
from absl import app, flags
from jaxrl_m.agents import agents
from jaxrl_m.data.text_processing import text_processors
from jaxrl_m.vision import encoders
from pyquaternion import Quaternion

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

import wandb

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2
ENV_PARAMS = {
    "camera_topics": [{"name": "/front/image_raw", "flip": False}],
    "return_full_image": False,
    # toysink2
    "override_workspace_boundaries": [
        [0.21, -0.13, 0.06, -1.57, 0],
        [0.37, 0.25, 0.18, 1.57, 0],
    ],
}

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
trajectory_path = "gs://sacson_data/sacson_new/Dec-06-2022-bww8_00000002_0"
traj_init_idx = 0

##############################################################################

np.set_printoptions(suppress=True)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "policy_checkpoint", None, "Path to policy checkpoint", required=True
)
flags.DEFINE_string(
    "policy_wandb", None, "Policy checkpoint wandb run name", required=True
)

flags.DEFINE_integer("im_size", None, "Image size", required=True)

flags.DEFINE_string("video_save_path", None, "Path to save video")

flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", True, "Use the blocking controller")

flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")

flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

def load_policy_checkpoint(path, wandb_run_name):
    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)
    config = run.config

    # create encoder from wandb config
    encoder_def = encoders[config["encoder"]](**config["encoder_kwargs"])

    example_actions = np.zeros((1, 7), dtype=np.float32)

    example_obs = {
        "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    }

    example_batch = {
        "observations": example_obs,
        "goals": {
            "language": np.zeros(
                (
                    1,
                    512,
                ),
                dtype=np.float32,
            ),
        },
        "actions": example_actions,
    }

    # create agent from wandb config
    agent = jax.eval_shape(
        partial(
            agents[config["agent"]].create,
            rng=jax.random.PRNGKey(0),
            encoder_def=encoder_def,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            **config["agent_kwargs"],
        ),
    )

    # load action metadata from wandb
    action_metadata = config["bridgedata_config"]["action_proprio_metadata"]["action"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    # load text processor
    text_processor = text_processors[config["text_processor"]](
        **config["text_processor_kwargs"]
    )

    # hydrate agent with parameters from checkpoint
    agent = orbax.checkpoint.PyTreeCheckpointer().restore(
        path,
        item=agent,
    )

    rng = jax.random.PRNGKey(0)

    def get_action(obs, goal_obs):
        nonlocal rng
        if "128" in path:
            obs["image"] = cv2.resize(obs["image"], (128, 128))
            goal_obs["image"] = cv2.resize(goal_obs["image"], (128, 128))
        rng, key = jax.random.split(rng)
        action = jax.device_get(
            agent.sample_actions(obs, goal_obs, seed=key, argmax=True)
        )
        action = action * action_std + action_mean
        action += np.random.normal(0, FIXED_STD)
        return action

    return get_action, text_processor


def rollout_subgoal(get_action, prompt_embed, num_timesteps):
    goal_obs = {
        "language": prompt_embed,
    }

    is_gripper_closed = False
    num_consecutive_gripper_change_actions = 0

    last_tstep = time.time()
    images = []
    full_images = []
    t = 0
    try:
        while t < num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                obs = iio.imread(os.path.join(trajectory_path, f"{traj_init_idx}.jpg"))
                traj_init_idx += 1
                if obs is None:
                    print("WARNING retrying to get observation...")
                    continue

                image_obs = (
                    obs["image"]
                    .reshape(3, FLAGS.im_size, FLAGS.im_size)
                    .transpose(1, 2, 0)
                    * 255
                ).astype(np.uint8)
                images.append(image_obs)
                obs = {"image": image_obs, "proprio": obs["state"]}

                last_tstep = time.time()

                action = get_action(obs, goal_obs)[0]

    except KeyboardInterrupt:
        return images, full_images, True
    return images, full_images, False


def main(_):
    get_action, text_processor = load_policy_checkpoint(
        FLAGS.policy_checkpoint, FLAGS.policy_wandb
    )

    # goal sampling loop
    prompt_embed = None
    done = False
    while True:
        # ask for new goal
        if prompt_embed is None:
            ch = "y"
        else:
            ch = input("New instruction? [y/n]")
        if ch == "y":
            prompt = input("Enter Prompt: ")
            prompt_embed = text_processor.encode(prompt)

        input("Press [Enter] to start.")

        # do rollout
        images, full_images, done = rollout_subgoal(
            get_action, prompt_embed, FLAGS.num_timesteps
        )

        if FLAGS.video_save_path is not None:
            save_path = os.path.join(
                FLAGS.video_save_path,
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp4"),
            )
            video = np.array(images)
            imageio.mimsave(
                save_path,
                video,
                fps=3.0 / STEP_DURATION,
            )
            with open(save_path.replace(".mp4", "_prompt.txt"), "w") as f:
                f.write(prompt)


if __name__ == "__main__":
    app.run(main)
