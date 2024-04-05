import datetime
import os
from collections import deque

from susie.model import create_sample_fn

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import time
from typing import Callable, List, Tuple

import imageio
import jax
import numpy as np
from absl import app, flags
from pyquaternion import Quaternion

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs

from susie.jax_utils import (
    deserialize_jax_fn,
    initialize_compilation_cache,
)

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 2
ENV_PARAMS = {
    "camera_topics": [{"name": "/blue/image_raw", "flip": False}],
    # forward, left, up
    # wallpaper
    # "override_workspace_boundaries": [
    #     [0.1, -0.15, 0.0, -1.57, 0],
    #     [0.60, 0.25, 0.18, 1.57, 0],
    # ],
    # toysink2
    # "override_workspace_boundaries": [
    #     [0.21, -0.13, 0.06, -1.57, 0],
    #     [0.36, 0.25, 0.18, 1.57, 0],
    # ],
    # microwave
    "override_workspace_boundaries": [
        [0.1, -0.15, 0.05, -1.57, 0],
        [0.35, 0.25, 0.23, 1.57, 0],
    ],
    "move_duration": STEP_DURATION,
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
    "diffusion_checkpoint", None, "Path to diffusion checkpoint", required=True
)
flags.DEFINE_string(
    "diffusion_wandb",
    None,
    "Name of wandb run to get diffusion config from.",
    required=True,
)
flags.DEFINE_integer("diffusion_num_steps", 50, "Number of diffusion steps")
flags.DEFINE_string(
    "diffusion_pretrained_path",
    None,
    "Path to pretrained model to get text encoder + VAE from.",
    required=True,
)

flags.DEFINE_float("prompt_w", 1.0, "CFG weight to use for diffusion sampler")
flags.DEFINE_float("context_w", 1.0, "CFG weight to use for diffusion sampler")

flags.DEFINE_string("video_save_path", None, "Path to save video")

flags.DEFINE_integer("num_timesteps", 40, "num timesteps")
flags.DEFINE_bool("blocking", True, "Use the blocking controller")

flags.DEFINE_spaceseplist("initial_eep", None, "Initial position", required=True)

flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

'''
def rollout_subgoal(
    get_action: Callable[[np.ndarray, np.ndarray], np.ndarray],
    goal_obs: np.ndarray,
    num_timesteps: int,
    obs_horizon: int,
    is_gripper_closed: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:
    num_consecutive_gripper_change_actions = 0

    last_tstep = time.time()
    images = []
    full_images = []
    t = 0
    actions = None
    rng = jax.random.PRNGKey(int(time.time()))
    if obs_horizon is not None:
        obs_hist = deque(maxlen=obs_horizon)
    try:
        while t < num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                obs = widowx_client.get_observation()
                if obs is None:
                    print("WARNING retrying to get observation...")
                    continue

                obs = (
                    obs["image"]
                    .reshape(3, goal_obs.shape[0], goal_obs.shape[1])
                    .transpose(1, 2, 0)
                    * 255
                ).astype(np.uint8)
                images.append(obs)

                # deal with obs history
                if obs_horizon is not None:
                    if len(obs_hist) == 0:
                        obs_hist.extend([obs] * obs_horizon)
                    else:
                        obs_hist.append(obs)
                    obs = np.stack(obs_hist)

                last_tstep = time.time()

                # deal with mutli-action prediction
                rng, key = jax.random.split(rng)
                pred_actions = jax.device_get(get_action(key, obs, goal_obs))
                if len(pred_actions.shape) == 1:
                    pred_actions = pred_actions[None]
                if actions is None:
                    actions = np.zeros_like(pred_actions)
                    weights = 1 / (np.arange(len(pred_actions)) + 1)
                else:
                    actions = np.concatenate([actions[1:], np.zeros_like(actions[-1:])])
                    weights = np.concatenate([weights[1:], [1 / len(weights)]])
                actions += pred_actions * weights[:, None]

                action = actions[0]

                # sticky gripper logic
                if (action[-1] < 0.5) != is_gripper_closed:
                    num_consecutive_gripper_change_actions += 1
                else:
                    num_consecutive_gripper_change_actions = 0

                if num_consecutive_gripper_change_actions >= STICKY_GRIPPER_NUM_STEPS:
                    is_gripper_closed = not is_gripper_closed
                    num_consecutive_gripper_change_actions = 0

                action[-1] = 0.0 if is_gripper_closed else 1.0

                # remove degrees of freedom
                if NO_PITCH_ROLL:
                    action[3] = 0
                    action[4] = 0
                if NO_YAW:
                    action[5] = 0

                action_norm = np.linalg.norm(action[:3])

                print(
                    f"Timestep {t}, action norm: {action_norm * 100:.1f}cm, gripper state: {action[-1]}"
                )
                widowx_client.step_action(action, blocking=FLAGS.blocking)

                t += 1
    except KeyboardInterrupt:
        return images, full_images, is_gripper_closed, True
    return images, full_images, is_gripper_closed, False

'''
def main(_):
    initialize_compilation_cache()
    get_action = deserialize_jax_fn(FLAGS.policy_checkpoint)

    obs_horizon = get_action.args_info[0][1].aval.shape[0]
    im_size = get_action.args_info[0][1].aval.shape[1]

    diffusion_sample = create_sample_fn(
        FLAGS.diffusion_checkpoint,
        FLAGS.diffusion_wandb,
        FLAGS.diffusion_num_steps,
        FLAGS.prompt_w,
        FLAGS.context_w,
        0.0,
        FLAGS.diffusion_pretrained_path,
    )

    print(f"obs horizon: {obs_horizon}, im size: {im_size}")

    # goal sampling loop
    prompt = None
    is_gripper_closed = False  # track gripper state between subgoals
    while True:
        # ask for new goal
        if prompt is None or input("New prompt? [y/n]") == "y":
            prompt = input("Enter prompt: ")

        input("Press [Enter] to start.")

        obs = imageio.imread(os.path.join(trajectory_path, f"{traj_init_idx}.jpg"))
        traj_init_idx += 1
        image_obs = (
            obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
        ).astype(np.uint8)

        images = []
        goals = []
        full_images = []
        done = False
        n = 0
        while not done:
            # sample goal
            print(f"Sampling goal {n}...")
            imageio.imwrite("start.png", image_obs)
            print(
                image_obs.shape, image_obs.dtype, np.max(image_obs), np.min(image_obs)
            )
            print(f"'{prompt}'")
            image_goal = diffusion_sample(image_obs, prompt)
            imageio.imwrite("goal.png", image_goal)
            '''
            # do rollout
            (
                rollout_images,
                rollout_full_images,
                is_gripper_closed,
                done,
            ) = rollout_subgoal(
                widowx_client,
                get_action,
                image_goal,
                FLAGS.num_timesteps,
                obs_horizon,
                is_gripper_closed,
            )
            images.extend(rollout_images)
            full_images.extend(rollout_full_images)
            goals.extend([image_goal] * len(rollout_images))

            image_obs = imageio.imread(os.path.

            n += 1
            '''
            image_obs = imageio.imread(os.path.join(trajectory_path, f"{traj_init_idx}.jpg"))
            traj_init_idx += 1
        if FLAGS.video_save_path is not None:
            save_path = os.path.join(
                FLAGS.video_save_path,
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.mp4"),
            )
            video = np.concatenate([goals, images], axis=1)
            imageio.mimsave(
                save_path,
                video,
                fps=1.0 / STEP_DURATION * 3,
            )
            with open(save_path.replace(".mp4", "_prompt.txt"), "w") as f:
                f.write(prompt)


if __name__ == "__main__":
    app.run(main)
