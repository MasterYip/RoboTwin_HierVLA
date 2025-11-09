import numpy as np
import torch
import dill
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from pi_model import *


# Encode observation for the model
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def get_model(usr_args):
    """
    Get model based on configuration

    Args:
        usr_args: Dictionary with keys:
            - train_config_name: Training config name
            - model_name: Model name  
            - checkpoint_id: Checkpoint ID
            - pi0_step: Number of PI0 steps
            - hierarchical (optional): If True, use hierarchical Qwen+PI0 policy
            - qwen_model_path (optional): Path to Qwen3VL model
            - replan_frequency (optional): Replanning frequency for hierarchical policy

    Returns:
        Policy model (PI0 or HierarchicalQwenPI0)
    """
    train_config_name, model_name, checkpoint_id, pi0_step = (
        usr_args["train_config_name"],
        usr_args["model_name"],
        usr_args["checkpoint_id"],
        usr_args["pi0_step"]
    )

    # Check if hierarchical policy is requested
    if usr_args.get("hierarchical", False):
        print("Loading Hierarchical Qwen3VL + PI0 Policy...")
        from hier_qwen_pi import HierarchicalQwenPI0

        return HierarchicalQwenPI0(
            train_config_name=train_config_name,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            pi0_step=pi0_step,
            qwen_model_path=usr_args.get("qwen_model_path",
                                         "/inspire/ssd/project/25jinqiu07/public/hiervla_003/Qwen3-VL-8B-Instruct"),
            replan_frequency=usr_args.get("replan_frequency", 10)
        )
    else:
        print("Loading Standard PI0 Policy...")
        return PI0(train_config_name, model_name, checkpoint_id, pi0_step)


def eval(TASK_ENV, model, observation):

    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # ======== Get Action ========

    actions = model.get_action()[:model.pi0_step]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    # ============================


def reset_model(model):
    model.reset_obsrvationwindows()
