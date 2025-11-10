#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


# ### Qwen作为高层规划器 ###
# # 导入Qwen3VLForConditionalGeneration模型类和AutoProcessor处理器类
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# def VLM_taskplanning(input_text):
#     # 默认方式加载Qwen3-VL-8B-Instruct模型
#     # dtype="auto"会自动选择合适的数据类型
#     # device_map="auto"会自动将模型分配到可用设备(CPU/GPU)
#     model = Qwen3VLForConditionalGeneration.from_pretrained(
#         "/inspire/ssd/project/25jinqiu07/liyiheng-P-253130468/RoboTwin_HierVLA/policy/pi0/Qwen/Qwen3-VL-8B-Instruct",
#         dtype="auto", device_map="auto",
#         local_files_only=True
#     )
#     # 加载与模型匹配的处理器
#     processor = AutoProcessor.from_pretrained("/inspire/ssd/project/25jinqiu07/liyiheng-P-253130468/RoboTwin_HierVLA/policy/pi0/Qwen/Qwen3-VL-8B-Instruct", local_files_only=True)

#     # 构建对话消息列表
#     messages = [
#         {
#             "role": "user",  # 用户角色
#             "content": [  # 多模态内容
#                 # {  # 图像部分
#                 #     "type": "image",
#                 #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#                 # },
#                 {  # 文本部分
#                     "type": "text", 
#                     "text": f"""
#                     You are an advanced task planner responsible for decomposing complex RoboTwin tasks into executable atomic subtask sequences. Each subtask must be a single, indivisible action and maintain the natural chronological order of actions. The final output should only include the obtained subtask sequence, without any redundant information.

#                     Example:
#                     【Input】
#                     Take the apple from the fridge and place it on the desk
#                     【Output】
#                     1. Move to the kitchen fridge
#                     2. Open the fridge door
#                     3. Take the apple from inside the fridge
#                     4. Close the fridge door
#                     5. Move to the study desk
#                     6. Place the apple on the desk

#                     Below is the input you need to process this time:
#                     {input_text}
#                     """
#                 },
#             ],
#         }
#     ]

#     # 准备推理输入
#     inputs = processor.apply_chat_template(
#         messages,
#         tokenize=True,  # 进行tokenization
#         add_generation_prompt=True,  # 添加生成提示
#         return_dict=True,  # 返回字典形式
#         return_tensors="pt",  # 返回PyTorch张量
#         max_length=2048
#     )

#     # 将输入数据移动到模型所在的设备
#     inputs = inputs.to(model.device)

#     # 执行推理: 生成输出
#     generated_ids = model.generate(
#         **inputs,  # 传入预处理后的输入
#         max_new_tokens=128  # 限制生成的最大token数
#     )

#     # 修剪生成的ID，去除输入部分只保留生成的输出部分
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] 
#         for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]

#     # 将生成的token ID解码为文本
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, 
#         skip_special_tokens=True,  # 跳过特殊token
#         clean_up_tokenization_spaces=False  # 保留原始空格
#     )

#     # 返回生成的文本描述
#     print(output_text[0])
#     with open("output.txt", "a") as f: f.write(output_text[0] + "\n")
#     return output_text[0]


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=model_name)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        # self.instruction = VLM_taskplanning(instruction)
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
