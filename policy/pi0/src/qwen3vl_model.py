#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3VL High-Level Planner
Generates subtask instructions from visual observations and task descriptions
"""

import torch
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image


class Qwen3VLPlanner:
    """
    High-level planner using Qwen3VL for task decomposition
    Generates subtask instructions for low-level VLA executor
    """

    def __init__(self, model_path, device_map="auto", dtype="auto"):
        """
        Initialize Qwen3VL model

        Args:
            model_path: Path to Qwen3VL model checkpoint
            device_map: Device mapping strategy
            dtype: Model dtype (auto/bfloat16/float16)
        """
        print(f"Loading Qwen3VL planner from {model_path}...")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.device = self.model.device
        print("Qwen3VL planner loaded successfully!")

        # Planning state
        self.current_task = None
        self.subtask_queue = []
        self.current_subtask_idx = 0

    def set_task(self, task_instruction):
        """Set the main task instruction"""
        self.current_task = task_instruction
        self.subtask_queue = []
        self.current_subtask_idx = 0
        print(f"Set main task: {task_instruction}")

    def generate_subtasks(self, images, state=None, max_new_tokens=256):
        """
        Generate subtask plan from current observation

        Args:
            images: List of RGB images [front, right, left] or dict
            state: Optional robot state
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of subtask instructions
        """
        if self.current_task is None:
            raise ValueError("Must set main task first using set_task()")

        # Convert images to PIL format
        pil_images = self._prepare_images(images)

        # Construct planning prompt
        prompt = self._construct_planning_prompt(self.current_task, state)

        # Build messages for Qwen3VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[0]},  # Front camera
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Generate subtask plan
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        # Parse subtasks from output
        subtasks = self._parse_subtasks(output_text)
        self.subtask_queue = subtasks
        self.current_subtask_idx = 0

        print(f"Generated {len(subtasks)} subtasks:")
        for i, subtask in enumerate(subtasks):
            print(f"  {i+1}. {subtask}")

        return subtasks

    def get_current_subtask(self):
        """Get current subtask instruction for low-level executor"""
        if not self.subtask_queue:
            return self.current_task  # Fallback to main task

        if self.current_subtask_idx >= len(self.subtask_queue):
            return None  # All subtasks completed

        return self.subtask_queue[self.current_subtask_idx]

    def advance_subtask(self):
        """Move to next subtask"""
        self.current_subtask_idx += 1
        if self.current_subtask_idx < len(self.subtask_queue):
            print(f"Advanced to subtask {self.current_subtask_idx + 1}/{len(self.subtask_queue)}: "
                  f"{self.subtask_queue[self.current_subtask_idx]}")
            return True
        else:
            print("All subtasks completed!")
            return False

    def _prepare_images(self, images):
        """Convert numpy arrays or dict to PIL Images"""
        pil_images = []

        if isinstance(images, dict):
            # Extract images from dict
            img_list = [
                images.get("cam_high", images.get("front")),
                images.get("cam_right_wrist", images.get("right")),
                images.get("cam_left_wrist", images.get("left"))
            ]
        else:
            img_list = images

        for img in img_list:
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                # Convert from (C, H, W) or (H, W, C) to PIL
                if img.shape[0] == 3:  # (C, H, W)
                    img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(img.astype(np.uint8))
            pil_images.append(img)

        return pil_images

    def _construct_planning_prompt(self, task, state=None):
        """Construct prompt for subtask generation"""
        prompt = f"""You are a robot task planner. Given the main task and current visual observation, decompose it into a sequence of concrete subtasks.

Main Task: {task}

Requirements:
1. Each subtask should be a single, actionable instruction
2. Subtasks should be in sequential order
3. Use simple, clear language
4. Output format: numbered list (1. ... 2. ... 3. ...)

Generate the subtask sequence:"""

        return prompt

    def _parse_subtasks(self, output_text):
        """Parse subtasks from model output"""
        subtasks = []
        lines = output_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Match numbered list: "1.", "1)", etc.
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering
                subtask = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if subtask and not subtask.startswith('-'):
                    subtask = subtask.lstrip('- *').strip()
                if subtask:
                    subtasks.append(subtask)

        # Fallback: if no structured list found, use the whole output
        if not subtasks and output_text.strip():
            subtasks = [output_text.strip()]

        return subtasks

    def reset(self):
        """Reset planning state"""
        self.current_task = None
        self.subtask_queue = []
        self.current_subtask_idx = 0
        print("Planner reset")
