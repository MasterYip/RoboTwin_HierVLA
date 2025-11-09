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
        self.initial_subtask_plan = []  # Fixed plan generated at start
        self.subtask_queue = []
        self.current_subtask_idx = 0
        self.completed_subtasks = []  # Track what has been done

    def set_task(self, task_instruction):
        """Set the main task instruction"""
        self.current_task = task_instruction
        self.initial_subtask_plan = []
        self.subtask_queue = []
        self.current_subtask_idx = 0
        self.completed_subtasks = []
        print(f"Set main task: {task_instruction}")

    def generate_initial_plan(self, images, state=None, max_new_tokens=512):
        """
        Generate initial high-level subtask plan (only called once at start)

        Args:
            images: List of RGB images [front, right, left] or dict
            state: Optional robot state
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of high-level subtask descriptions
        """
        if self.current_task is None:
            raise ValueError("Must set main task first using set_task()")

        # Convert images to PIL format
        pil_images = self._prepare_images(images)

        # Construct initial planning prompt
        prompt = self._construct_initial_planning_prompt(self.current_task, state)

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

        # Generate initial plan
        output_text = self._generate_qwen_response(messages, max_new_tokens)

        # Parse subtasks from output
        subtasks = self._parse_subtasks(output_text)
        self.initial_subtask_plan = subtasks
        self.subtask_queue = subtasks.copy()
        self.current_subtask_idx = 0

        print(f"\n{'='*60}")
        print(f"Generated Initial Plan ({len(subtasks)} high-level steps):")
        print(f"{'='*60}")
        for i, subtask in enumerate(subtasks):
            print(f"  Step {i+1}: {subtask}")
        print(f"{'='*60}\n")

        return subtasks

    def generate_motion_command(self, images, state=None, max_new_tokens=256):
        """
        Generate motion-level command based on current observation and progress
        Uses initial plan as context to maintain consistency

        Args:
            images: List of RGB images [front, right, left] or dict
            state: Optional robot state
            max_new_tokens: Maximum tokens to generate

        Returns:
            Motion-level command string for PI0 (describes ~10sec action for both arms)
        """
        if not self.initial_subtask_plan:
            raise ValueError("Must generate initial plan first using generate_initial_plan()")

        # Convert images to PIL format
        pil_images = self._prepare_images(images)

        # Construct progress-aware motion command prompt
        prompt = self._construct_motion_command_prompt(state)

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

        # Generate motion command
        output_text = self._generate_qwen_response(messages, max_new_tokens)

        # Parse motion command (single instruction)
        motion_command = self._parse_motion_command(output_text)

        print(f"\n[Motion Command Generated]")
        print(f"  Current Step: {self.current_subtask_idx + 1}/{len(self.initial_subtask_plan)}")
        print(f"  Command: {motion_command}")

        return motion_command

    def mark_subtask_completed(self):
        """Mark current subtask as completed and advance"""
        if self.current_subtask_idx < len(self.initial_subtask_plan):
            completed = self.initial_subtask_plan[self.current_subtask_idx]
            self.completed_subtasks.append(completed)
            self.current_subtask_idx += 1
            print(f"✓ Completed: {completed}")
            print(f"  Progress: {self.current_subtask_idx}/{len(self.initial_subtask_plan)}")
            return True
        return False

    def get_current_subtask(self):
        """Get current high-level subtask from initial plan"""
        if not self.initial_subtask_plan:
            return self.current_task

        if self.current_subtask_idx >= len(self.initial_subtask_plan):
            return None  # All subtasks completed

        return self.initial_subtask_plan[self.current_subtask_idx]

    def advance_subtask(self):
        """Move to next subtask (deprecated, use mark_subtask_completed)"""
        return self.mark_subtask_completed()

    def _generate_qwen_response(self, messages, max_new_tokens):
        """Common method for Qwen3VL inference"""
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

        return output_text

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

    def _construct_initial_planning_prompt(self, task, state=None):
        """Construct prompt for initial high-level planning"""
        prompt = f"""You are a robot task planner with dual arms. Analyze the scene and decompose the task into high-level sequential steps.

Main Task: {task}

Requirements:
1. Break down into 3-6 high-level steps (not too detailed)
2. Each step should be a meaningful milestone (e.g., "pick up the red block", "move to assembly area")
3. Steps should be in sequential order
4. Consider both arms' coordination when needed
5. Output ONLY a numbered list (1. ... 2. ... 3. ...)

Generate the high-level plan:"""

        return prompt

    def _construct_motion_command_prompt(self, state=None):
        """Construct prompt for generating motion-level command with progress context"""
        # Build context from initial plan
        plan_context = "Initial Plan:\n"
        for i, step in enumerate(self.initial_subtask_plan):
            status = "✓" if i < self.current_subtask_idx else "→" if i == self.current_subtask_idx else "○"
            plan_context += f"  {status} Step {i+1}: {step}\n"

        # Build completed steps context
        completed_context = ""
        if self.completed_subtasks:
            completed_context = "\nCompleted:\n"
            for i, step in enumerate(self.completed_subtasks):
                completed_context += f"  ✓ {step}\n"

        # Current step
        current_step = self.initial_subtask_plan[self.current_subtask_idx] if self.current_subtask_idx < len(self.initial_subtask_plan) else "Final step"

        prompt = f"""You are controlling a dual-arm robot. Based on the initial plan and current observation, generate the NEXT motion-level command.

{plan_context}
{completed_context}
Current Step to Execute: {current_step}

Task: {self.current_task}

Requirements:
1. Generate ONE specific motion command for the current moment (~10 seconds of action)
2. Describe what BOTH arms should do (e.g., "Left arm: pick red block. Right arm: hold position")
3. Be specific about objects and actions (pick/place/move/grasp/release)
4. The command should progress toward completing the current step
5. Output ONLY the motion command, no explanation

Motion Command:"""

        return prompt

    def _parse_subtasks(self, output_text):
        """Parse subtasks from model output"""
        subtasks = []
        lines = output_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            # Match numbered list: "1.", "1)", etc.
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Remove numbering and status symbols
                subtask = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                subtask = subtask.lstrip('- *✓→○').strip()
                if subtask:
                    subtasks.append(subtask)

        # Fallback: if no structured list found, use the whole output
        if not subtasks and output_text.strip():
            subtasks = [output_text.strip()]

        return subtasks

    def _parse_motion_command(self, output_text):
        """Parse motion command from output (extract single command)"""
        # Clean up the output
        command = output_text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Motion Command:",
            "Command:",
            "Next action:",
            "Action:",
        ]
        for prefix in prefixes_to_remove:
            if command.startswith(prefix):
                command = command[len(prefix):].strip()
        
        # Take first line if multi-line
        lines = command.split('\n')
        command = lines[0].strip()
        
        return command

    def get_progress_info(self):
        """Get current progress information"""
        return {
            "total_steps": len(self.initial_subtask_plan),
            "current_step": self.current_subtask_idx + 1,
            "completed": len(self.completed_subtasks),
            "current_subtask": self.get_current_subtask(),
            "initial_plan": self.initial_subtask_plan,
            "completed_subtasks": self.completed_subtasks
        }

    def reset(self):
        """Reset planning state"""
        self.current_task = None
        self.initial_subtask_plan = []
        self.subtask_queue = []
        self.current_subtask_idx = 0
        self.completed_subtasks = []
        print("Planner reset")
