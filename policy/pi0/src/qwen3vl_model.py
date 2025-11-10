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
import re


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
        self.current_subtask_idx = 0
        self.completed_subtasks = []  # Track what has been done
        
        # Store last motion command and completion status
        self.last_motion_command = None
        self.last_completion_status = None

    def set_task(self, task_instruction):
        """Set the main task instruction"""
        self.current_task = task_instruction
        self.initial_subtask_plan = []
        self.current_subtask_idx = 0
        self.completed_subtasks = []
        self.last_motion_command = None
        self.last_completion_status = None
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
        self.current_subtask_idx = 0

        print(f"\n{'='*60}")
        print(f"Generated Initial Plan ({len(subtasks)} high-level steps):")
        print(f"{'='*60}")
        for i, subtask in enumerate(subtasks):
            print(f"  Step {i+1}: {subtask}")
        print(f"{'='*60}\n")

        return subtasks

    def generate_motion_command_with_evaluation(self, images, state=None, max_new_tokens=512):
        """
        Generate motion-level command AND evaluate current subtask completion
        Uses perception-based evaluation instead of step counting

        Args:
            images: List of RGB images [front, right, left] or dict
            state: Optional robot state
            max_new_tokens: Maximum tokens to generate

        Returns:
            dict with keys:
                - 'motion_command': str, PI0 instruction for next action
                - 'current_subtask_complete': bool, whether current subtask is done
                - 'completion_reasoning': str, explanation for completion decision
                - 'progress_summary': str, overall progress description
        """
        if not self.initial_subtask_plan:
            raise ValueError("Must generate initial plan first using generate_initial_plan()")

        # Convert images to PIL format
        pil_images = self._prepare_images(images)

        # Construct progress-aware prompt with evaluation request
        prompt = self._construct_motion_command_with_evaluation_prompt(state)

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

        # Generate response
        output_text = self._generate_qwen_response(messages, max_new_tokens)

        # Parse structured output
        result = self._parse_motion_command_with_evaluation(output_text)
        
        # Store for tracking
        self.last_motion_command = result['motion_command']
        self.last_completion_status = result['current_subtask_complete']

        print(f"\n[Motion Planning & Evaluation]")
        print(f"  Current Step: {self.current_subtask_idx + 1}/{len(self.initial_subtask_plan)}")
        print(f"  Subtask: {self.get_current_subtask()}")
        print(f"  Motion Command: {result['motion_command']}")
        print(f"  Subtask Complete: {result['current_subtask_complete']}")
        print(f"  Reasoning: {result['completion_reasoning']}")
        print(f"  Progress: {result['progress_summary']}")

        return result

    def process_completion_evaluation(self, evaluation_result):
        """
        Process the completion evaluation and advance subtask if needed
        
        Args:
            evaluation_result: dict returned from generate_motion_command_with_evaluation
            
        Returns:
            bool: True if subtask was advanced, False otherwise
        """
        if evaluation_result['current_subtask_complete']:
            return self.mark_subtask_completed()
        return False

    def mark_subtask_completed(self):
        """Mark current subtask as completed and advance"""
        if self.current_subtask_idx < len(self.initial_subtask_plan):
            completed = self.initial_subtask_plan[self.current_subtask_idx]
            self.completed_subtasks.append(completed)
            self.current_subtask_idx += 1
            print(f"\n{'='*50}")
            print(f"✓ Subtask Completed: {completed}")
            print(f"  Progress: {self.current_subtask_idx}/{len(self.initial_subtask_plan)}")
            if self.current_subtask_idx < len(self.initial_subtask_plan):
                print(f"  Next Subtask: {self.initial_subtask_plan[self.current_subtask_idx]}")
            else:
                print(f"  🎉 All subtasks completed!")
            print(f"{'='*50}\n")
            return True
        return False

    def get_current_subtask(self):
        """Get current high-level subtask from initial plan"""
        if not self.initial_subtask_plan:
            return self.current_task

        if self.current_subtask_idx >= len(self.initial_subtask_plan):
            return None  # All subtasks completed

        return self.initial_subtask_plan[self.current_subtask_idx]

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

    def _construct_motion_command_with_evaluation_prompt(self, state=None):
        """Construct prompt for generating motion command WITH completion evaluation"""
        # Build context from initial plan
        plan_context = "Initial Plan:\n"
        for i, step in enumerate(self.initial_subtask_plan):
            status = "✓" if i < self.current_subtask_idx else "→" if i == self.current_subtask_idx else "○"
            plan_context += f"  {status} Step {i+1}: {step}\n"

        # Build completed steps context
        completed_context = ""
        if self.completed_subtasks:
            completed_context = "\nCompleted Steps:\n"
            for i, step in enumerate(self.completed_subtasks):
                completed_context += f"  ✓ {step}\n"

        # Current step
        current_step = self.initial_subtask_plan[self.current_subtask_idx] if self.current_subtask_idx < len(self.initial_subtask_plan) else "All steps completed"

        prompt = f"""You are controlling a dual-arm robot. Based on the initial plan, current visual observation, and execution state, you need to:
1. Generate the NEXT motion-level command for PI0
2. Evaluate whether the CURRENT subtask is complete based on visual perception

{plan_context}
{completed_context}
Current Subtask to Execute: {current_step}

Main Task: {self.current_task}

Your response must follow this EXACT format:

MOTION_COMMAND: [One specific motion instruction for both arms, describing ~10 seconds of action]
SUBTASK_COMPLETE: [YES or NO - based on what you SEE in the current image]
COMPLETION_REASONING: [Brief explanation of why the current subtask is/isn't complete based on visual evidence]
PROGRESS_SUMMARY: [One sentence describing overall task progress]

Requirements for MOTION_COMMAND:
- Describe what BOTH arms should do (e.g., "Left arm: grasp red block. Right arm: stay steady")
- Be specific about objects, positions, and actions (pick/place/move/grasp/release/reach)
- The command should progress toward completing the current subtask

Requirements for SUBTASK_COMPLETE evaluation:
- Answer YES only if you can SEE clear visual evidence that the current subtask goal is achieved
- Answer NO if the subtask is still in progress or hasn't started
- Base your decision on the CURRENT IMAGE, not assumptions
- Examples:
  * "Pick up red block" → YES if block is grasped and lifted, NO if still on table
  * "Stack blocks" → YES if blocks are stacked, NO if still arranging
  * "Move to position" → YES if object reached target location, NO if still moving

Generate your response:"""

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

    def _parse_motion_command_with_evaluation(self, output_text):
        """
        Parse structured output containing motion command and completion evaluation
        
        Returns:
            dict with keys: motion_command, current_subtask_complete, completion_reasoning, progress_summary
        """
        result = {
            'motion_command': '',
            'current_subtask_complete': False,
            'completion_reasoning': '',
            'progress_summary': ''
        }
        
        # Extract fields using regex
        motion_match = re.search(r'MOTION_COMMAND:\s*(.+?)(?=\n\s*(?:SUBTASK_COMPLETE|COMPLETION_REASONING|PROGRESS_SUMMARY|$))', 
                                output_text, re.IGNORECASE | re.DOTALL)
        complete_match = re.search(r'SUBTASK_COMPLETE:\s*(YES|NO)', output_text, re.IGNORECASE)
        reasoning_match = re.search(r'COMPLETION_REASONING:\s*(.+?)(?=\n\s*(?:MOTION_COMMAND|SUBTASK_COMPLETE|PROGRESS_SUMMARY|$))', 
                                   output_text, re.IGNORECASE | re.DOTALL)
        progress_match = re.search(r'PROGRESS_SUMMARY:\s*(.+?)(?=\n\s*(?:MOTION_COMMAND|SUBTASK_COMPLETE|COMPLETION_REASONING|$))', 
                                  output_text, re.IGNORECASE | re.DOTALL)
        
        if motion_match:
            result['motion_command'] = motion_match.group(1).strip()
        
        if complete_match:
            result['current_subtask_complete'] = complete_match.group(1).upper() == 'YES'
        
        if reasoning_match:
            result['completion_reasoning'] = reasoning_match.group(1).strip()
        
        if progress_match:
            result['progress_summary'] = progress_match.group(1).strip()
        
        # Fallback: if parsing failed, try to extract at least motion command
        if not result['motion_command']:
            # Try to find any actionable instruction
            lines = output_text.strip().split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['arm', 'grasp', 'pick', 'place', 'move', 'reach']):
                    result['motion_command'] = line.strip()
                    break
            
            # Last resort: use the whole output
            if not result['motion_command']:
                result['motion_command'] = output_text.strip()
        
        # Default reasoning if not provided
        if not result['completion_reasoning']:
            result['completion_reasoning'] = "Evaluation not explicitly provided by model"
        
        if not result['progress_summary']:
            result['progress_summary'] = f"Executing step {self.current_subtask_idx + 1}/{len(self.initial_subtask_plan)}"
        
        return result

    def get_progress_info(self):
        """Get current progress information"""
        return {
            "total_steps": len(self.initial_subtask_plan),
            "current_step": self.current_subtask_idx + 1,
            "completed": len(self.completed_subtasks),
            "current_subtask": self.get_current_subtask(),
            "initial_plan": self.initial_subtask_plan,
            "completed_subtasks": self.completed_subtasks,
            "last_motion_command": self.last_motion_command,
            "last_completion_status": self.last_completion_status
        }

    def reset(self):
        """Reset planning state"""
        self.current_task = None
        self.initial_subtask_plan = []
        self.current_subtask_idx = 0
        self.completed_subtasks = []
        self.last_motion_command = None
        self.last_completion_status = None
        print("Planner reset")
