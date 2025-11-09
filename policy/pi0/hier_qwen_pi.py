#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical VLA Policy: Qwen3VL (High-level Planner) + PI0 (Low-level Executor)
Implements hierarchical prompting strategy with consistent planning
"""

import numpy as np
import sys
import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
sys.path.append(os.path.join(parent_directory, "src"))

from pi_model import PI0
from qwen3vl_model import Qwen3VLPlanner


class HierarchicalQwenPI0:
    """
    Hierarchical VLA Policy with two-level architecture:
    - High-level: Qwen3VL for task decomposition and progress-aware motion planning
    - Low-level: PI0 for executing motion commands

    Planning Strategy:
    1. Generate initial high-level plan once at the beginning
    2. During execution, generate motion-level commands while tracking progress
    3. Motion commands reference the initial plan for consistency
    """

    def __init__(self,
                 train_config_name,
                 model_name,
                 checkpoint_id,
                 pi0_step,
                 qwen_model_path="/inspire/ssd/project/25jinqiu07/public/hiervla_003/Qwen3-VL-8B-Instruct",
                 replan_frequency=10,
                 steps_per_subtask=50):
        """
        Initialize hierarchical policy

        Args:
            train_config_name: PI0 training config name
            model_name: PI0 model name
            checkpoint_id: PI0 checkpoint ID
            pi0_step: Number of steps PI0 executes per inference
            qwen_model_path: Path to Qwen3VL model
            replan_frequency: Steps between motion command regeneration (0 = no replanning)
            steps_per_subtask: Estimated steps per high-level subtask
        """
        print("=" * 60)
        print("Initializing Hierarchical Qwen3VL + PI0 Policy")
        print("Strategy: Initial Planning + Progress-Aware Motion Commands")
        print("=" * 60)

        # High-level planner
        self.planner = Qwen3VLPlanner(
            model_path=qwen_model_path,
            device_map="auto",
            dtype="auto"
        )

        # Low-level executor
        self.executor = PI0(
            train_config_name=train_config_name,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            pi0_step=pi0_step
        )

        self.pi0_step = pi0_step
        self.replan_frequency = replan_frequency
        self.steps_per_subtask = steps_per_subtask

        # Hierarchical state tracking
        self.step_count = 0
        self.motion_command_step_count = 0  # Steps since last motion command
        self.current_motion_command = None
        self.initial_plan_generated = False

        print("=" * 60)
        print("Hierarchical Policy Initialized Successfully!")
        print(f"  - Planner: Qwen3VL (two-phase planning)")
        print(f"  - Executor: PI0 (step={pi0_step})")
        print(f"  - Motion command regeneration: every {replan_frequency} steps")
        print(f"  - Estimated steps per subtask: {steps_per_subtask}")
        print("=" * 60)

    @property
    def observation_window(self):
        """Expose executor's observation window for compatibility"""
        return self.executor.observation_window

    def set_language(self, instruction):
        """
        Set main task instruction
        This will trigger initial planning at first observation

        Args:
            instruction: Main task description
        """
        print(f"\n[Hierarchical Policy] Main Task: {instruction}")
        self.planner.set_task(instruction)
        self.step_count = 0
        self.motion_command_step_count = 0
        self.current_motion_command = None
        self.initial_plan_generated = False

    def update_observation_window(self, img_arr, state):
        """
        Update observation and perform hierarchical planning

        Phase 1 (step 0): Generate initial high-level plan
        Phase 2 (ongoing): Generate motion-level commands based on progress

        Args:
            img_arr: List of RGB images [front, right, left]
            state: Robot joint state
        """
        # Phase 1: Generate initial high-level plan (only once at start)
        if not self.initial_plan_generated:
            print("\n" + "=" * 60)
            print("[Phase 1] Generating Initial High-Level Plan...")
            print("=" * 60)
            self.planner.generate_initial_plan(img_arr, state)
            self.initial_plan_generated = True

        # Phase 2: Generate/update motion-level command based on progress
        need_new_command = False

        if self.current_motion_command is None:
            # First motion command
            need_new_command = True
            print("\n[Phase 2] Generating first motion command...")

        elif self.replan_frequency > 0 and self.motion_command_step_count >= self.replan_frequency:
            # Time to regenerate motion command
            need_new_command = True
            print(f"\n[Phase 2] Regenerating motion command (step {self.step_count})...")

        # Check if we should advance to next high-level subtask
        progress_info = self.planner.get_progress_info()
        if self.step_count > 0 and self.step_count % self.steps_per_subtask == 0:
            if progress_info['current_step'] <= progress_info['total_steps']:
                print(f"\n[Progress Check] ~{self.steps_per_subtask} steps completed, considering subtask advancement...")
                # Could add automatic advancement here if needed
                # For now, manual control allows better flexibility

        # Generate new motion command if needed
        if need_new_command:
            self.current_motion_command = self.planner.generate_motion_command(img_arr, state)
            self.motion_command_step_count = 0

            # Display progress
            progress = self.planner.get_progress_info()
            print(f"[Progress] Step {progress['current_step']}/{progress['total_steps']}: "
                  f"{progress['current_subtask']}")

        # Update low-level executor with current motion command
        if self.current_motion_command:
            self.executor.set_language(self.current_motion_command)
        else:
            # Fallback to current high-level subtask
            current_subtask = self.planner.get_current_subtask()
            if current_subtask:
                self.executor.set_language(current_subtask)

        # Update executor observation
        self.executor.update_observation_window(img_arr, state)

        # Increment counters
        self.step_count += 1
        self.motion_command_step_count += 1

    def get_action(self):
        """
        Get action from low-level executor

        Returns:
            Action array from PI0
        """
        return self.executor.get_action()

    def advance_subtask_manual(self):
        """
        Manually advance to next high-level subtask
        Use this when current subtask is visually completed

        Returns:
            bool: True if advanced, False if no more subtasks
        """
        if self.planner.mark_subtask_completed():
            # Reset motion command to trigger regeneration
            self.current_motion_command = None
            self.motion_command_step_count = 0
            return True
        return False

    def set_steps_per_subtask(self, steps):
        """Set estimated steps per high-level subtask"""
        self.steps_per_subtask = steps
        print(f"Set steps per subtask to {steps}")

    def set_replan_frequency(self, frequency):
        """Set motion command regeneration frequency"""
        self.replan_frequency = frequency
        print(f"Set motion command regeneration frequency to {frequency}")

    def reset_obsrvationwindows(self):
        """Reset both planner and executor state"""
        self.planner.reset()
        self.executor.reset_obsrvationwindows()
        self.step_count = 0
        self.motion_command_step_count = 0
        self.current_motion_command = None
        self.initial_plan_generated = False
        print("Hierarchical policy reset")

    # Additional compatibility methods
    def set_img_size(self, img_size):
        """Set image size for executor"""
        self.executor.set_img_size(img_size)

    @property
    def instruction(self):
        """Get current instruction (for compatibility)"""
        return self.planner.current_task

    def get_planning_state(self):
        """Get current planning state for debugging/logging"""
        progress = self.planner.get_progress_info()
        return {
            "main_task": self.planner.current_task,
            "initial_plan": progress['initial_plan'],
            "current_step": progress['current_step'],
            "total_steps": progress['total_steps'],
            "completed_subtasks": progress['completed_subtasks'],
            "current_subtask": progress['current_subtask'],
            "current_motion_command": self.current_motion_command,
            "step_count": self.step_count,
            "motion_command_step_count": self.motion_command_step_count,
            "initial_plan_generated": self.initial_plan_generated
        }

    def print_progress(self):
        """Print current planning progress (for debugging)"""
        state = self.get_planning_state()
        print("\n" + "=" * 60)
        print("Hierarchical Policy Progress")
        print("=" * 60)
        print(f"Main Task: {state['main_task']}")
        print(f"\nInitial Plan:")
        for i, step in enumerate(state['initial_plan']):
            status = "✓" if i < state['current_step'] - 1 else "→" if i == state['current_step'] - 1 else "○"
            print(f"  {status} Step {i+1}: {step}")
        print(f"\nCurrent Motion Command: {state['current_motion_command']}")
        print(f"Total Steps Executed: {state['step_count']}")
        print(f"Steps since Motion Command: {state['motion_command_step_count']}")
        print("=" * 60 + "\n")


# Factory function for compatibility with deploy_policy.py
def create_hierarchical_policy(train_config_name, model_name, checkpoint_id, pi0_step, **kwargs):
    """
    Factory function to create hierarchical policy
    Compatible with get_model() interface in deploy_policy.py
    """
    return HierarchicalQwenPI0(
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        pi0_step=pi0_step,
        **kwargs
    )
