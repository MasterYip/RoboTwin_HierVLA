#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hierarchical VLA Policy: Qwen3VL (High-level Planner) + PI0 (Low-level Executor)
Implements hierarchical prompting strategy
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
    - High-level: Qwen3VL for task decomposition into subtasks
    - Low-level: PI0 for executing each subtask
    """
    
    def __init__(self, 
                 train_config_name, 
                 model_name, 
                 checkpoint_id, 
                 pi0_step,
                 qwen_model_path="/inspire/ssd/project/25jinqiu07/public/hiervla_003/Qwen3-VL-8B-Instruct",
                 replan_frequency=10):
        """
        Initialize hierarchical policy
        
        Args:
            train_config_name: PI0 training config name
            model_name: PI0 model name
            checkpoint_id: PI0 checkpoint ID
            pi0_step: Number of steps PI0 executes per inference
            qwen_model_path: Path to Qwen3VL model
            replan_frequency: Steps between replanning (0 = plan once at start)
        """
        print("=" * 60)
        print("Initializing Hierarchical Qwen3VL + PI0 Policy")
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
        
        # Hierarchical state tracking
        self.step_count = 0
        self.subtask_step_count = 0
        self.max_subtask_steps = 50  # Max steps per subtask before advancing
        
        print("=" * 60)
        print("Hierarchical Policy Initialized Successfully!")
        print(f"  - Planner: Qwen3VL")
        print(f"  - Executor: PI0 (step={pi0_step})")
        print(f"  - Replan frequency: {replan_frequency}")
        print("=" * 60)
    
    @property
    def observation_window(self):
        """Expose executor's observation window for compatibility"""
        return self.executor.observation_window
    
    def set_language(self, instruction):
        """
        Set main task instruction
        This triggers high-level planning
        
        Args:
            instruction: Main task description
        """
        print(f"\n[Hierarchical Policy] Main Task: {instruction}")
        self.planner.set_task(instruction)
        self.step_count = 0
        self.subtask_step_count = 0
        
    def update_observation_window(self, img_arr, state):
        """
        Update observation and perform hierarchical planning if needed
        
        Args:
            img_arr: List of RGB images [front, right, left]
            state: Robot joint state
        """
        # Step 1: Check if we need to generate/update subtask plan
        if self.step_count == 0:
            # Initial planning
            print("\n[High-level Planner] Generating initial subtask plan...")
            self.planner.generate_subtasks(img_arr, state)
            
        elif self.replan_frequency > 0 and self.step_count % self.replan_frequency == 0:
            # Periodic replanning
            print(f"\n[High-level Planner] Replanning at step {self.step_count}...")
            self.planner.generate_subtasks(img_arr, state)
        
        # Step 2: Get current subtask instruction
        current_subtask = self.planner.get_current_subtask()
        
        if current_subtask is None:
            print("[Hierarchical Policy] All subtasks completed, using main task")
            current_subtask = self.planner.current_task
        else:
            print(f"[Low-level Executor] Current subtask: {current_subtask}")
        
        # Step 3: Update low-level executor with current subtask
        self.executor.set_language(current_subtask)
        self.executor.update_observation_window(img_arr, state)
        
        # Step 4: Check if we should advance to next subtask
        self.subtask_step_count += 1
        if self.subtask_step_count >= self.max_subtask_steps:
            print(f"\n[Hierarchical Policy] Max steps reached for current subtask, advancing...")
            if self.planner.advance_subtask():
                self.subtask_step_count = 0
        
        self.step_count += 1
    
    def get_action(self):
        """
        Get action from low-level executor
        
        Returns:
            Action array from PI0
        """
        return self.executor.get_action()
    
    def advance_subtask_manual(self):
        """Manually advance to next subtask (for external control)"""
        if self.planner.advance_subtask():
            self.subtask_step_count = 0
            return True
        return False
    
    def set_max_subtask_steps(self, steps):
        """Set maximum steps per subtask"""
        self.max_subtask_steps = steps
        print(f"Set max subtask steps to {steps}")
    
    def reset_obsrvationwindows(self):
        """Reset both planner and executor state"""
        self.planner.reset()
        self.executor.reset_obsrvationwindows()
        self.step_count = 0
        self.subtask_step_count = 0
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
        return {
            "main_task": self.planner.current_task,
            "subtasks": self.planner.subtask_queue,
            "current_subtask_idx": self.planner.current_subtask_idx,
            "current_subtask": self.planner.get_current_subtask(),
            "step_count": self.step_count,
            "subtask_step_count": self.subtask_step_count
        }


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
