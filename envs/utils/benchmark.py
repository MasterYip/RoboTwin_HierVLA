#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark utilities for evaluating robot policies
Tracks metrics like success rate, completion steps, action smoothness, etc.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import time


class EpisodeBenchmark:
    """
    Benchmark tracker for a single episode
    Tracks detailed metrics during task execution
    """
    
    def __init__(self, episode_id: int, seed: int, task_name: str, instruction: str):
        self.episode_id = episode_id
        self.seed = seed
        self.task_name = task_name
        self.instruction = instruction
        
        # Timing
        self.start_time = time.time()
        self.end_time = None
        
        # Success tracking
        self.success = False
        self.completion_steps = 0
        self.step_limit = None
        
        # Action tracking
        self.actions = []  # Store all actions
        self.joint_states = []  # Store joint states at each step
        
        # Smoothness metrics
        self.action_velocities = []  # Action differences between steps
        self.joint_accelerations = []  # Joint acceleration (second derivative)
        
        # Performance metrics
        self.planning_failures = 0
        self.collision_count = 0
        
    def record_step(self, action: np.ndarray, joint_state: np.ndarray):
        """Record a single step with action and joint state"""
        self.actions.append(action.copy())
        self.joint_states.append(joint_state.copy())
        self.completion_steps += 1
        
        # Calculate action velocity (difference between consecutive actions)
        if len(self.actions) > 1:
            action_diff = np.abs(self.actions[-1] - self.actions[-2])
            self.action_velocities.append(action_diff)
        
        # Calculate joint acceleration (second derivative)
        if len(self.joint_states) > 2:
            vel_curr = self.joint_states[-1] - self.joint_states[-2]
            vel_prev = self.joint_states[-2] - self.joint_states[-3]
            accel = np.abs(vel_curr - vel_prev)
            self.joint_accelerations.append(accel)
    
    def mark_success(self, success: bool):
        """Mark episode as success/failure"""
        self.success = success
        self.end_time = time.time()
    
    def set_step_limit(self, step_limit: int):
        """Set the step limit for this episode"""
        self.step_limit = step_limit
    
    def record_planning_failure(self):
        """Record a planning failure"""
        self.planning_failures += 1
    
    def record_collision(self):
        """Record a collision event"""
        self.collision_count += 1
    
    def compute_smoothness_metrics(self) -> Dict[str, float]:
        """
        Compute action smoothness metrics
        
        Returns:
            dict with smoothness scores
        """
        metrics = {}
        
        if len(self.action_velocities) > 0:
            action_vels = np.array(self.action_velocities)
            
            # Mean action change (lower is smoother)
            metrics['mean_action_change'] = float(np.mean(action_vels))
            metrics['max_action_change'] = float(np.max(action_vels))
            metrics['std_action_change'] = float(np.std(action_vels))
            
            # Smoothness score (inverse of variance, normalized)
            variance = np.var(action_vels)
            metrics['action_smoothness_score'] = float(1.0 / (1.0 + variance))
        else:
            metrics['mean_action_change'] = 0.0
            metrics['max_action_change'] = 0.0
            metrics['std_action_change'] = 0.0
            metrics['action_smoothness_score'] = 1.0
        
        if len(self.joint_accelerations) > 0:
            joint_accels = np.array(self.joint_accelerations)
            
            # Joint jerk metrics (lower is smoother)
            metrics['mean_joint_acceleration'] = float(np.mean(joint_accels))
            metrics['max_joint_acceleration'] = float(np.max(joint_accels))
            
            # Joint smoothness score
            jerk_variance = np.var(joint_accels)
            metrics['joint_smoothness_score'] = float(1.0 / (1.0 + jerk_variance))
        else:
            metrics['mean_joint_acceleration'] = 0.0
            metrics['max_joint_acceleration'] = 0.0
            metrics['joint_smoothness_score'] = 1.0
        
        # Overall smoothness score (average of action and joint smoothness)
        metrics['overall_smoothness'] = (
            metrics['action_smoothness_score'] + metrics['joint_smoothness_score']
        ) / 2.0
        
        return metrics
    
    def to_dict(self) -> Dict:
        """
        Convert episode benchmark to dictionary for JSON export
        
        Returns:
            dict with all metrics
        """
        duration = self.end_time - self.start_time if self.end_time else 0.0
        smoothness = self.compute_smoothness_metrics()
        
        return {
            'episode_id': self.episode_id,
            'seed': self.seed,
            'task_name': self.task_name,
            'instruction': self.instruction,
            'success': self.success,
            'completion_steps': self.completion_steps,
            'step_limit': self.step_limit,
            'step_utilization': self.completion_steps / self.step_limit if self.step_limit else 0.0,
            'duration_seconds': duration,
            'planning_failures': self.planning_failures,
            'collision_count': self.collision_count,
            'smoothness_metrics': smoothness,
        }


class PolicyBenchmark:
    """
    Aggregated benchmark across multiple episodes
    Computes aggregate statistics for policy evaluation
    """
    
    def __init__(self, policy_name: str, task_config: str, ckpt_setting: str):
        self.policy_name = policy_name
        self.task_config = task_config
        self.ckpt_setting = ckpt_setting
        
        self.episodes: List[EpisodeBenchmark] = []
        self.current_episode: Optional[EpisodeBenchmark] = None
    
    def start_episode(self, episode_id: int, seed: int, task_name: str, instruction: str, step_limit: int):
        """Start tracking a new episode"""
        self.current_episode = EpisodeBenchmark(episode_id, seed, task_name, instruction)
        self.current_episode.set_step_limit(step_limit)
    
    def record_step(self, action: np.ndarray, joint_state: np.ndarray):
        """Record a step in current episode"""
        if self.current_episode:
            self.current_episode.record_step(action, joint_state)
    
    def mark_episode_success(self, success: bool):
        """Mark current episode as complete"""
        if self.current_episode:
            self.current_episode.mark_success(success)
            self.episodes.append(self.current_episode)
            self.current_episode = None
    
    def record_planning_failure(self):
        """Record planning failure in current episode"""
        if self.current_episode:
            self.current_episode.record_planning_failure()
    
    def record_collision(self):
        """Record collision in current episode"""
        if self.current_episode:
            self.current_episode.record_collision()
    
    def compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all episodes
        
        Returns:
            dict with aggregated statistics
        """
        if not self.episodes:
            return {}
        
        episodes_data = [ep.to_dict() for ep in self.episodes]
        
        # Success metrics
        successes = [ep['success'] for ep in episodes_data]
        success_rate = np.mean(successes)
        
        # Step metrics
        steps = [ep['completion_steps'] for ep in episodes_data]
        successful_steps = [ep['completion_steps'] for ep in episodes_data if ep['success']]
        
        # Duration metrics
        durations = [ep['duration_seconds'] for ep in episodes_data]
        
        # Smoothness metrics (aggregate from successful episodes)
        successful_episodes = [ep for ep in episodes_data if ep['success']]
        if successful_episodes:
            smoothness_scores = [ep['smoothness_metrics']['overall_smoothness'] for ep in successful_episodes]
            action_smoothness = [ep['smoothness_metrics']['action_smoothness_score'] for ep in successful_episodes]
            joint_smoothness = [ep['smoothness_metrics']['joint_smoothness_score'] for ep in successful_episodes]
            mean_action_change = [ep['smoothness_metrics']['mean_action_change'] for ep in successful_episodes]
        else:
            smoothness_scores = [0.0]
            action_smoothness = [0.0]
            joint_smoothness = [0.0]
            mean_action_change = [0.0]
        
        # Planning and collision metrics
        planning_failures = [ep['planning_failures'] for ep in episodes_data]
        collisions = [ep['collision_count'] for ep in episodes_data]
        
        aggregate = {
            'policy_name': self.policy_name,
            'task_config': self.task_config,
            'ckpt_setting': self.ckpt_setting,
            'total_episodes': len(self.episodes),
            'success_metrics': {
                'success_rate': float(success_rate),
                'success_count': int(np.sum(successes)),
                'failure_count': int(len(successes) - np.sum(successes)),
            },
            'step_metrics': {
                'mean_steps': float(np.mean(steps)),
                'std_steps': float(np.std(steps)),
                'min_steps': int(np.min(steps)),
                'max_steps': int(np.max(steps)),
                'mean_steps_successful': float(np.mean(successful_steps)) if successful_steps else 0.0,
            },
            'duration_metrics': {
                'mean_duration': float(np.mean(durations)),
                'std_duration': float(np.std(durations)),
                'total_duration': float(np.sum(durations)),
            },
            'smoothness_metrics': {
                'mean_overall_smoothness': float(np.mean(smoothness_scores)),
                'std_overall_smoothness': float(np.std(smoothness_scores)),
                'mean_action_smoothness': float(np.mean(action_smoothness)),
                'mean_joint_smoothness': float(np.mean(joint_smoothness)),
                'mean_action_change': float(np.mean(mean_action_change)),
            },
            'robustness_metrics': {
                'mean_planning_failures': float(np.mean(planning_failures)),
                'total_planning_failures': int(np.sum(planning_failures)),
                'mean_collisions': float(np.mean(collisions)),
                'total_collisions': int(np.sum(collisions)),
            },
        }
        
        return aggregate
    
    def get_episodes_list(self) -> List[Dict]:
        """
        Get list of all episode dictionaries for JSON export
        
        Returns:
            list of episode dicts
        """
        return [ep.to_dict() for ep in self.episodes]
    
    def to_dict(self) -> Dict:
        """
        Complete benchmark data for JSON export
        
        Returns:
            dict with aggregate metrics and all episodes
        """
        return {
            'aggregate_metrics': self.compute_aggregate_metrics(),
            'episodes': self.get_episodes_list(),
        }
