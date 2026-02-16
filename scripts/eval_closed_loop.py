#!/usr/bin/env python3
"""Closed-loop evaluation for Reachy 2 GR00T policy in MuJoCo."""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

GROOT_PATH = Path(__file__).parent.parent / "Isaac-GR00T"
sys.path.insert(0, str(GROOT_PATH))


def load_modality_config(config_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("modality_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class ReachyGR00TController:
    ACTION_MIN = np.array([4.93, -11.67, -42.07, -126.53, -20.28, -20.16, -21.24])
    ACTION_MAX = np.array([41.03, 11.44, 42.19, -74.73, 20.34, 16.37, 21.08])

    def __init__(self, policy, reachy, arm: str = "right", control_freq: float = 15.0,
                 action_horizon: int = 16, execute_horizon: int = 4, output_dir: Optional[Path] = None,
                 save_debug_frames: bool = False, max_delta_per_step: float = None,
                 clip_to_training_bounds: bool = True):
        self.policy = policy
        self.reachy = reachy
        self.arm = arm
        self.control_freq = control_freq
        self.control_period = 1.0 / control_freq
        self.action_horizon = action_horizon
        self.execute_horizon = execute_horizon
        self.output_dir = output_dir
        self.save_debug_frames = save_debug_frames
        self.max_delta_per_step = max_delta_per_step
        self.clip_to_training_bounds = clip_to_training_bounds
        self.arm_ref = getattr(reachy, f"{arm[0]}_arm")

    def process_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        processed = action.copy()
        if self.clip_to_training_bounds:
            processed = np.clip(processed, self.ACTION_MIN, self.ACTION_MAX)
        if self.max_delta_per_step is not None:
            delta = np.clip(processed - current_state, -self.max_delta_per_step, self.max_delta_per_step)
            processed = current_state + delta
        return processed

    def get_camera_frame(self, camera: str = "front_cam") -> np.ndarray:
        import rpyc
        frame_netref = self.reachy.render_camera(camera, width=224, height=224)
        frame = np.asarray(rpyc.classic.obtain(frame_netref), dtype=np.uint8)
        if len(frame.shape) == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        return frame[:, :, ::-1]  # BGR -> RGB

    def get_joint_positions(self) -> np.ndarray:
        import rpyc
        positions = rpyc.classic.obtain(self.arm_ref.get_present_positions())
        return np.array(positions, dtype=np.float32)

    def get_gripper_state(self) -> float:
        return 0.8

    def save_frame(self, frame: np.ndarray, episode: int, step: int, suffix: str = ""):
        if not self.save_debug_frames or self.output_dir is None:
            return
        frames_dir = self.output_dir / "debug_frames" / f"episode_{episode:03d}"
        frames_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(frames_dir / f"step_{step:04d}{suffix}.png")

    def verify_cube_visibility(self, frame: np.ndarray, cube_color: str) -> dict:
        color_ranges = {
            "red": ([150, 0, 0], [255, 100, 100]),
            "green": ([0, 100, 0], [150, 255, 150]),
            "blue": ([0, 0, 150], [100, 100, 255]),
            "yellow": ([150, 150, 0], [255, 255, 100]),
            "cyan": ([0, 150, 150], [150, 255, 255]),
            "magenta": ([150, 0, 150], [255, 100, 255]),
            "orange": ([180, 80, 0], [255, 200, 100]),
            "purple": ([80, 0, 100], [200, 100, 255]),
        }

        if cube_color not in color_ranges:
            return {"visible": False, "reason": "unknown_color", "pixel_count": 0}

        low, high = [np.array(x, dtype=np.uint8) for x in color_ranges[cube_color]]
        mask = np.all((frame >= low) & (frame <= high), axis=2)
        pixel_count = np.sum(mask)
        visible = pixel_count >= 50

        result = {"visible": visible, "pixel_count": int(pixel_count), "min_required": 50}
        if visible:
            y_coords, x_coords = np.where(mask)
            centroid_x, centroid_y = int(np.mean(x_coords)), int(np.mean(y_coords))
            result["centroid"] = (centroid_x, centroid_y)
            h, w = frame.shape[:2]
            mid_x, mid_y = w // 2, h // 2
            if centroid_x < mid_x and centroid_y < mid_y:
                result["image_quadrant"] = "top-left"
            elif centroid_x >= mid_x and centroid_y < mid_y:
                result["image_quadrant"] = "top-right"
            elif centroid_x < mid_x:
                result["image_quadrant"] = "bottom-left"
            else:
                result["image_quadrant"] = "bottom-right"
        return result

    def get_cube_world_position(self) -> np.ndarray:
        try:
            return np.array(self.reachy.get_object_pose('cube')[:3])
        except Exception:
            return np.array([0, 0, 0])

    def prepare_observation(self, task_description: str) -> dict:
        frame = self.get_camera_frame("front_cam")
        joints = self.get_joint_positions()
        return {
            "video": {"front_cam": frame[np.newaxis, np.newaxis, :, :, :]},
            "state": {"arm_joints": joints[np.newaxis, np.newaxis, :]},
            "language": {"annotation.human.task_description": [[task_description]]},
        }

    def execute_action(self, arm_action: np.ndarray, gripper_action: float):
        joints = [self.arm_ref.shoulder.pitch, self.arm_ref.shoulder.roll, self.arm_ref.elbow.yaw,
                  self.arm_ref.elbow.pitch, self.arm_ref.wrist.roll, self.arm_ref.wrist.pitch, self.arm_ref.wrist.yaw]
        for i, joint in enumerate(joints):
            joint.goal_position = float(arm_action[i])

        if gripper_action > 0.5:
            self.arm_ref.gripper.open()
        else:
            self.arm_ref.gripper.close()

        self.reachy.send_goal_positions()
        time.sleep(0.02)

    def run_episode(self, task_description: str, max_steps: int = 200, verbose: bool = True,
                    episode_num: int = 0, cube_color: str = "red") -> dict:
        actions_executed = []
        step = 0
        visibility_checks = []

        if verbose:
            print(f"  Task: {task_description}")

        while step < max_steps:
            obs = self.prepare_observation(task_description)
            frame = obs['video']['front_cam'][0, 0]

            if step == 0 or step % 20 == 0:
                visibility = self.verify_cube_visibility(frame, cube_color)
                visibility["world_pos"] = self.get_cube_world_position().tolist()
                visibility["step"] = step
                visibility_checks.append(visibility)
                self.save_frame(frame, episode_num, step, f"_{cube_color}")

                if verbose and step == 0:
                    status = "VISIBLE" if visibility["visible"] else "NOT VISIBLE"
                    print(f"    Cube: {status} ({visibility['pixel_count']} pixels)")

            action_pred, _ = self.policy.get_action(obs)
            arm_actions = action_pred["arm_joints"][0]
            gripper_actions = action_pred["gripper"][0]

            current_state = self.get_joint_positions()
            for i in range(min(self.execute_horizon, max_steps - step)):
                arm_action = self.process_action(arm_actions[i], current_state)
                self.execute_action(arm_action, gripper_actions[i, 0])
                actions_executed.append(np.concatenate([arm_action, [gripper_actions[i, 0]]]))
                current_state = arm_action
                step += 1

                if verbose and step % 20 == 0:
                    print(f"    Step {step}/{max_steps}")

            time.sleep(0.05)

        success = self.check_success()
        self.save_frame(self.get_camera_frame("front_cam"), episode_num, step, "_final")

        return {"success": success, "steps": step, "actions": actions_executed, "cube_visibility": visibility_checks}

    def check_success(self) -> bool:
        try:
            obj_pos = np.array(self.reachy.get_object_pose('cube')[:3])
            box_center = np.array([0.45, -0.05, 0.65])
            return (abs(obj_pos[0] - box_center[0]) < 0.06 and
                    abs(obj_pos[1] - box_center[1]) < 0.06 and
                    0.63 < obj_pos[2] < 0.75)
        except Exception:
            return False

    def reset_episode(self, randomize: bool = True):
        self.reachy.reset_scene()
        time.sleep(0.5)

        home_joints = [22, 0, 0, -110, 0, -3, 0]
        joints = [self.arm_ref.shoulder.pitch, self.arm_ref.shoulder.roll, self.arm_ref.elbow.yaw,
                  self.arm_ref.elbow.pitch, self.arm_ref.wrist.roll, self.arm_ref.wrist.pitch, self.arm_ref.wrist.yaw]
        for i, joint in enumerate(joints):
            joint.goal_position = home_joints[i]
        self.arm_ref.gripper.open()
        self.reachy.send_goal_positions()
        time.sleep(2.0)

        if randomize:
            try:
                import random
                x = random.uniform(0.28, 0.42)
                y = random.uniform(-0.28, -0.10) if self.arm == "right" else random.uniform(0.10, 0.28)
                self.reachy.set_object_pose('cube', [x, y, 0.65], [1, 0, 0, 0])

                colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]
                color = random.choice(colors)
                rgba = {"red": [1, 0, 0, 1], "green": [0, 1, 0, 1], "blue": [0, 0, 1, 1],
                        "yellow": [1, 1, 0, 1], "cyan": [0, 1, 1, 1], "magenta": [1, 0, 1, 1],
                        "orange": [1, 0.5, 0, 1], "purple": [0.5, 0, 1, 1]}[color]
                self.reachy.set_object_color('cube', rgba)
                return color
            except Exception:
                return "unknown"
        return "red"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--modality-config", type=str, default="configs/reachy2_modality_config.py")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="eval_results")
    parser.add_argument("--execute-horizon", type=int, default=4)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--max-delta", type=float, default=None)
    parser.add_argument("--no-clip", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_modality_config(args.modality_config)

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from reachy2_mujoco import ReachySDK

    print(f"Loading model from {args.model_path}...")
    policy = Gr00tPolicy(embodiment_tag=EmbodimentTag.REACHY2, model_path=args.model_path,
                         device=args.device, strict=True)

    print("Connecting to Reachy MuJoCo...")
    reachy = ReachySDK("localhost")

    controller = ReachyGR00TController(
        policy=policy, reachy=reachy, arm=args.arm, control_freq=15.0,
        execute_horizon=args.execute_horizon, output_dir=output_dir,
        save_debug_frames=args.save_frames, max_delta_per_step=args.max_delta,
        clip_to_training_bounds=not args.no_clip,
    )

    results = []
    successes = 0

    print(f"\nRunning {args.episodes} episodes...")

    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")
        color = controller.reset_episode(randomize=not args.no_randomize)
        task = f"Pick up the {color} cube and place it in the box"

        result = controller.run_episode(task, max_steps=args.max_steps, verbose=True,
                                        episode_num=ep, cube_color=color)
        result["episode"] = ep
        result["task"] = task
        results.append(result)

        if result["success"]:
            successes += 1
        print(f"  Result: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"  Success rate: {successes}/{ep + 1} ({100 * successes / (ep + 1):.1f}%)")

    print(f"\nFinal: {successes}/{args.episodes} ({100 * successes / args.episodes:.1f}%)")

    def convert_for_json(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    results_path = output_dir / "closed_loop_results.json"
    json_results = [{k: convert_for_json(v) for k, v in r.items() if k != "actions"} for r in results]
    for jr, r in zip(json_results, results):
        jr["num_actions"] = len(r["actions"])

    with open(results_path, "w") as f:
        json.dump({"config": vars(args), "summary": {"total_episodes": args.episodes,
                  "successes": successes, "success_rate": successes / args.episodes},
                  "episodes": json_results}, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
