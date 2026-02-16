#!/usr/bin/env python3
"""
NOVA Robot Pipeline - Voice/text to robot action via Parakeet, Cosmos, and GR00T.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np

GROOT_PATH = Path(__file__).parent.parent / "Isaac-GR00T"
sys.path.insert(0, str(GROOT_PATH))


class RobotPipeline:
    def __init__(self, model_path: Optional[str] = None, use_asr: bool = False,
                 use_cosmos: bool = True, device: str = "cuda", mock: bool = False,
                 arm: str = "right"):
        self.model_path = model_path
        self.use_asr = use_asr
        self.use_cosmos = use_cosmos
        self.device = device
        self.mock = mock
        self.arm = arm

        self.asr = None
        self.cosmos = None
        self.policy = None
        self.reachy = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        print("Initializing pipeline...")

        if self.use_asr:
            self._init_asr()
        if self.use_cosmos:
            self._init_cosmos()
        if self.model_path:
            self._init_policy()
        self._init_reachy()

        self._initialized = True
        print("Pipeline ready")

    def _init_asr(self):
        if self.mock:
            self.asr = MockASR()
        else:
            from parakeet_asr import ParakeetASR
            self.asr = ParakeetASR(device=self.device)
            self.asr.load_model()

    def _init_cosmos(self):
        if self.mock:
            from cosmos_reason import MockCosmosReason
            self.cosmos = MockCosmosReason()
        else:
            from cosmos_reason import CosmosReason
            self.cosmos = CosmosReason(model_size="2b", device=self.device)
            self.cosmos.load_model()

    def _init_policy(self):
        if self.mock:
            self.policy = MockPolicy()
            return

        config_path = Path(__file__).parent.parent / "configs/reachy2_modality_config.py"
        if config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("modality_config", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.REACHY2,
            model_path=self.model_path,
            device=self.device,
            strict=True,
        )

    def _init_reachy(self):
        if self.mock:
            self.reachy = MockReachy()
            return
        try:
            from reachy2_mujoco import ReachySDK
            self.reachy = ReachySDK("localhost")
        except Exception as e:
            print(f"Reachy connection failed: {e}, using mock")
            self.reachy = MockReachy()

    def get_camera_frame(self, camera: str = "front_cam") -> np.ndarray:
        if hasattr(self.reachy, "cameras"):
            frame = np.array(self.reachy.cameras.get_frame(camera))
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            return frame.astype(np.uint8)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_state(self) -> np.ndarray:
        arm_ref = getattr(self.reachy, f"{self.arm[0]}_arm", None)
        if arm_ref:
            return np.array(arm_ref.get_present_positions(), dtype=np.float32)
        return np.zeros(7, dtype=np.float32)

    def get_voice_command(self, duration: float = 3.0) -> str:
        if not self.asr:
            raise RuntimeError("ASR not initialized")
        print("Listening...")
        text = self.asr.record_and_transcribe(duration=duration)
        print(f"Heard: \"{text}\"")
        return text

    def reason_about_task(self, task_description: str, image: Optional[np.ndarray] = None) -> Dict:
        if not self.cosmos:
            return {"gr00t_instruction": task_description, "steps": [],
                    "target_object": "", "target_location": ""}

        if image is not None:
            self.cosmos.analyze_scene(image)
        plan = self.cosmos.plan_task(task_description, image)
        return plan

    def predict_action(self, instruction: str, image: np.ndarray, state: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.policy:
            raise RuntimeError("GR00T policy not initialized")

        obs = {
            "video": {"front_cam": image[np.newaxis, np.newaxis, :, :, :]},
            "state": {"arm_joints": state[np.newaxis, np.newaxis, :]},
            "language": {"annotation.human.task_description": [[instruction]]},
        }
        action_pred, _ = self.policy.get_action(obs)
        return {
            "arm_joints": action_pred["arm_joints"][0],
            "gripper": action_pred["gripper"][0],
        }

    def execute_action(self, arm_action: np.ndarray, gripper_action: float, duration: float = 0.067):
        arm_ref = getattr(self.reachy, f"{self.arm[0]}_arm", None)
        if not arm_ref:
            return

        arm_ref.goto(arm_action.tolist(), duration=duration)
        if gripper_action > 0.5:
            arm_ref.gripper.open()
        else:
            arm_ref.gripper.close()
        self.reachy.send_goal_positions()

    def run_task(self, task_description: str, max_steps: int = 200,
                 execute_horizon: int = 8, verbose: bool = True) -> Dict:
        self.initialize()

        if verbose:
            print(f"Task: {task_description}")

        image = self.get_camera_frame()
        plan = self.reason_about_task(task_description, image)
        instruction = plan["gr00t_instruction"]

        step = 0
        control_period = 1.0 / 15.0

        while step < max_steps:
            loop_start = time.time()
            image = self.get_camera_frame()
            state = self.get_state()

            if self.policy:
                actions = self.predict_action(instruction, image, state)
                arm_actions = actions["arm_joints"]
                gripper_actions = actions["gripper"]
            else:
                arm_actions = np.zeros((16, 7))
                gripper_actions = np.ones((16, 1)) * 0.8

            for i in range(min(execute_horizon, max_steps - step)):
                self.execute_action(arm_actions[i], gripper_actions[i, 0], duration=control_period)

                elapsed = time.time() - loop_start
                if control_period - elapsed > 0:
                    time.sleep(control_period - elapsed)
                loop_start = time.time()
                step += 1

                if verbose and step % 20 == 0:
                    print(f"Step {step}/{max_steps}")

        success = self._check_success()
        if verbose:
            print(f"Result: {'SUCCESS' if success else 'INCOMPLETE'}")

        return {"success": success, "steps": step, "instruction": instruction, "plan": plan}

    def _check_success(self) -> bool:
        try:
            if hasattr(self.reachy, "_reachy_mujoco"):
                obj_pos = np.array(self.reachy._reachy_mujoco.get_object_pose()[:3])
                box_center = np.array([0.45, -0.05, 0.65])
                return np.linalg.norm(obj_pos - box_center) < 0.08
        except Exception:
            pass
        return False

    def interactive_mode(self):
        self.initialize()
        print("Interactive mode - Enter=speak, t=text, q=quit")

        while True:
            try:
                cmd = input("\n> ").strip().lower()
                if cmd == "q":
                    break
                elif cmd == "t":
                    task = input("Task: ").strip()
                else:
                    task = self.get_voice_command()
                if task:
                    self.run_task(task, verbose=True)
            except KeyboardInterrupt:
                break


class MockASR:
    def record_and_transcribe(self, duration=3.0):
        return "pick up the red cube and place it in the box"
    def load_model(self):
        pass


class MockPolicy:
    def get_action(self, obs):
        return {"arm_joints": np.zeros((1, 16, 7)), "gripper": np.ones((1, 16, 1)) * 0.8}, None


class MockReachy:
    def __init__(self):
        self.r_arm = MockArm()
        self.l_arm = MockArm()
    def send_goal_positions(self):
        pass


class MockArm:
    def __init__(self):
        self.gripper = MockGripper()
    def get_present_positions(self):
        return [0.0] * 7
    def goto(self, target, duration=1.0):
        pass


class MockGripper:
    def open(self): pass
    def close(self): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--task", "-t", type=str)
    parser.add_argument("--voice", "-v", action="store_true")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--no-cosmos", action="store_true")
    parser.add_argument("--arm", default="right", choices=["right", "left"])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    pipeline = RobotPipeline(
        model_path=args.model_path,
        use_asr=args.voice or args.interactive,
        use_cosmos=not args.no_cosmos,
        device=args.device,
        mock=args.mock,
        arm=args.arm,
    )

    if args.interactive:
        pipeline.interactive_mode()
    elif args.task:
        pipeline.run_task(args.task, max_steps=args.max_steps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
