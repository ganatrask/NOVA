#!/usr/bin/env python3
"""NOVA Gradio GUI - Parakeet ASR + Cosmos + GR00T + Reachy 2 MuJoCo."""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

GROOT_PATH = Path(__file__).parent.parent / "Isaac-GR00T"
sys.path.insert(0, str(GROOT_PATH))

import gradio as gr


@dataclass
class PipelineState:
    mujoco_connected: bool = False
    cosmos_available: bool = False
    groot_loaded: bool = False
    is_executing: bool = False
    should_stop: bool = False
    current_step: int = 0
    max_steps: int = 200
    reachy: object = None
    cosmos_client: object = None
    asr: object = None
    policy: object = None
    controller: object = None
    model_path: str = ""
    mock_mode: bool = False
    arm: str = "right"
    log_messages: List[str] = field(default_factory=list)


state = PipelineState()


def log(message: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S")
    state.log_messages.append(f"[{ts}] {message}")
    if len(state.log_messages) > 50:
        state.log_messages = state.log_messages[-50:]
    return "\n".join(state.log_messages)


def connect_mujoco() -> Tuple[bool, str, Optional[np.ndarray]]:
    if state.mock_mode:
        state.mujoco_connected = True
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:, :, 1] = 50
        return True, log("Mock MuJoCo"), placeholder

    try:
        from reachy2_mujoco import ReachySDK
        state.reachy = ReachySDK("localhost")
        state.mujoco_connected = True
        return True, log("MuJoCo connected"), get_camera_frame("front_cam")
    except Exception as e:
        state.mujoco_connected = False
        return False, log(f"MuJoCo failed: {e}"), None


def check_cosmos() -> Tuple[bool, str]:
    if state.mock_mode:
        state.cosmos_available = True
        return True, log("Mock Cosmos")

    try:
        from cosmos_client import CosmosClient
        state.cosmos_client = CosmosClient(host="localhost", port=8100, timeout=30)
        if state.cosmos_client.is_available():
            state.cosmos_available = True
            return True, log("Cosmos available")
        state.cosmos_available = False
        return False, log("Cosmos not responding")
    except Exception as e:
        state.cosmos_available = False
        return False, log(f"Cosmos failed: {e}")


def load_groot(model_path: str) -> Tuple[bool, str]:
    if state.groot_loaded and state.policy is not None:
        return True, log("GR00T already loaded")

    if state.mock_mode:
        state.groot_loaded = True
        return True, log("Mock GR00T")

    if not model_path:
        return False, log("No model path")

    try:
        config_path = Path(__file__).parent.parent / "configs/reachy2_modality_config.py"
        if config_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("modality_config", config_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        state.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.REACHY2,
            model_path=model_path, device="cuda", strict=True,
        )
        state.groot_loaded = True
        state.model_path = model_path
        return True, log(f"GR00T loaded: {model_path}")
    except Exception as e:
        state.groot_loaded = False
        return False, log(f"GR00T failed: {e}")


def get_camera_frame(camera_name: str = "front_cam") -> Optional[np.ndarray]:
    if state.mock_mode:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 1] = 50
        return frame

    if not state.mujoco_connected or state.reachy is None:
        return None

    try:
        import rpyc
        frame_netref = state.reachy.render_camera(camera_name, width=640, height=480)
        frame = np.asarray(rpyc.classic.obtain(frame_netref), dtype=np.uint8)
        if len(frame.shape) == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        return frame[:, :, ::-1]
    except Exception as e:
        log(f"Camera error: {e}")
        return None


def get_connection_status() -> str:
    parts = [
        f"{'🟢' if state.mujoco_connected else '🔴'} MuJoCo",
        f"{'🟢' if state.cosmos_available else '🔴'} Cosmos",
        f"{'🟢' if state.groot_loaded else '🔴'} GR00T",
    ]
    return " | ".join(parts)


def transcribe_audio(audio_data) -> Tuple[str, str]:
    if audio_data is None:
        return "", log("No audio")

    if state.mock_mode:
        return "pick up the red cube and place it in the box", log("Mock transcription")

    try:
        if state.asr is None:
            from parakeet_asr import ParakeetASR
            state.asr = ParakeetASR(device="cuda")
            state.asr.load_model()

        sample_rate, audio_array = audio_data
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32) / 32768.0

        text = state.asr.transcribe_audio(audio_array, sample_rate)
        return text, log(f"Transcribed: \"{text}\"")
    except Exception as e:
        return "", log(f"Transcription failed: {e}")


def parse_command(text: str) -> dict:
    text_lower = text.lower()
    result = {"action": None, "object": None, "color": None, "location": None, "raw": text}

    if any(w in text_lower for w in ["pick", "grab", "grasp", "get"]):
        result["action"] = "pick"
    elif any(w in text_lower for w in ["place", "put", "drop"]):
        result["action"] = "place"
    elif "move" in text_lower:
        result["action"] = "move"

    for obj in ["cube", "box", "cylinder", "capsule", "block"]:
        if obj in text_lower:
            result["object"] = obj
            break

    for color in ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]:
        if color in text_lower:
            result["color"] = color
            break

    if "box" in text_lower and result["object"] != "box":
        result["location"] = "box"
    elif "left" in text_lower:
        result["location"] = "left"
    elif "right" in text_lower:
        result["location"] = "right"

    return result


def call_cosmos(image: np.ndarray, prompt: str) -> Tuple[str, str, str, str]:
    if state.mock_mode:
        scene = "red cube (center), white drop box (right)"
        plan = "1. Move above cube\n2. Grasp\n3. Lift\n4. Move to box\n5. Release"
        return scene, plan, prompt or "Pick up the red cube", log("Mock Cosmos")

    if not state.cosmos_available or state.cosmos_client is None:
        return "", "", prompt, log("Cosmos unavailable")

    try:
        result = state.cosmos_client.reason(image, prompt)
        return (result.get("scene_analysis", ""), result.get("task_plan", ""),
                result.get("gr00t_instruction", prompt), log("Cosmos done"))
    except Exception as e:
        return "", "", prompt, log(f"Cosmos error: {e}")


def create_controller():
    if state.mock_mode or state.controller is not None:
        return True
    if not state.groot_loaded or not state.mujoco_connected:
        return False

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from eval_closed_loop import ReachyGR00TController
        state.controller = ReachyGR00TController(
            policy=state.policy, reachy=state.reachy, arm=state.arm,
            control_freq=15.0, execute_horizon=4, clip_to_training_bounds=True,
        )
        return True
    except Exception as e:
        log(f"Controller error: {e}")
        return False


def reset_robot() -> str:
    if state.mock_mode:
        return log("Mock reset")
    if not state.mujoco_connected:
        return log("MuJoCo not connected")

    try:
        state.reachy.reset_scene()
        time.sleep(0.5)
        arm_ref = getattr(state.reachy, f"{state.arm[0]}_arm")
        for i, joint in enumerate([22, 0, 0, -110, 0, -3, 0]):
            [arm_ref.shoulder.pitch, arm_ref.shoulder.roll, arm_ref.elbow.yaw,
             arm_ref.elbow.pitch, arm_ref.wrist.roll, arm_ref.wrist.pitch,
             arm_ref.wrist.yaw][i].goal_position = joint
        arm_ref.gripper.open()
        state.reachy.send_goal_positions()
        time.sleep(1.0)
        return log("Robot reset")
    except Exception as e:
        return log(f"Reset failed: {e}")


def run_groot_execution(instruction: str, max_steps: int, progress_callback=None):
    state.is_executing = True
    state.should_stop = False
    state.current_step = 0

    if state.mock_mode:
        for step in range(min(max_steps, 50)):
            if state.should_stop:
                yield step, "Stopped", log(f"Stopped at {step}")
                break
            state.current_step = step
            time.sleep(0.1)
            if step % 10 == 0:
                yield step, f"Step {step}/{max_steps}", log(f"Step {step}")
        state.is_executing = False
        yield state.current_step, "Complete (mock)", log("Mock done")
        return

    if not create_controller():
        state.is_executing = False
        yield 0, "Failed", log("Controller failed")
        return

    try:
        state.controller.reset_episode(randomize=False)
        step = 0
        while step < max_steps and not state.should_stop:
            obs = state.controller.prepare_observation(instruction)
            action_pred, _ = state.policy.get_action(obs)
            arm_actions, gripper_actions = action_pred["arm_joints"][0], action_pred["gripper"][0]

            current_state = state.controller.get_joint_positions()
            for i in range(min(4, max_steps - step)):
                if state.should_stop:
                    break
                arm_action = state.controller.process_action(arm_actions[i], current_state)
                state.controller.execute_action(arm_action, gripper_actions[i, 0])
                current_state = arm_action
                step += 1
                state.current_step = step
                if step % 10 == 0:
                    yield step, f"Step {step}/{max_steps}", log(f"Step {step}")
            time.sleep(0.05)

        success = state.controller.check_success()
        state.is_executing = False
        yield step, "SUCCESS" if success else "Incomplete", log(f"Done: {'SUCCESS' if success else 'incomplete'}")
    except Exception as e:
        state.is_executing = False
        yield state.current_step, f"Error", log(f"Error: {e}")


def stop_execution() -> str:
    state.should_stop = True
    return log("Stop requested")


def run_pipeline(task_text: str, camera_choice: str, max_steps: int, arm_choice: str):
    state.arm = arm_choice.lower()
    log_text = log(f"Task: \"{task_text}\"")
    yield None, "", "", "", f"Step 0/{max_steps}", log_text

    frame = get_camera_frame(camera_choice)
    if frame is None:
        yield frame, "", "", "", "Error", log("No frame")
        return

    scene, plan, instruction, log_text = call_cosmos(frame, task_text)
    cosmos_output = f"**Scene:**\n{scene}\n\n**Plan:**\n{plan}\n\n**Instruction:**\n`{instruction}`"
    yield frame, cosmos_output, instruction, "", f"Step 0/{max_steps}", log_text

    for step, status, log_text in run_groot_execution(instruction, max_steps):
        yield frame, cosmos_output, instruction, status, f"Step {step}/{max_steps} - {status}", log_text


def create_ui():
    with gr.Blocks(title="NOVA") as demo:
        gr.Markdown("# NOVA - Neural Open Vision Actions")
        status_display = gr.Markdown(get_connection_status())

        with gr.Row():
            with gr.Column(scale=1):
                camera_image = gr.Image(label="Camera", type="numpy", height=400)
                with gr.Row():
                    camera_dropdown = gr.Dropdown(["front_cam", "workspace_cam"], value="front_cam", label="Camera", scale=2)
                    live_toggle = gr.Checkbox(label="Live", value=True, scale=1)
                    refresh_btn = gr.Button("Refresh", scale=1)
                camera_timer = gr.Timer(value=0.5, active=True)

            with gr.Column(scale=1):
                task_input = gr.Textbox(label="Task", placeholder="Pick up the red cube...", lines=2)
                voice_input = gr.Audio(sources=["microphone"], type="numpy", label="Voice")
                with gr.Row():
                    voice_btn = gr.Button("Transcribe", variant="secondary")
                    run_btn = gr.Button("Run", variant="primary")
                with gr.Row():
                    max_steps_slider = gr.Slider(50, 500, value=200, step=10, label="Max Steps")
                    arm_dropdown = gr.Dropdown(["Right", "Left"], value="Right", label="Arm")

        cosmos_output = gr.Markdown("*Run a task to see analysis*")

        with gr.Row():
            stop_btn = gr.Button("Stop", variant="stop")
            reset_btn = gr.Button("Reset")
            progress_display = gr.Textbox(label="Status", value="Ready", interactive=False, scale=2)

        log_output = gr.Textbox(label="Log", lines=8, max_lines=15, interactive=False)
        instruction_state = gr.State("")
        execution_status = gr.State("")

        def on_connect():
            _, log_text, frame = connect_mujoco()
            check_cosmos()
            if state.model_path:
                success, log_text = load_groot(state.model_path)
                print(f"GR00T load: success={success}")
            else:
                log_text = log("No model path provided")
            return frame, get_connection_status(), log_text

        camera_timer.tick(fn=lambda c: get_camera_frame(c), inputs=[camera_dropdown], outputs=[camera_image])
        live_toggle.change(fn=lambda l: gr.Timer(active=l), inputs=[live_toggle], outputs=[camera_timer])
        refresh_btn.click(fn=lambda c: get_camera_frame(c), inputs=[camera_dropdown], outputs=[camera_image])
        voice_btn.click(fn=transcribe_audio, inputs=[voice_input], outputs=[task_input, log_output])
        stop_btn.click(fn=stop_execution, outputs=[log_output])
        reset_btn.click(fn=lambda: (get_camera_frame("front_cam"), reset_robot()), outputs=[camera_image, log_output])
        run_btn.click(fn=run_pipeline, inputs=[task_input, camera_dropdown, max_steps_slider, arm_dropdown],
                      outputs=[camera_image, cosmos_output, instruction_state, execution_status, progress_display, log_output])
        demo.load(fn=on_connect, outputs=[camera_image, status_display, log_output])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    state.mock_mode = args.mock
    state.model_path = args.model_path
    print(f"NOVA {'(mock)' if args.mock else '(live)'}")
    print(f"Model path: '{state.model_path}' (empty={not state.model_path})")

    demo = create_ui()
    demo.launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()