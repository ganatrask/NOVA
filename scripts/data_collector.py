#!/usr/bin/env python3
"""LeRobot v2.1 dataset collector for Reachy 2 pick-and-place tasks."""

import time
import json
import signal
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from threading import Thread, Event
import cv2

from reachy2_mujoco import ReachySDK

_dataset_writer = None
_shutdown_requested = False


def signal_handler(signum, frame):
    global _shutdown_requested, _dataset_writer
    if _shutdown_requested:
        sys.exit(1)
    _shutdown_requested = True
    print("\nSaving partial dataset...")
    if _dataset_writer:
        try:
            _dataset_writer.finalize()
        except Exception as e:
            print(f"Save error: {e}")
    sys.exit(0)


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    return obj


# Config
RECORD_FPS = 15
TORSO_HEIGHT = 0.97
RIGHT_ARM_WORKSPACE = {'x_range': (0.22, 0.34), 'y_range': (-0.28, -0.18)}
LEFT_ARM_WORKSPACE = {'x_range': (0.22, 0.42), 'y_range': (0.08, 0.30)}
DROP_BOX_CENTER = np.array([0.22, 0.01, 0.625])
DROP_BOX_HALF_SIZE = 0.09
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 480
EXTERNAL_CAMERAS = ['front_cam', 'workspace_cam']

OBJECTS = {
    'cube': {'type': 'box', 'size': [0.020, 0.020, 0.020], 'half_height': 0.020},
    'rect_box': {'type': 'box', 'size': [0.025, 0.020, 0.018], 'half_height': 0.018},
    'cylinder': {'type': 'cylinder', 'size': [0.018, 0.020], 'half_height': 0.020},
    'capsule': {'type': 'capsule', 'size': [0.018, 0.012], 'half_height': 0.030},
}

COLORS = {
    'red': [1.0, 0.0, 0.0, 1.0],
    'green': [0.0, 1.0, 0.0, 1.0],
    'blue': [0.0, 0.0, 1.0, 1.0],
    'yellow': [1.0, 1.0, 0.0, 1.0],
    'cyan': [0.0, 1.0, 1.0, 1.0],
    'magenta': [1.0, 0.0, 1.0, 1.0],
    'orange': [1.0, 0.5, 0.0, 1.0],
    'purple': [0.5, 0.0, 1.0, 1.0],
}


class LeRobotDatasetWriter:
    def __init__(self, output_dir: str, robot_type: str = "reachy2",
                 fps: int = RECORD_FPS, cameras: list = None):
        self.output_dir = Path(output_dir)
        self.robot_type = robot_type
        self.fps = fps
        self.cameras = cameras or EXTERNAL_CAMERAS

        self.meta_dir = self.output_dir / "meta"
        self.data_dir = self.output_dir / "data" / "chunk-000"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.video_dirs = {}
        for cam in self.cameras:
            vdir = self.output_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
            vdir.mkdir(parents=True, exist_ok=True)
            self.video_dirs[cam] = vdir

        self.episodes = []
        self.tasks = {}
        self.total_frames = 0
        self.global_index = 0

    def _get_or_create_task_index(self, task_description: str) -> int:
        if task_description not in self.tasks:
            self.tasks[task_description] = len(self.tasks)
        return self.tasks[task_description]

    def save_episode(self, episode_id: int, frames: list, camera_images: dict = None,
                     success: bool = False, arm_name: str = "right", object_metadata: dict = None):
        if not frames:
            return

        obj_name = object_metadata.get('object_name', 'cube') if object_metadata else 'cube'
        obj_color = object_metadata.get('object_color', 'red') if object_metadata else 'red'
        task_description = f"Pick up the {obj_color} {obj_name} and place it in the box"
        task_index = self._get_or_create_task_index(task_description)

        rows = []
        num_frames = len(frames)

        for i, frame in enumerate(frames):
            state = frame['observation.state']
            if i < num_frames - 1:
                action_joints = frames[i + 1]['observation.state']
                action_gripper = frames[i + 1]['gripper_state']
            else:
                action_joints = state
                action_gripper = frame['gripper_state']

            rows.append({
                'index': self.global_index,
                'episode_index': episode_id,
                'frame_index': i,
                'timestamp': frame['timestamp'],
                'task_index': task_index,
                'next.done': i == num_frames - 1,
                'observation.state': state,
                'action': action_joints + [action_gripper],
            })
            self.global_index += 1

        df = pd.DataFrame(rows)
        parquet_path = self.data_dir / f"episode_{episode_id:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

        if camera_images:
            for cam, images in camera_images.items():
                if images and cam in self.video_dirs:
                    self._save_video(episode_id, images, cam)

        episode_entry = {
            'episode_index': episode_id, 'tasks': [task_index], 'length': num_frames,
            'success': success, 'arm': arm_name, 'task_description': task_description,
        }
        if object_metadata:
            episode_entry.update({
                'object_name': object_metadata.get('object_name', 'cube'),
                'object_color': object_metadata.get('object_color', 'red'),
                'object_color_rgba': object_metadata.get('object_color_rgba', COLORS['red']),
                'object_initial': object_metadata.get('object_initial', [0, 0, 0]),
                'object_final': object_metadata.get('object_final', [0, 0, 0]),
                'object_displacement': object_metadata.get('object_displacement', 0.0),
            })

        self.episodes.append(episode_entry)
        self.total_frames += num_frames
        print(f"  Episode {episode_id}: {num_frames} frames")

    def _save_video(self, episode_id: int, images: list, cam_name: str):
        video_path = self.video_dirs[cam_name] / f"episode_{episode_id:06d}.mp4"
        h, w = images[0].shape[:2]

        writer = None
        for codec in ['avc1', 'h264', 'x264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(video_path), fourcc, RECORD_FPS, (w, h))
            if writer.isOpened():
                break
            writer.release()
            writer = None

        if not writer:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), RECORD_FPS, (w, h))

        for img in images:
            if img.shape[-1] == 3:
                writer.write(img)
        writer.release()

    def _compute_and_save_stats(self):
        all_states, all_actions = [], []

        for pq in sorted(self.data_dir.glob("episode_*.parquet")):
            df = pd.read_parquet(pq)
            if 'observation.state' in df.columns:
                all_states.append(np.stack(df['observation.state'].values))
            if 'action' in df.columns:
                all_actions.append(np.stack(df['action'].values))

        stats = {}
        if all_states:
            s = np.concatenate(all_states, axis=0)
            stats["observation.state"] = {
                "mean": s.mean(axis=0).tolist(), "std": s.std(axis=0).tolist(),
                "min": s.min(axis=0).tolist(), "max": s.max(axis=0).tolist()
            }
        if all_actions:
            a = np.concatenate(all_actions, axis=0)
            stats["action"] = {
                "mean": a.mean(axis=0).tolist(), "std": a.std(axis=0).tolist(),
                "min": a.min(axis=0).tolist(), "max": a.max(axis=0).tolist()
            }

        with open(self.meta_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

    def finalize(self):
        joint_names = ["shoulder_pitch", "shoulder_roll", "elbow_yaw",
                       "elbow_pitch", "wrist_roll", "wrist_pitch", "wrist_yaw"]
        features = {
            "observation.state": {"dtype": "float32", "shape": [7], "names": joint_names},
            "action": {"dtype": "float32", "shape": [8], "names": joint_names + ["gripper"]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]}
        }
        for cam in self.cameras:
            features[f"observation.images.{cam}"] = {
                "dtype": "video", "shape": [VIDEO_HEIGHT, VIDEO_WIDTH, 3],
                "video_info": {"video.fps": RECORD_FPS, "video.codec": "mp4v", "video.pix_fmt": "bgr24"}
            }

        info = {
            "codebase_version": "v2.1", "robot_type": self.robot_type, "fps": self.fps,
            "total_episodes": len(self.episodes), "total_frames": self.total_frames,
            "total_tasks": len(self.tasks), "total_chunks": 1, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/observation.images.{camera_name}/episode_{episode_index:06d}.mp4",
            "cameras": self.cameras, "features": features,
        }
        with open(self.meta_dir / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        self._compute_and_save_stats()

        with open(self.meta_dir / "tasks.jsonl", 'w') as f:
            for desc, idx in sorted(self.tasks.items(), key=lambda x: x[1]):
                f.write(json.dumps({"task_index": idx, "task": desc}) + '\n')

        with open(self.meta_dir / "episodes.jsonl", 'w') as f:
            for ep in self.episodes:
                entry = convert_numpy_types({
                    "episode_index": ep['episode_index'], "tasks": ep['tasks'], "length": ep['length']
                })
                f.write(json.dumps(entry) + '\n')

        success_count = sum(1 for ep in self.episodes if ep.get('success', False))
        object_stats, color_stats = {}, {}
        for ep in self.episodes:
            obj, col = ep.get('object_name', 'cube'), ep.get('object_color', 'red')
            for stats, key in [(object_stats, obj), (color_stats, col)]:
                if key not in stats:
                    stats[key] = {'count': 0, 'success': 0}
                stats[key]['count'] += 1
                if ep.get('success'):
                    stats[key]['success'] += 1

        collection_info = convert_numpy_types({
            "total_episodes": len(self.episodes), "total_frames": self.total_frames,
            "total_tasks": len(self.tasks), "success_count": success_count,
            "success_rate": success_count / len(self.episodes) if self.episodes else 0,
            "fps": self.fps, "cameras": self.cameras, "timestamp": datetime.now().isoformat(),
            "object_stats": object_stats, "color_stats": color_stats,
            "tasks": dict(self.tasks), "episodes": self.episodes,
        })
        with open(self.meta_dir / "collection_info.json", 'w') as f:
            json.dump(collection_info, f, indent=2)

        print(f"Dataset saved: {self.output_dir} ({len(self.episodes)} episodes, {self.total_frames} frames)")


class TrajectoryRecorder:
    def __init__(self, reachy, arm_name: str = 'right', cameras: list = None, fps: float = RECORD_FPS):
        self.reachy = reachy
        self.arm_name = arm_name
        self.cameras = cameras or EXTERNAL_CAMERAS
        self.interval = 1.0 / fps
        self.arm = reachy.r_arm if arm_name == 'right' else reachy.l_arm
        self.frames = []
        self.camera_images = {cam: [] for cam in self.cameras}
        self.recording = False
        self.stop_event = Event()
        self.record_thread = None

    def get_arm_state(self) -> list:
        return [float(p) for p in self.arm.get_present_positions()]

    def get_gripper_state(self) -> float:
        try:
            pos = float(self.arm.gripper.present_position)
            return max(0.0, min(1.0, (pos + 20) / 70.0))
        except Exception:
            return 0.5

    def get_external_camera_frame(self, camera_name: str) -> np.ndarray:
        try:
            return np.array(self.reachy.render_camera(camera_name, width=VIDEO_WIDTH, height=VIDEO_HEIGHT))
        except Exception:
            return np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

    def _record_loop(self):
        start_time = time.time()
        frame_idx = 0

        while not self.stop_event.is_set():
            frame_start = time.time()
            self.frames.append({
                'frame_index': frame_idx,
                'timestamp': time.time() - start_time,
                'observation.state': self.get_arm_state(),
                'gripper_state': self.get_gripper_state(),
            })
            for cam in self.cameras:
                img = self.get_external_camera_frame(cam)
                if img is not None:
                    self.camera_images[cam].append(img)
            frame_idx += 1
            elapsed = time.time() - frame_start
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)

    def start(self):
        if self.recording:
            return
        self.frames = []
        self.camera_images = {cam: [] for cam in self.cameras}
        self.stop_event.clear()
        self.recording = True
        self.record_thread = Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

    def stop(self) -> tuple:
        if not self.recording:
            return [], {}
        self.stop_event.set()
        self.recording = False
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        return self.frames, self.camera_images


def make_pose_matrix(position: np.ndarray, euler_angles_deg: np.ndarray) -> np.ndarray:
    rot = R.from_euler('xyz', euler_angles_deg, degrees=True).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = position
    return pose


def world_to_torso(world_pos: np.ndarray) -> np.ndarray:
    return np.array([world_pos[0], world_pos[1], world_pos[2] - TORSO_HEIGHT])


def quat_to_yaw(quat: np.ndarray) -> float:
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz', degrees=True)[2]


def get_object_pose(reachy, object_name: str = 'cube') -> tuple:
    try:
        obj_qpos = reachy.get_object_pose(object_name)
        if obj_qpos is not None:
            obj_qpos = np.array([float(x) for x in obj_qpos])
            return obj_qpos[:3], obj_qpos[3:7]
    except Exception:
        pass
    try:
        cube_qpos = np.array([float(x) for x in reachy.get_cube_pose()])
        return cube_qpos[:3], cube_qpos[3:7]
    except Exception:
        return np.array([0.32, -0.15, 0.65]), np.array([1, 0, 0, 0])


def set_object_pose(reachy, object_name: str, position, orientation=None):
    orientation = orientation or [1, 0, 0, 0]
    try:
        return reachy.set_object_pose(object_name, list(position), list(orientation))
    except Exception:
        try:
            return reachy.set_cube_pose(list(position), list(orientation))
        except Exception:
            return False


def is_in_drop_box(x, y, margin=0.06):
    return (abs(x - DROP_BOX_CENTER[0]) < DROP_BOX_HALF_SIZE + margin and
            abs(y - DROP_BOX_CENTER[1]) < DROP_BOX_HALF_SIZE + margin)


def randomize_object(reachy, object_name: str = None, color_name: str = None,
                     arm_name: str = None, seed: int = None) -> dict:
    if seed is not None:
        np.random.seed(seed)

    if object_name is None or object_name not in OBJECTS:
        object_name = str(np.random.choice(list(OBJECTS.keys()))) if object_name is None else 'cube'
    if color_name is None or color_name not in COLORS:
        color_name = str(np.random.choice(list(COLORS.keys()))) if color_name is None else 'red'

    color_rgba = COLORS[color_name]
    arm_name = arm_name or np.random.choice(['right', 'left'])
    workspace = RIGHT_ARM_WORKSPACE if arm_name == 'right' else LEFT_ARM_WORKSPACE

    for _ in range(20):
        x = np.random.uniform(*workspace['x_range'])
        y = np.random.uniform(*workspace['y_range'])
        if not is_in_drop_box(x, y):
            break

    position = [x, y, 0.65]
    yaw = np.random.uniform(-15, 15)
    quat_xyzw = R.from_euler('z', yaw, degrees=True).as_quat()
    orientation = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]

    try:
        reachy.set_active_object(object_name, position, color_rgba)
    except Exception:
        set_object_pose(reachy, object_name, position, orientation)

    return {
        'object_name': object_name, 'color_name': color_name, 'color_rgba': color_rgba,
        'position': position, 'orientation': orientation, 'arm': arm_name,
        'half_height': OBJECTS[object_name]['half_height'],
    }


def collect_episode(reachy, episode_id: int, arm_name: str = 'right',
                    randomize: bool = True, cameras: list = None,
                    object_name: str = None, color_name: str = None,
                    randomize_object_type: bool = False,
                    randomize_color: bool = False):
    """
    Collect a single episode with trajectory recording.

    Returns:
        (frames, camera_images, success, arm_name, object_metadata)
    """
    print(f"\n{'='*60}")
    print(f"EPISODE {episode_id} ({arm_name.upper()} arm)")
    print("="*60)

    cameras = cameras or EXTERNAL_CAMERAS
    arm = reachy.r_arm if arm_name == 'right' else reachy.l_arm
    recorder = TrajectoryRecorder(reachy, arm_name, cameras)

    # Safe position
    print("Moving to safe position...")
    safe_joints = [32, 0, 11, -109, -59, -12, -10]
    for i, joint in enumerate(['shoulder.pitch', 'shoulder.roll', 'elbow.yaw',
                               'elbow.pitch', 'wrist.roll', 'wrist.pitch', 'wrist.yaw']):
        parts = joint.split('.')
        getattr(getattr(arm, parts[0]), parts[1]).goal_position = safe_joints[i]
    arm.gripper.open()
    reachy.send_goal_positions()
    time.sleep(1.5)

    # Determine object and color
    obj = object_name if object_name else (None if randomize_object_type else 'cube')
    col = color_name if color_name else (None if randomize_color else 'red')

    # Reset and randomize object
    object_info = None
    if randomize or randomize_object_type or randomize_color:
        object_info = randomize_object(reachy, object_name=obj, color_name=col,
                                        arm_name=arm_name, seed=None)  # No seed for true randomization
        time.sleep(0.5)
    else:
        # Reset to default position
        set_object_pose(reachy, 'cube', [0.32, -0.15, 0.65], [1, 0, 0, 0])
        object_info = {
            'object_name': 'cube',
            'color_name': 'red',
            'color_rgba': COLORS['red'],
            'position': [0.32, -0.15, 0.65],
        }
        time.sleep(0.3)

    # Get object pose
    active_object = object_info.get('object_name', 'cube')
    obj_world, obj_quat = get_object_pose(reachy, active_object)
    obj_torso = world_to_torso(obj_world)
    initial_obj_pos = obj_world.copy()
    object_half_height = OBJECTS.get(active_object, OBJECTS['cube'])['half_height']
    print(f"  Object: {active_object} at {obj_world.round(3)}")

    # Orientations
    gripper_yaw = min(15, max(-15, quat_to_yaw(obj_quat)))
    grasp_orient = [0, -90, gripper_yaw]
    place_orient = [0, -90, 0]

    # Place position and ready joints
    if arm_name == 'right':
        place_torso = np.array([0.22, -0.05, 0.72 - TORSO_HEIGHT])
        ready_joints = [15, -10, 0, -90, 0, -15, 0]
    else:
        place_torso = np.array([0.22, 0.05, 0.72 - TORSO_HEIGHT])
        ready_joints = [15, 10, 0, -90, 0, -15, 0]

    # Start recording
    print("Recording trajectory...")
    recorder.start()

    try:
        # Execute pick and place sequence
        def set_joints(joints):
            arm.shoulder.pitch.goal_position = joints[0]
            arm.shoulder.roll.goal_position = joints[1]
            arm.elbow.yaw.goal_position = joints[2]
            arm.elbow.pitch.goal_position = joints[3]
            arm.wrist.roll.goal_position = joints[4]
            arm.wrist.pitch.goal_position = joints[5]
            arm.wrist.yaw.goal_position = joints[6]

        # 1. Ready
        set_joints(ready_joints)
        arm.gripper.open()
        reachy.send_goal_positions()
        time.sleep(1.0)

        # 2. Above object
        obj_top_z = obj_torso[2] + object_half_height
        above = np.array([obj_torso[0] - 0.02, obj_torso[1], obj_top_z + 0.10])
        arm.goto(make_pose_matrix(above, np.array(grasp_orient)), duration=1.5)
        time.sleep(1.8)

        # 3. Pre-grasp
        pre_grasp = np.array([obj_torso[0] - 0.02, obj_torso[1], obj_top_z + 0.12])
        arm.goto(make_pose_matrix(pre_grasp, np.array(grasp_orient)), duration=1.5)
        time.sleep(1.8)

        # 4. Grasp position
        grasp = np.array([obj_torso[0] - 0.02, obj_torso[1], obj_top_z - 0.06])
        arm.goto(make_pose_matrix(grasp, np.array(grasp_orient)), duration=2.0)
        time.sleep(2.3)

        # 5. Close gripper
        arm.gripper.close()
        reachy.send_goal_positions()
        time.sleep(1.5)

        # 6. Lift
        lift = np.array([obj_torso[0] - 0.02, obj_torso[1], obj_top_z + 0.10])
        arm.goto(make_pose_matrix(lift, np.array(grasp_orient)), duration=1.5)
        time.sleep(1.8)

        # 7. To place (high)
        place_high = np.array([place_torso[0], place_torso[1], place_torso[2] + 0.08])
        arm.goto(make_pose_matrix(place_high, np.array(place_orient)), duration=1.5)
        time.sleep(1.8)

        # 8. Lower
        arm.goto(make_pose_matrix(place_torso, np.array(place_orient)), duration=1.5)
        time.sleep(1.8)

        # 9. Release
        arm.gripper.open()
        reachy.send_goal_positions()
        time.sleep(0.5)

        # 10. Retreat
        arm.goto(make_pose_matrix(place_high, np.array(place_orient)), duration=1.5)
        time.sleep(1.8)

        # Back to ready
        set_joints(ready_joints)
        reachy.send_goal_positions()
        time.sleep(1.0)

    finally:
        frames, camera_images = recorder.stop()

    # Check success
    time.sleep(0.5)
    final_obj_pos, _ = get_object_pose(reachy, active_object)
    displacement = np.linalg.norm(final_obj_pos[:2] - initial_obj_pos[:2])

    in_box_x = abs(final_obj_pos[0] - DROP_BOX_CENTER[0]) < DROP_BOX_HALF_SIZE
    in_box_y = abs(final_obj_pos[1] - DROP_BOX_CENTER[1]) < DROP_BOX_HALF_SIZE
    success = in_box_x and in_box_y and final_obj_pos[2] > 0.5

    status = "SUCCESS" if success else ("PARTIAL" if displacement > 0.05 else "FAILED")
    print(f"  Result: {status} (displacement: {displacement:.3f}m)")

    # Build object metadata
    object_metadata = {
        'object_name': active_object,
        'object_color': object_info.get('color_name', 'red'),
        'object_color_rgba': object_info.get('color_rgba', COLORS['red']),
        'object_initial': initial_obj_pos.tolist(),
        'object_final': final_obj_pos.tolist(),
        'object_displacement': float(displacement),
    }

    return frames, camera_images, success, arm_name, object_metadata


def collect_dataset(num_episodes: int, output_dir: str, arm: str = 'both',
                    randomize: bool = True, cameras: list = None,
                    only_successful: bool = False, randomize_object: bool = False,
                    object_name: str = None, randomize_color: bool = False,
                    color_name: str = None, seed: int = None, start_episode: int = 0):
    global _dataset_writer, _shutdown_requested
    _shutdown_requested = False
    cameras = cameras or EXTERNAL_CAMERAS

    if seed is not None:
        np.random.seed(seed)

    print(f"Collecting {num_episodes} episodes to {output_dir}")
    signal.signal(signal.SIGINT, signal_handler)

    reachy = ReachySDK("localhost")
    writer = LeRobotDatasetWriter(output_dir, cameras=cameras, fps=RECORD_FPS)
    _dataset_writer = writer

    collected, success_count, attempt = start_episode, 0, start_episode

    if seed is not None and start_episode > 0:
        for _ in range(start_episode):
            np.random.choice(list(OBJECTS.keys()))
            np.random.choice(list(COLORS.keys()))
            np.random.choice(['right', 'left'])
            np.random.uniform(0, 1, size=5)

    while collected < num_episodes and not _shutdown_requested:
        arm_name = ('right' if attempt % 2 == 0 else 'left') if arm == 'both' else arm

        # Collect episode
        frames, camera_images, success, arm_name, object_metadata = collect_episode(
            reachy, attempt, arm_name, randomize, cameras,
            object_name=object_name, color_name=color_name,
            randomize_object_type=randomize_object, randomize_color=randomize_color
        )

        attempt += 1

        # Skip failed if only_successful
        if only_successful and not success:
            print(f"  Skipping failed episode (attempt {attempt})")
            continue

        # Save episode
        writer.save_episode(collected, frames, camera_images, success, arm_name, object_metadata)

        if success:
            success_count += 1
        collected += 1

        print(f"  Progress: {collected}/{num_episodes} episodes")

    writer.finalize()
    print(f"Done: {collected} episodes, {success_count} successful ({100*success_count/max(1,collected):.0f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--output', default='reachy2_dataset')
    parser.add_argument('--arm', choices=['right', 'left', 'both'], default='both')
    parser.add_argument('--no-randomize', action='store_true')
    parser.add_argument('--only-successful', action='store_true')
    parser.add_argument('--randomize-object', action='store_true')
    parser.add_argument('--object', choices=list(OBJECTS.keys()))
    parser.add_argument('--randomize-color', action='store_true')
    parser.add_argument('--color', choices=list(COLORS.keys()))
    parser.add_argument('--cameras', nargs='+', default=EXTERNAL_CAMERAS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--start-episode', type=int, default=0)
    args = parser.parse_args()

    collect_dataset(
        num_episodes=args.episodes, output_dir=args.output, arm=args.arm,
        randomize=not args.no_randomize, cameras=args.cameras,
        only_successful=args.only_successful, randomize_object=args.randomize_object,
        object_name=args.object, randomize_color=args.randomize_color,
        color_name=args.color, seed=args.seed, start_episode=args.start_episode,
    )
