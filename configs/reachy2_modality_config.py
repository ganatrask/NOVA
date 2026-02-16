"""Reachy 2 modality config for GR00T N1.6 fine-tuning."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

register_modality_config(
    config={
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front_cam"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["arm_joints"]),
        "action": ModalityConfig(
            delta_indices=list(range(0, 16)),
            modality_keys=["arm_joints", "gripper"],
            action_configs=[
                ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF,
                             format=ActionFormat.DEFAULT, state_key="arm_joints"),
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                             format=ActionFormat.DEFAULT),
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    },
    embodiment_tag=EmbodimentTag.REACHY2,
)
