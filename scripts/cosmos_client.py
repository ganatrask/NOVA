#!/usr/bin/env python3
"""Cosmos Reason 2 Client for the main robot pipeline."""

import argparse
import base64
import io
import requests
from typing import Optional, Union
import numpy as np
from PIL import Image


class CosmosClient:
    def __init__(self, host: str = "localhost", port: int = 8100, timeout: int = 60):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def health(self) -> dict:
        response = requests.get(f"{self.base_url}/health", timeout=5)
        response.raise_for_status()
        return response.json()

    def is_available(self) -> bool:
        try:
            health = self.health()
            return health.get("status") == "healthy" and health.get("model_loaded")
        except Exception:
            return False

    def reason(self, image: Union[np.ndarray, Image.Image, str], command: Optional[str] = None) -> dict:
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR to RGB
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        data = {"image_base64": image_base64}
        if command:
            data["command"] = command

        response = requests.post(f"{self.base_url}/reason", json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_instruction(self, image: Union[np.ndarray, Image.Image, str], command: Optional[str] = None) -> str:
        return self.reason(image, command).get("gr00t_instruction", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--command", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    client = CosmosClient(host=args.host, port=args.port)

    if not client.is_available():
        print("Cosmos server not available")
        return

    result = client.reason(args.image, args.command)
    print(f"Scene: {result.get('scene_analysis', 'N/A')}")
    print(f"Plan: {result.get('task_plan', 'N/A')}")
    print(f"Instruction: {result.get('gr00t_instruction', 'N/A')}")


if __name__ == "__main__":
    main()
