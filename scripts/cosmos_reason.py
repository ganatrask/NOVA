#!/usr/bin/env python3
"""Cosmos Reason 2 for scene understanding and task planning."""

import argparse
import re
from pathlib import Path
from typing import Optional, Union, List, Dict
import warnings

warnings.filterwarnings("ignore")


class CosmosReason:
    MODELS = {"2b": "nvidia/Cosmos-Reason2-2B", "8b": "nvidia/Cosmos-Reason2-8B"}

    def __init__(self, model_size: str = "2b", device: str = "cuda", load_in_4bit: bool = False):
        self.model_size = model_size
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return

        import torch
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        model_id = self.MODELS[self.model_size]
        kwargs = {"trust_remote_code": True}

        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["dtype"] = torch.float16

        if self.device == "cuda":
            kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        self._loaded = True

    def _prepare_image(self, image):
        from PIL import Image
        import numpy as np

        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")

    def _generate(self, prompt: str, image=None, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        import torch
        self.load_model()

        if image is not None:
            pil_image = self._prepare_image(image)
            messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
        else:
            messages = [{"role": "user", "content": prompt}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                          do_sample=True, pad_token_id=self.processor.tokenizer.eos_token_id)

        return self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _parse_think_answer(self, response: str) -> Dict[str, str]:
        think = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        return {
            "thinking": think.group(1).strip() if think else "",
            "answer": answer.group(1).strip() if answer else response.strip(),
            "raw": response,
        }

    def analyze_scene(self, image) -> Dict:
        prompt = """Analyze this robot workspace. List objects as:
- Object: [type], Color: [color], Position: [position]
Use <think></think> and <answer></answer> tags."""

        response = self._generate(prompt, image)
        parsed = self._parse_think_answer(response)

        objects = []
        for line in parsed["answer"].split("\n"):
            if line.strip().startswith("- Object"):
                parts = {}
                for part in line[2:].split(","):
                    if ":" in part:
                        k, v = part.split(":", 1)
                        parts[k.strip().lower()] = v.strip()
                if parts:
                    objects.append(parts)

        return {"objects": objects, "description": parsed["answer"],
                "thinking": parsed["thinking"], "raw": response}

    def plan_task(self, task_description: str, image=None) -> Dict:
        prompt = f"""Task: "{task_description}"
Provide:
1. Target object: [description]
2. Target location: [where]
3. Steps: - Step N: [action]
4. GR00T instruction: [single sentence]
Use <think></think> and <answer></answer> tags."""

        response = self._generate(prompt, image)
        parsed = self._parse_think_answer(response)

        target_object, target_location, gr00t_instruction = "", "", ""
        steps = []

        for line in parsed["answer"].split("\n"):
            l = line.strip().lower()
            if l.startswith("target object:"):
                target_object = line.split(":", 1)[1].strip()
            elif l.startswith("target location:"):
                target_location = line.split(":", 1)[1].strip()
            elif line.strip().startswith("- Step"):
                steps.append(line.strip()[2:].strip())
            elif l.startswith("gr00t instruction:"):
                gr00t_instruction = line.split(":", 1)[1].strip()

        return {
            "target_object": target_object, "target_location": target_location,
            "steps": steps, "gr00t_instruction": gr00t_instruction or task_description,
            "thinking": parsed["thinking"], "raw": response,
        }

    def verify_success(self, task_description: str, before_image, after_image) -> Dict:
        prompt = f"""Task: "{task_description}"
Was it completed? Answer:
- Success: [yes/no]
- Confidence: [0-100]%
- Explanation: [brief]
Use <think></think> and <answer></answer> tags."""

        response = self._generate(prompt, after_image)
        parsed = self._parse_think_answer(response)

        success, confidence, explanation = False, 0.5, ""
        for line in parsed["answer"].split("\n"):
            l = line.strip().lower()
            if l.startswith("success:"):
                success = "yes" in l
            elif l.startswith("confidence:"):
                m = re.search(r"(\d+)", l)
                if m:
                    confidence = int(m.group(1)) / 100
            elif l.startswith("explanation:"):
                explanation = line.split(":", 1)[1].strip()

        return {"success": success, "confidence": confidence, "explanation": explanation,
                "thinking": parsed["thinking"], "raw": response}

    def generate_instruction(self, scene_description: str = None, objects: List[str] = None, image=None) -> str:
        if image:
            prompt = "Generate a pick-and-place instruction: <answer>[instruction]</answer>"
        else:
            prompt = f"Given {', '.join(objects) if objects else 'objects'}, generate instruction: <answer>[instruction]</answer>"
        return self._parse_think_answer(self._generate(prompt, image, max_new_tokens=100))["answer"]


class MockCosmosReason:
    def __init__(self, **kwargs): pass
    def load_model(self): pass

    def analyze_scene(self, image) -> Dict:
        return {"objects": [{"object": "cube", "color": "red", "position": "center"}],
                "description": "Red cube center, white box right", "thinking": "", "raw": ""}

    def plan_task(self, task_description: str, image=None) -> Dict:
        return {"target_object": "red cube", "target_location": "box",
                "steps": ["Move above", "Grasp", "Lift", "Move to box", "Release"],
                "gr00t_instruction": task_description, "thinking": "", "raw": ""}

    def verify_success(self, task_description: str, before_image, after_image) -> Dict:
        return {"success": True, "confidence": 0.8, "explanation": "Mock", "thinking": "", "raw": ""}

    def generate_instruction(self, **kwargs) -> str:
        return "Pick up the red cube and place it in the box"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str)
    parser.add_argument("--task", "-t", type=str)
    parser.add_argument("--analyze", "-a", action="store_true")
    parser.add_argument("--model-size", default="2b", choices=["2b", "8b"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    cosmos = MockCosmosReason() if args.mock else CosmosReason(
        model_size=args.model_size, device=args.device, load_in_4bit=args.quantize)

    if args.image and args.analyze:
        result = cosmos.analyze_scene(args.image)
        print(f"Objects: {result['objects']}")

    if args.task:
        result = cosmos.plan_task(args.task, args.image)
        print(f"Target: {result['target_object']} -> {result['target_location']}")
        print(f"Instruction: {result['gr00t_instruction']}")

    if not args.image and not args.task:
        parser.print_help()


if __name__ == "__main__":
    main()
