#!/usr/bin/env python3
"""Cosmos Reason 2 HTTP Server - avoids transformers version conflicts with GR00T."""

import argparse
import base64
import io
import os
import re
import sys
from pathlib import Path

GROOT_PATH = Path(__file__).parent.parent / "Isaac-GR00T"
sys.path.insert(0, str(GROOT_PATH))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from flask import Flask, request, jsonify

model = None
processor = None
device = None
app = Flask(__name__)


def load_model(model_name: str = "nvidia/Cosmos-Reason2-2B"):
    global model, processor, device

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Cosmos Reason 2 on {device}...")

    from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration

    processor = Qwen3VLProcessor.from_pretrained(model_name, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )

    if device == "cpu":
        model = model.to(device)

    print("Cosmos Reason 2 loaded!")


def analyze_scene(image: Image.Image, user_command: str = None) -> dict:
    global model, processor, device

    if user_command:
        prompt_text = f"""Analyze this robot workspace image and help plan a manipulation task.

User command: "{user_command}"

Please provide:
1. <scene_analysis>: Describe objects visible, their positions, and colors
2. <task_plan>: Step-by-step plan to accomplish the user's command
3. <gr00t_instruction>: A single clear instruction for the robot policy

Think step by step before answering."""
    else:
        prompt_text = """Analyze this robot workspace image.

Please provide:
1. <scene_analysis>: Describe all objects visible, their positions, colors, and any obstacles
2. <available_tasks>: List possible manipulation tasks the robot could perform
3. <gr00t_instruction>: The most useful single task instruction for the robot

Think step by step before answering."""

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    result = {"raw_response": response, "scene_analysis": "", "task_plan": "", "gr00t_instruction": ""}

    scene_match = re.search(r'<scene_analysis>(.*?)</scene_analysis>', response, re.DOTALL)
    if scene_match:
        result["scene_analysis"] = scene_match.group(1).strip()

    task_match = re.search(r'<task_plan>(.*?)</task_plan>', response, re.DOTALL)
    if task_match:
        result["task_plan"] = task_match.group(1).strip()

    instruction_match = re.search(r'<gr00t_instruction>(.*?)</gr00t_instruction>', response, re.DOTALL)
    if instruction_match:
        result["gr00t_instruction"] = instruction_match.group(1).strip()

    if not result["gr00t_instruction"]:
        quote_match = re.search(r'"([^"]+(?:pick|place|move|grasp|put)[^"]*)"', response, re.IGNORECASE)
        if quote_match:
            result["gr00t_instruction"] = quote_match.group(1)
        else:
            for s in reversed(response.split('.')):
                s = s.strip()
                if any(word in s.lower() for word in ['pick', 'place', 'move', 'grasp', 'put']):
                    result["gr00t_instruction"] = s
                    break

    return result


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None, "device": str(device) if device else None})


@app.route('/reason', methods=['POST'])
def reason():
    try:
        if 'image' in request.files:
            image = Image.open(request.files['image'].stream).convert('RGB')
        elif 'image_base64' in request.form:
            image = Image.open(io.BytesIO(base64.b64decode(request.form['image_base64']))).convert('RGB')
        elif request.is_json and 'image_base64' in request.json:
            image = Image.open(io.BytesIO(base64.b64decode(request.json['image_base64']))).convert('RGB')
        else:
            return jsonify({"error": "No image provided"}), 400

        command = request.form.get('command') or (request.json.get('command') if request.is_json else None)
        return jsonify(analyze_scene(image, command))

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason2-2B")
    args = parser.parse_args()

    load_model(args.model)
    print(f"\nCosmos server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
