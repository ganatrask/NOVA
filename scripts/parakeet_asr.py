#!/usr/bin/env python3
"""Parakeet ASR for voice-controlled robot commands."""

import argparse
import os
from pathlib import Path
from typing import Optional, Union
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)


class ParakeetASR:
    MODELS = {
        "parakeet-ctc-0.6b": "nvidia/parakeet-ctc-0.6b",
        "parakeet-ctc-1.1b": "nvidia/parakeet-ctc-1.1b",
        "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
    }

    def __init__(self, model_name: str = "parakeet-ctc-0.6b", device: str = "cuda", sample_rate: int = 16000):
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.model = None
        self._loaded = False

    def load_model(self):
        if self._loaded:
            return

        import nemo.collections.asr as nemo_asr
        import torch

        os.environ["NEMO_RNNT_USE_CUDA_GRAPH"] = "0"
        model_path = self.MODELS.get(self.model_name, self.model_name)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_path)

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.device = "cpu"

        try:
            if hasattr(self.model, 'decoding') and hasattr(self.model.decoding, 'decoding'):
                dec = self.model.decoding.decoding
                if hasattr(dec, 'use_cuda_graph_decoder'):
                    dec.use_cuda_graph_decoder = False
                if hasattr(dec, 'decoding_computer') and dec.decoding_computer:
                    dec.decoding_computer.use_cuda_graphs = False
        except Exception:
            pass

        self._loaded = True

    def transcribe_file(self, audio_path: Union[str, Path]) -> str:
        import torch
        import soundfile as sf
        import numpy as np

        self.load_model()
        audio, sr = sf.read(str(audio_path), dtype='float32')

        if sr != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                ratio = self.sample_rate / sr
                indices = np.linspace(0, len(audio) - 1, int(len(audio) * ratio))
                audio = np.interp(indices, np.arange(len(audio)), audio)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        device = next(self.model.parameters()).device
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
        audio_length = torch.tensor([len(audio)], dtype=torch.long).to(device)

        self.model.eval()
        with torch.inference_mode():
            log_probs, _, _ = self.model.forward(input_signal=audio_tensor, input_signal_length=audio_length)
            predictions = log_probs.argmax(dim=-1)

            if hasattr(self.model, 'tokenizer'):
                pred = predictions[0].cpu().tolist()
                blank_id = self.model.decoder.num_classes_with_blank - 1
                decoded, prev = [], None
                for p in pred:
                    if p != blank_id and p != prev:
                        decoded.append(p)
                    prev = p
                return self.model.tokenizer.ids_to_text(decoded)
        return ""

    def transcribe_audio(self, audio_array, sample_rate: Optional[int] = None) -> str:
        import torch
        import numpy as np

        self.load_model()
        sample_rate = sample_rate or self.sample_rate

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        if sample_rate != self.sample_rate:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=self.sample_rate)
            except ImportError:
                ratio = self.sample_rate / sample_rate
                indices = np.linspace(0, len(audio_array) - 1, int(len(audio_array) * ratio))
                audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)

        device = next(self.model.parameters()).device
        audio_tensor = torch.tensor(audio_array).unsqueeze(0).to(device)
        audio_length = torch.tensor([len(audio_array)], dtype=torch.long).to(device)

        self.model.eval()
        with torch.inference_mode():
            log_probs, _, _ = self.model.forward(input_signal=audio_tensor, input_signal_length=audio_length)
            predictions = log_probs.argmax(dim=-1)

            if hasattr(self.model, 'tokenizer'):
                pred = predictions[0].cpu().tolist()
                blank_id = self.model.decoder.num_classes_with_blank - 1
                decoded, prev = [], None
                for p in pred:
                    if p != blank_id and p != prev:
                        decoded.append(p)
                    prev = p
                return self.model.tokenizer.ids_to_text(decoded)
        return ""

    def record_and_transcribe(self, duration: float = 3.0, device_id: Optional[int] = None,
                              show_countdown: bool = True) -> str:
        import sounddevice as sd
        import numpy as np
        import time

        if show_countdown:
            for i in range(3, 0, -1):
                print(f"{i}...")
                time.sleep(1)
            print("Recording...")

        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                       channels=1, dtype=np.float32, device=device_id)
        sd.wait()
        return self.transcribe_audio(audio.flatten())

    def stream_transcribe(self, callback, chunk_duration: float = 3.0, device_id: Optional[int] = None):
        import sounddevice as sd
        import numpy as np

        chunk_samples = int(chunk_duration * self.sample_rate)
        try:
            while True:
                audio = sd.rec(chunk_samples, samplerate=self.sample_rate, channels=1,
                               dtype=np.float32, device=device_id)
                sd.wait()
                text = self.transcribe_audio(audio.flatten())
                if text.strip():
                    callback(text)
        except KeyboardInterrupt:
            pass

    def list_audio_devices(self):
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"[{i}] {d['name']}{' *' if i == sd.default.device[0] else ''}")


def parse_command(text: str) -> dict:
    t = text.lower().strip()
    result = {"action": None, "object": None, "color": None, "location": None, "raw": text}

    for words, action in [(["pick", "grab", "grasp", "get"], "pick"),
                          (["place", "put", "drop"], "place"),
                          (["move", "go"], "move"), (["stop", "halt"], "stop")]:
        if any(w in t for w in words):
            result["action"] = action
            break

    for patterns, obj in [(["cube", "block"], "cube"), (["cylinder", "can"], "cylinder"),
                          (["capsule", "pill"], "capsule"), (["rect"], "rect_box")]:
        if any(p in t for p in patterns):
            result["object"] = obj
            break

    for c in ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]:
        if c in t:
            result["color"] = c
            break

    if "box" in t or "container" in t:
        result["location"] = "box"
    elif "left" in t:
        result["location"] = "left"
    elif "right" in t:
        result["location"] = "right"

    return result


def generate_task_description(parsed: dict) -> str:
    if not parsed["action"]:
        return parsed["raw"]

    if parsed["action"] == "pick":
        obj = parsed["object"] or "object"
        color = f"the {parsed['color']} " if parsed["color"] else "the "
        suffix = " and place it in the box" if parsed["location"] == "box" else ""
        return f"Pick up {color}{obj}{suffix}"
    elif parsed["action"] == "place":
        loc = f" in the {parsed['location']}" if parsed["location"] else ""
        return f"Place the object{loc}"
    elif parsed["action"] == "move":
        return f"Move to the {parsed['location']}" if parsed["location"] else "Move"
    elif parsed["action"] == "stop":
        return "Stop"
    return parsed["raw"]


def interactive_mode(asr: ParakeetASR):
    print("Parakeet ASR - Enter=record, d=devices, q=quit")
    while True:
        try:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            elif cmd == "d":
                asr.list_audio_devices()
                continue
            text = asr.record_and_transcribe(duration=3.0)
            print(f"Transcribed: \"{text}\"")
            parsed = parse_command(text)
            print(f"Task: {generate_task_description(parsed)}")
        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--file", "-f", type=str)
    parser.add_argument("--model", default="parakeet-ctc-0.6b", choices=list(ParakeetASR.MODELS.keys()))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    asr = ParakeetASR(model_name=args.model, device=args.device)

    if args.list_devices:
        asr.list_audio_devices()
    elif args.file:
        text = asr.transcribe_file(args.file)
        print(f"{text} -> {generate_task_description(parse_command(text))}")
    elif args.stream:
        asr.stream_transcribe(lambda t: print(f"{t} -> {generate_task_description(parse_command(t))}"),
                              chunk_duration=args.duration)
    elif args.interactive:
        interactive_mode(asr)
    else:
        text = asr.record_and_transcribe(duration=args.duration)
        print(f"{text} -> {generate_task_description(parse_command(text))}")


if __name__ == "__main__":
    main()
