import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Thin wrapper around Qwen3-VL for reusable inference.
class Qwen3VLClient:
    def __init__(self,model_path: str,dtype: str = "bf16",device_map: str = "auto",max_new_tokens: int = 32,):
        if not model_path:
            raise ValueError("model_path is empty.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MODEL_PATH does not exist: {model_path}")

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        if dtype.lower() in ["bf16", "bfloat16"]:
            torch_dtype = torch.bfloat16
        elif dtype.lower() in ["fp16", "float16"]:
            torch_dtype = torch.float16
        elif dtype.lower() in ["fp32", "float32"]:
            torch_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        print(f"[Model] Loading Qwen3-VL from: {model_path}")
        print(f"[Model] dtype={torch_dtype}, device_map={device_map}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def generate_from_messages(self,messages: List[Dict[str, Any]],max_new_tokens: Optional[int] = None,do_sample: bool = False,) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()