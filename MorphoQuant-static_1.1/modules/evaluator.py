import os
import json
import csv
import time
import glob
import ast
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from datetime import datetime
from io import BytesIO
from datasets import load_dataset, get_dataset_config_names
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from qwen_omni_utils import process_mm_info

from modules import device_utils

# Optional import for morpho
try:
    from bitsandbytes.quantization_utils.quant_modules import QuantAct
except ImportError:
    QuantAct = type("DummyQuantAct", (object,), {})


# def _quantact_use_full_channel_outlier_mask(config, layer_name: str) -> bool:
#     """Whether to set outlier_mask = all True (every channel can use sparse FP compensation in QuantAct)."""
#     ln = (layer_name or "").lower()
#     subs = getattr(config.quant, "force_outlier_quantact_substrings", None) or []
#     if not isinstance(subs, (list, tuple)):
#         subs = [subs]
#     if any(str(s).lower() in ln for s in subs):
#         return True
#     is_internvl = str(getattr(config.model, "model_type", "")).lower().startswith("internvl")
#     # Qwen
#     if (not is_internvl) and ln and (
#         "down_proj" in ln or "feed_forward.w2" in ln or "mlp.fc2" in ln
#     ):
#         return True
#     return False

def _quantact_use_full_channel_outlier_mask(config, layer_name: str) -> bool:
    """基于 X光分布图 的精准免死金牌分发"""
    ln = (layer_name or "").lower()
    
    # =================================================================
    # 🌟 战术一：死保跨模态“绞肉机”（根据图表右半边前三分之一的核爆区）
    # LLM 的前 6 层（或者前 8 层，你看你具体模型 depth 是多少，覆盖红线密集区）
    # =================================================================
    if ".mlp.down_proj" in ln:
        # print(ln)
        return True

    # if any(f".{i}.mlp.down_proj" in ln for i in range(10)):
    #     return True
    
    # if ".self_attn.o_proj" in ln and "thinker.model.layers." in ln:
    #     # print(ln)
    #     return True

    # if any(f"thinker.model.layers.{i}." in ln for i in range(3)): 
    #     return True

    # if any(f"thinker.model.layers.{i}.self_attn.v_proj" in ln for i in range(8)): 
    #     # print(ln)
    #     return True

    # if any(f"thinker.model.layers.{i}.self_attn.o_proj" in ln for i in range(10)): 
    #     print(ln)
    #     return True

    # if any(f".{i}.self_attn.o_proj" in ln for i in range(10)): 
    #     return True

    # if any(f"visual.blocks.{i}.mlp.down_proj" in ln for i in range(3)): 
    #     return True

    # =================================================================
    # 🌟 战术二：死保视觉的“咽喉”（根据图表左半边的首尾异动）
    # 保护视觉投影层 和 视觉编码器的极其敏感层
    # =================================================================
    # if "vision_projector" in ln or "mm_projector" in ln or "patch_embed" in ln:
    #     return True

    # =================================================================
    # 🌟 战术三：大魔王 down_proj 的常规保护（看图中那些周期性的小尖刺）
    # =================================================================
    # is_internvl = str(getattr(config.model, "model_type", "")).lower().startswith("internvl")
    # if (not is_internvl) and ln and (
    #     "down_proj" in ln or "feed_forward.w2" in ln or "mlp.fc2" in ln
    # ):
    #     return True

    return False


def _processor_expects_multimodal_chat_blocks(processor):
    """True when apply_chat_template must receive list-shaped message content (type/text/image/...).

    Qwen2.5-OmniProcessor.apply_chat_template reads conversation[0]['content'][0]['text']; flattening
    content to a string breaks that path. Other stacks may expect plain-string content per turn.
    """
    if processor is None:
        return False
    cls = type(processor)
    name = cls.__name__
    mod = (getattr(cls, "__module__", None) or "").lower()
    if name == "Qwen2_5OmniProcessor":
        return True
    return "qwen2_5_omni" in mod and "processing" in mod


def _normalize_conversation_for_template(conversation, processor=None):
    if _processor_expects_multimodal_chat_blocks(processor):
        return conversation
    conv_for_template = []
    for msg in conversation:
        msg_copy = {k: v for k, v in msg.items() if k != "content"}
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    t = c.get("type")
                    if t == "text":
                        parts.append(str(c.get("text", "")))
                    elif t == "image":
                        parts.append("<image>")
                    elif t == "audio":
                        parts.append("<audio>")
                    elif t == "video":
                        parts.append("<video>")
                    else:
                        parts.append(str(c))
                else:
                    parts.append(str(c))
            msg_copy["content"] = " ".join([p for p in parts if p])
        else:
            msg_copy["content"] = str(content)
        conv_for_template.append(msg_copy)
    return conv_for_template

def get_image_path(base_dir, qid, split="test"):
    img_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/images/{split}/{qid}/image.png"
    return img_path

def build_prompt(question, choices, context=""):
    prompt = f"题目：{question}\n"
    if context:
        prompt += f"背景：{context}\n"
    prompt += "选项：\n"
    for idx, choice in enumerate(choices):
        prompt += f"{chr(65+idx)}. {choice}\n"
    prompt += (
        "请你仔细阅读题目，只能输出一个大写字母（A、B、C 或 D），"
        "不要输出任何解释或多余的内容。"
    )
    return prompt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_internvl_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGBA').convert('RGB') if img.mode in ('P', 'PA') else (img.convert('RGB') if img.mode != 'RGB' else img)),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images

def load_internvl_image(image_input, input_size=448, max_num=12):
    def _safe_rgb(img):
        if img.mode in ('P', 'PA'):
            img = img.convert('RGBA')
        return img.convert('RGB')

    if isinstance(image_input, Image.Image):
        image = _safe_rgb(image_input)
    elif isinstance(image_input, str):
        image = _safe_rgb(Image.open(image_input))
    elif isinstance(image_input, dict):
        if image_input.get("path"):
            image = _safe_rgb(Image.open(image_input["path"]))
        elif image_input.get("bytes"):
            image = _safe_rgb(Image.open(BytesIO(image_input["bytes"])))
        else:
            raise ValueError("Unsupported image dict without path or bytes.")
    elif hasattr(image_input, "path") and getattr(image_input, "path"):
        image = _safe_rgb(Image.open(image_input.path))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    transform = build_internvl_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)

def extract_image_entries(example):
    image_entries = []

    for key in sorted(example.keys()):
        if key.startswith("image_") and example.get(key) is not None:
            image_entries.append(example[key])

    if not image_entries and example.get("image") is not None:
        image_entries.append(example["image"])

    if not image_entries and example.get("images"):
        images = example["images"]
        if isinstance(images, list):
            image_entries.extend([img for img in images if img is not None])
        else:
            image_entries.append(images)

    return image_entries

def build_internvl_question(question, options, num_images):
    question = question.replace("<image 1>", "").replace("<image 2>", "").replace("<image 3>", "")
    prompt = f"Question: {question.strip()}\n"
    prompt += "Options:\n"
    prompt += options
    prompt += "Answer with the option letter directly."

    if num_images <= 1:
        return f"<image>\n{prompt}"

    image_prefix = ''.join([f"Image-{idx + 1}: <image>\n" for idx in range(num_images)])
    return image_prefix + prompt

class ScienceQADataset(Dataset):
    def __init__(self, base_dir, problems_path, split_path, split="test", only_samples_with_images=False, max_samples=None, skip_samples=0):
        with open(problems_path, "r", encoding="utf-8") as f:
            self.problems = json.load(f)
        with open(split_path, "r", encoding="utf-8") as f:
            self.pid_splits = json.load(f)

        self.base_dir = base_dir
        self.split = split
        raw_pids = self.pid_splits[split]

        self.pids = []
        for pid in raw_pids:
            sample = self.problems[str(pid)]
            if only_samples_with_images and not sample.get("image"):
                continue
            self.pids.append(pid)

        # 增加 skip 逻辑以使数据集切片不重叠
        if skip_samples > 0:
            self.pids = self.pids[skip_samples:]

        if max_samples is not None:
            self.pids = self.pids[:max_samples]

        # Pre-compute image path existence to avoid per-sample os.path.exists calls
        self._image_paths = {}
        for pid in self.pids:
            sample = self.problems[str(pid)]
            if "image" in sample and sample["image"]:
                candidate = get_image_path(self.base_dir, pid, split=self.split)
                if os.path.exists(candidate):
                    self._image_paths[pid] = candidate

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        sample = self.problems[str(pid)]

        question = sample["question"]
        choices = sample["choices"]
        answer_idx = sample["answer"]
        answer = chr(65 + answer_idx)
        context = sample.get("hint", "")

        prompt = build_prompt(question, choices, context)

        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请回答科学题目。"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        image_path = self._image_paths.get(pid)
        if image_path:
            conversation[1]["content"].append({"type": "image", "image": image_path})

        return {
            "pid": pid,
            "question": question,
            "answer": answer,
            "choices": choices,
            "conversation": conversation,
            "image_path": image_path
        }

def collate_fn_scienceqa(batch):
    pids = [item["pid"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    conversations = [item["conversation"] for item in batch]
    
    return {
        "pids": pids,
        "questions": questions,
        "answers": answers,
        "conversations": conversations
    }

class BaseEvaluator:
    def __init__(self, model, processor, config, base_dir=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.base_dir = base_dir
        self.model_type = getattr(config.model, "model_type", "qwen2_5_omni").lower()

    def _is_internvl(self):
        return self.model_type.startswith("internvl")

    def _count_visual_tokens_from_inputs(self, conversation, inputs):
        """Estimate visual token count by comparing multimodal vs text-only token counts.

        Returns (visual_token_count, total_tokens).
        """
        try:
            do_mquant = getattr(self.config.quant, 'simulate_mquant', False)
        except Exception:
            do_mquant = False
        if not do_mquant:
            return 0, inputs["input_ids"].size(1)

        total_tokens = inputs["input_ids"].size(1)

        # Build a text-only conversation (strip images/audio/video)
        text_only_conv = []
        has_visual = False
        for msg in conversation:
            msg_copy = {"role": msg["role"]}
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for c in content:
                    if isinstance(c, dict):
                        t = c.get("type", "")
                        if t == "text":
                            text_parts.append(str(c.get("text", "")))
                        elif t in ("image", "video"):
                            has_visual = True
                    else:
                        text_parts.append(str(c))
                msg_copy["content"] = " ".join(text_parts)
            else:
                msg_copy["content"] = str(content)
            text_only_conv.append(msg_copy)

        if not has_visual:
            return 0, total_tokens

        # Process text-only to count text tokens
        try:
            text_only_str = self.processor.apply_chat_template(
                _normalize_conversation_for_template(text_only_conv, self.processor),
                add_generation_prompt=True, tokenize=False
            )
            if isinstance(text_only_str, (list, tuple)):  # transformers 4.5x 返回 List[str]
                assert len(text_only_str) == 1, f"apply_chat_template returned {len(text_only_str)} prompts, expected 1"
                text_only_str = text_only_str[0]
            text_inputs = self.processor(
                text=text_only_str, return_tensors="pt", padding=True
            )
            text_tokens = text_inputs["input_ids"].size(1)
        except Exception:
            return 0, total_tokens

        visual_cnt = max(0, total_tokens - text_tokens)
        return visual_cnt, total_tokens

    def _set_mquant_ctx(self, conversation, inputs):
        """Set MQuant visual token count on model before forward pass (for MSQ).

        首次调用时会自动在模型上创建 _mquant_visual_token_count 属性，
        后续 MQuantLinear 通过 _get_visual_token_count 读取。
        """
        try:
            do_mquant = getattr(self.config.quant, 'simulate_mquant', False)
        except Exception:
            return
        if not do_mquant:
            return
        v_cnt, _ = self._count_visual_tokens_from_inputs(conversation, inputs)
        self.model._mquant_visual_token_count = v_cnt

    def _build_mode_suffix(self):
        """Build a generic prefix suffix based on config method and size setup for output filename logs."""
        return f"{self.config.model.quant_method}_calib{self.config.quant.calib_size}_search{getattr(self.config.quant, 'search_size', 0)}"

    def _print_config_yaml(self):
        """打印当前使用的 YAML 配置文件完整内容。"""
        print(f"\n{'='*50}")
        print(f"  YAML 配置文件: {self.config.config_path}")
        print(f"{'='*50}")
        print(self.config.config_raw)
        print(f"{'='*50}\n")

    def _search_batch_size(self):
        return max(1, int(getattr(self.config.data, "batch_size", 1)))

    def _reset_quantact_search_best(self):
        """重置所有 QuantAct 的最佳搜索缓存"""
        for _, module in self.model.named_modules():
            if hasattr(module, 'reset_search_best'):
                module.reset_search_best()

    def _set_quantact_search_state(self, enabled: bool):
        """Enable or disable the quantization search state for QuantAct modules."""
        for _, module in self.model.named_modules():
            if type(module).__name__ == "QuantAct" or hasattr(module, "set_search"):
                if hasattr(module, "set_search"):
                    module.set_search(search=enabled)

    def _configure_quantact_for_prepare(self):
        disable_boundary_coopt = getattr(self.config.quant, 'disable_boundary_cooptimization', False)
        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                if not disable_boundary_coopt:
                    module.set_search(search=True)
                module.set_calibrate(calibrate=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

                if hasattr(module, 'set_search_ratio_lower_bound'):
                    module.set_search_ratio_lower_bound(getattr(self.config.quant, 'search_ratio_lower_bound', 0.6))

                if hasattr(module, "dispersion_score") and module.dispersion_score is not None:
                    dispersion_score = module.dispersion_score
                    if getattr(self.config.quant, 'disable_sparse_compensation', False):
                        outlier_mask = torch.zeros_like(dispersion_score, dtype=torch.bool)
                    elif _quantact_use_full_channel_outlier_mask(self.config, getattr(module, "layer_name", "") or ""):
                        outlier_mask = torch.ones_like(dispersion_score, dtype=torch.bool)
                    else:
                        outlier_std_threshold = getattr(self.config.quant, 'outlier_std_threshold', 2.0)
                        threshold = dispersion_score.mean() + outlier_std_threshold * dispersion_score.std()
                        outlier_mask = dispersion_score >= threshold
                    module.register_buffer("outlier_mask", outlier_mask)

    def _finalize_quantact_prepare(self):
        disable_boundary_coopt = getattr(self.config.quant, 'disable_boundary_cooptimization', False)
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                if disable_boundary_coopt:
                    if hasattr(module, 'finalize_without_search'):
                        module.finalize_without_search()
                else:
                    if hasattr(module, 'finalize_search'):
                        module.finalize_search()
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

    def _normalize_mcq_answer(self, answer, options=None):
        if answer is None:
            return ""

        normalized_options = None
        if options is not None:
            normalized_options = options
            if isinstance(normalized_options, str):
                try:
                    normalized_options = ast.literal_eval(normalized_options)
                except Exception:
                    normalized_options = [normalized_options]

        if isinstance(answer, (int, np.integer)):
            return chr(65 + int(answer))

        answer_text = str(answer).strip()
        if not answer_text:
            return ""

        if normalized_options is not None:
            for idx, option in enumerate(normalized_options):
                if answer_text == str(option).strip():
                    return chr(65 + idx)

        compact = answer_text.upper().strip().rstrip(".")
        compact = compact.strip("()")
        if len(compact) == 1 and compact in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return compact

        if answer_text.isdigit() and normalized_options is not None:
            index = int(answer_text)
            if 1 <= index <= len(normalized_options):
                return chr(64 + index)
            if 0 <= index < len(normalized_options):
                return chr(65 + index)

        if normalized_options is not None:
            for idx, option in enumerate(normalized_options):
                if answer_text.lower() == str(option).strip().lower():
                    return chr(65 + idx)

        return compact

    def _extract_assistant_response(self, text):
        raise NotImplementedError

    def _inference(self, conversation):
        raise NotImplementedError

    def calibrate(self):
        pass

    def prepare(self):
        pass

    def evaluate(self, save_dir):
        raise NotImplementedError


class ScienceQAEvaluator(BaseEvaluator):
    def __init__(self, model, processor, config, base_dir):
        super().__init__(model, processor, config, base_dir)
        
        self.problems_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/problems.json"
        self.split_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/pid_splits.json"
        self.model_type = getattr(config.model, "model_type", "qwen2_5_omni").lower()



    def _extract_assistant_response(self, text):
        response = text
        if "assistant" in text.lower():
            parts = text.split("assistant", 1)
            if len(parts) > 1:
                response = parts[1].strip()

        # 更严谨的正则匹配。先找 (A) 或 A. 这种，如果没有，从后往前找单独的字母。
        match = re.search(r'(?:\(([A-L])\)|([A-L])\.)', response)
        if match:
            return match.group(1) or match.group(2)
        
        # 兜底：找独立的 A-L，尽量避免匹配单词首字母
        match = re.search(r'\b([A-L])\b', response)
        if match:
            return match.group(1)
            
        # 终极兜底：如果就是不加标点，取最后一个出现的大写字母 A-L
        matches = re.findall(r'[A-L]', response)
        if matches:
            return matches[-1]
            
        return "Failed"

    def _inference(self, conversation):
        text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
        if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
            assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
            text = text[0]
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=False
        )

        inputs = inputs.to(self.model.device).to(self.model.dtype)
        self._set_mquant_ctx(conversation, inputs)
        text_ids = self.model.generate(
            **inputs,
            thinker_max_new_tokens=10,
            use_audio_in_video=False,
            return_audio=False,
        )
        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text[0]

    def _build_internvl_prompt(self, question, choices):
        prompt = f"Question: {question}\nOptions:\n"
        for idx, choice in enumerate(choices):
            prompt += f"{chr(65 + idx)}. {choice}\n"
        prompt += "Answer with the option letter directly."
        return prompt

    def _inference_internvl(self, data, max_new_tokens=10, *, quant_stats_pass: bool = False, memory_override=None):
        image_path = data.get("image_path")
        ovr = memory_override if isinstance(memory_override, dict) else {}

        pixel_values = None
        num_patches_list = None
        if image_path:
            tile_cap = ovr.get("max_num_tiles")
            if tile_cap is None:
                tile_cap = getattr(
                    self.config.data,
                    "internvl_calib_max_num_tiles" if quant_stats_pass else "internvl_eval_max_num_tiles",
                    None,
                )
            if tile_cap is None:
                tile_cap = getattr(self.config.data, "internvl_max_num_tiles", None)
            max_num = int(tile_cap) if tile_cap is not None else 12

            pixel_values = load_internvl_image(image_path, max_num=max_num)
            num_patches_list = [pixel_values.size(0)]

            model_device = next(self.model.parameters()).device
            model_dtype = getattr(self.model, "dtype", torch.bfloat16)
            pixel_values = pixel_values.to(model_device).to(model_dtype)

            prompt = f"<image>\n{self._build_internvl_prompt(data['question'], data['choices'])}"
        else:
            prompt = self._build_internvl_prompt(data['question'], data['choices'])

        # Set visual token count for FreeAct/MQuant token-type-aware quantization
        num_image_token = getattr(self.model, 'num_image_token', 256)
        if pixel_values is not None and num_patches_list:
            visual_cnt = sum(num_patches_list) * num_image_token + 2  # +2 for <img>/</img> tags
        else:
            visual_cnt = 0
        if hasattr(self.model, '_freeact_visual_token_count'):
            self.model._freeact_visual_token_count = visual_cnt
        if hasattr(self.model, '_mquant_visual_token_count'):
            self.model._mquant_visual_token_count = visual_cnt

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
        try:
            response = self.model.chat(
                self.processor,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        finally:
            del pixel_values

        if isinstance(response, tuple):
            return response[0]
        return response

    def calibrate(self):
        do_smoothquant = getattr(self.config.quant, 'simulate_smoothquant', False)
        do_awq = getattr(self.config.quant, 'simulate_awq', False)
        do_mbq = getattr(self.config.quant, 'simulate_mbq', False)
        do_mquant = getattr(self.config.quant, 'simulate_mquant', False)
        do_freeact = getattr(self.config.quant, 'simulate_freeact', False)
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)

        if self.config.model.quant_method not in ('morpho', 'qvlm', 'morpho_withhif8', 'morpho_withhif4', 'freeact') and not do_smoothquant and not do_awq and not do_mbq and not do_mquant and not do_freeact and not do_qlora_act:
            print(f"{self.config.model.quant_method}模式：跳过校准步骤。")
            return

        print('\n==> start batched calibrate')
        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=True)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        if do_smoothquant:
            from modules.smoothquant_layers import set_smoothquant_observe
            set_smoothquant_observe(self.model, enabled=True)

        if do_awq:
            from modules.awq_layers import set_awq_observe
            set_awq_observe(self.model, enabled=True)

        if do_mbq:
            from modules.mbq_layers import set_mbq_observe
            set_mbq_observe(self.model, enabled=True)

        if do_mquant:
            from modules.mquant_layers import set_mquant_observe
            set_mquant_observe(self.model, enabled=True)
            # MQuant 需要知道每个样本的视觉 token 数量以分模态统计
            self.model._mquant_visual_token_count = None
            # 重新注册一个 forward hook 来基于 processor 输出推断 visual_token_count
            # 对于 ScienceQA，我们通过数据集中的 image_path 来判断
            setattr(self.model, '_mquant_visual_token_count', None)

        if do_freeact:
            from modules.freeact_layers import set_freeact_observe
            set_freeact_observe(self.model, enabled=True)
            self.model._freeact_visual_token_count = 0

        dataset = ScienceQADataset(
            base_dir=self.base_dir,
            problems_path=self.problems_path,
            split_path=self.split_path,
            split="val",
            only_samples_with_images=self.config.data.only_samples_with_images,
            max_samples=self.config.quant.calib_size
        )

        if self._is_internvl():
            for idx in tqdm(range(len(dataset))):
                sample = dataset[idx]
                with torch.no_grad():
                    self._inference_internvl(sample, max_new_tokens=1, quant_stats_pass=True)

            for name, module in self.model.named_modules():
                if isinstance(module, QuantAct):
                    module.compute_dispersion_score()

            if do_freeact:
                from modules.freeact_layers import finalize_freeact
                finalize_freeact(self.model)

            if do_smoothquant:
                from modules.smoothquant_layers import finalize_smoothquant
                finalize_smoothquant(self.model, alpha=getattr(self.config.quant, 'smoothquant_alpha', 0.5))

            if do_awq:
                from modules.awq_layers import finalize_awq
                finalize_awq(self.model)

            if do_mbq:
                from modules.mbq_layers import finalize_mbq
                finalize_mbq(self.model)

            print('==> end calibrate')
            return

        dataloader = DataLoader(dataset, batch_size=self.config.quant.batch_size, shuffle=False, collate_fn=collate_fn_scienceqa)

        batch_count = 0

        for batch_data in tqdm(dataloader, total=len(dataloader)):
            conversations = batch_data["conversations"]
            
            batch_texts = []
            batch_images = []
            batch_audios = []
            batch_videos = []
            
            for conversation in conversations:
                if getattr(self.config.quant, "calib_without_audio", False):
                    for msg in conversation:
                        if msg.get("role") == "user" and "content" in msg:
                            msg["content"] = [c for c in msg["content"] if c.get("type") != "audio"]

                if getattr(self.config.quant, "calib_without_video", False):
                    for msg in conversation:
                        if msg.get("role") == "user" and "content" in msg:
                            msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]
                
                text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
                if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
                    assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
                    text = text[0]
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                
                batch_texts.append(text)
                if images: batch_images.extend(images)
                if audios: batch_audios.extend(audios)
                if videos: batch_videos.extend(videos)

            # NOTE: 不需要在 calibrate() 中设置 set_search(True)，因为 _calibrate=True 时 QuantAct.forward()
            # 始终走 _calibrate 分支，search 标志在此时无效。search 会在 prepare() 中正确启用。
            
            final_images = batch_images if len(batch_images) > 0 else None
            final_audios = batch_audios if len(batch_audios) > 0 else None
            final_videos = batch_videos if len(batch_videos) > 0 else None
            
            inputs = self.processor(
                text=batch_texts,
                images=final_images,
                audio=final_audios,
                videos=final_videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False
            )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            if do_mbq:
                from modules.mbq_layers import set_mbq_modality
                modality = "multimodal" if (batch_images or batch_audios or batch_videos) else "text"
                set_mbq_modality(self.model, modality)

            if do_mquant and len(conversations) == 1:
                self._set_mquant_ctx(conversations[0], inputs)

            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    thinker_max_new_tokens=1,
                    use_audio_in_video=False,
                    return_audio=False,
                )

            batch_count += 1

        # 校准结束后计算 Dispersion Score，用于离群点检测
        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.compute_dispersion_score()
                # print(f"  [Dispersion] Layer {getattr(module, 'layer_name', name)}: max score={module.dispersion_score.max().item():.4f}" if module.dispersion_score is not None else f"  [Dispersion] Layer {getattr(module, 'layer_name', name)}: skipped (no data)")

        if do_smoothquant:
            from modules.smoothquant_layers import finalize_smoothquant
            finalize_smoothquant(self.model, alpha=getattr(self.config.quant, 'smoothquant_alpha', 0.5))

        if do_awq:
            from modules.awq_layers import finalize_awq
            finalize_awq(self.model)

        if do_mbq:
            from modules.mbq_layers import finalize_mbq
            finalize_mbq(self.model)

        if do_mquant:
            from modules.mquant_layers import finalize_mquant
            finalize_mquant(self.model)
            print("[MQuant] 模态特化 scale 计算完成，权重已量化。")

        print('==> end calibrate')

    def prepare(self):
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)
        if self.config.model.quant_method not in ('morpho', 'qvlm', 'morpho_withhif8', 'morpho_withhif4') and not do_qlora_act:
            return

        do_mquant = getattr(self.config.quant, 'simulate_mquant', False)

        print('\n==> start prepare for inference')
        self._configure_quantact_for_prepare()

        disable_boundary_coopt = getattr(self.config.quant, 'disable_boundary_cooptimization', False)

        if not disable_boundary_coopt:
            dataset = ScienceQADataset(
                base_dir=self.base_dir,
                problems_path=self.problems_path,
                split_path=self.split_path,
                split="val",
                only_samples_with_images=self.config.data.only_samples_with_images,
                max_samples=self.config.quant.search_size,
                skip_samples=self.config.quant.calib_size  # 跳过前面用于校准的样本，保证数据隔离
            )

            if self._is_internvl():
                for idx in range(len(dataset)):
                    sample = dataset[idx]
                    with torch.no_grad():
                        self._inference_internvl(sample, max_new_tokens=1, quant_stats_pass=True)

                self._finalize_quantact_prepare()
                print('==> end prepare for inference')
                return

            dataloader = DataLoader(dataset, batch_size=self.config.quant.batch_size, shuffle=False, collate_fn=collate_fn_scienceqa)

            batch_idx = 0
            for batch_data in dataloader:
                print("Search Batch:", int(batch_idx + 1))
                batch_idx += 1
                conversations = batch_data["conversations"]

                batch_texts = []
                batch_images = []
                batch_audios = []
                batch_videos = []

                for conversation in conversations:
                    if getattr(self.config.quant, "calib_without_video", False):
                        for msg in conversation:
                            if msg.get("role") == "user" and "content" in msg:
                                msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]

                    conv_for_template = _normalize_conversation_for_template(conversation, self.processor)
                    text = self.processor.apply_chat_template(conv_for_template, add_generation_prompt=True, tokenize=False)
                    if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
                        assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
                        text = text[0]
                    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

                    batch_texts.append(text)
                    if images: batch_images.extend(images)
                    if audios: batch_audios.extend(audios)
                    if videos: batch_videos.extend(videos)

                final_images = batch_images if len(batch_images) > 0 else None
                final_audios = batch_audios if len(batch_audios) > 0 else None
                final_videos = batch_videos if len(batch_videos) > 0 else None

                if len(batch_texts) > 0:
                    inputs = self.processor(
                        text=batch_texts,
                        images=final_images,
                        audio=final_audios,
                        videos=final_videos,
                        return_tensors="pt",
                        padding=True,
                        use_audio_in_video=False
                    )
                    inputs = inputs.to(self.model.device).to(self.model.dtype)

                    if do_mquant and len(conversations) == 1:
                        self._set_mquant_ctx(conversations[0], inputs)

                    with torch.no_grad():
                        self.model.generate(
                            **inputs,
                            thinker_max_new_tokens=1,
                            use_audio_in_video=False,
                            return_audio=False,
                        )

        self._finalize_quantact_prepare()

        print('==> end prepare for inference')

    def evaluate(self, save_dir):
        """Evaluate model and save results to save_dir (a file path)."""
        num_samples = self.config.data.num_samples if self.config.data.num_samples != -1 else None
        dataset = ScienceQADataset(
            base_dir=self.base_dir,
            problems_path=self.problems_path,
            split_path=self.split_path,
            split="test",
            only_samples_with_images=self.config.data.only_samples_with_images,
            max_samples=num_samples
        )

        total, correct = 0, 0

        # Ensure QuantAct modules are in eval mode (calibrate=False, search=False)
        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=False)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        if self.config.quant.debug_quant_act:
            for name, module in self.model.named_modules():
                if isinstance(module, QuantAct):
                    module.set_debug(debug=True)
            print("✅ QuantAct debug mode enabled.")

        # Streaming write: open CSV once and write rows as they are produced
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        keys = ["pid", "question", "reference", "prediction", "raw_output"]
        f = open(save_dir, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        try:
            for i in tqdm(range(len(dataset))):
                data = dataset[i]

                pid = data["pid"]
                question = data["question"]
                answer = data["answer"]
                conversation = data["conversation"]
                output = ""

                start_time = time.time()
                try:
                    if self._is_internvl():
                        try:
                            output = self._inference_internvl(data, max_new_tokens=10)
                        except device_utils.oom_errors() as _oom:
                            if not device_utils.is_oom(_oom):
                                raise
                            device_utils.empty_cache()
                            print(f"  Device OOM on {pid}; retry once with stricter vision (4 tiles).")
                            output = self._inference_internvl(data, max_new_tokens=10, memory_override={"max_num_tiles": 4})
                    else:
                        output = self._inference(conversation)
                    elapsed_time = time.time() - start_time

                    if elapsed_time > 10:
                        print(f"样本 {pid} 计算时间超过 10 秒，跳过。")
                        pred = "错误"
                    else:
                        pred = self._normalize_mcq_answer(
                            self._extract_assistant_response(output),
                            data.get("choices"),
                        )
                except Exception as e:
                    import traceback
                    print(f"\n========== 样本 {pid} 报错详情 ==========")
                    traceback.print_exc()
                    print("==========================================\n")
                    pred = "错误"

                total += 1
                answer = self._normalize_mcq_answer(answer, data.get("choices"))
                if pred == answer:
                    correct += 1

                writer.writerow({
                    "pid": pid,
                    "question": question,
                    "reference": answer,
                    "prediction": pred,
                    "raw_output": output if 'output' in locals() else ""
                })
                # Periodic flush for crash-safety without excessive syscalls
                if total % 10 == 0:
                    f.flush()
                # 每 100 个样本报告一次running accuracy（tqdm.write 以免打断进度条）
                if total % 100 == 0:
                    tqdm.write(
                        f"  [{total}/{len(dataset)}] running accuracy: "
                        f"{correct / total:.4f} ({correct}/{total})"
                    )
        finally:
            f.close()

        acc = correct / total if total > 0 else 0.0
        print(f"\n{'='*20} ScienceQA Evaluation Summary {'='*20}")
        print(f"Quant Method   : {self.config.model.quant_method}")
        print(f"Total Samples  : {total}")
        print(f"Correct        : {correct}")
        print(f"Accuracy       : {acc:.4f} ({acc*100:.2f}%)")
        print(f"{'='*70}")
        self._print_config_yaml()

        print(f"预测结果已保存到 {save_dir}")
        return acc


class MMMUEvaluator(BaseEvaluator):
    def __init__(self, model, processor, config, base_dir):
        super().__init__(model, processor, config, base_dir)
        self.dataset_dir = config.data.dataset_dir
        self.target_subject = config.data.target_subject
        self.run_pca = config.data.run_pca
        self.model_type = getattr(config.model, "model_type", "qwen2_5_omni").lower()


    # def _extract_assistant_response(self, text):
    #     response = text
    #     if "assistant" in text.lower():
    #         parts = text.split("assistant", 1)
    #         if len(parts) > 1:
    #             response = parts[1].strip()

    #     match = re.search(r"[A-F]", response)
    #     if match:
    #         return match.group()
    #     return "Failed"
    def _extract_assistant_response(self, text):
        response = text
        if "assistant" in text.lower():
            parts = text.split("assistant", 1)
            if len(parts) > 1:
                response = parts[1].strip()

        # 【修复 Bug 4】: 更严谨的正则匹配。先找 (A) 或 A. 这种，如果没有，从后往前找单独的字母。
        match = re.search(r'(?:\(([A-F])\)|([A-F])\.)', response)
        if match:
            return match.group(1) or match.group(2)
        
        # 兜底：找独立的 A-F，尽量避免匹配单词首字母
        match = re.search(r'\b([A-F])\b', response)
        if match:
            return match.group(1)
            
        # 终极兜底：如果就是不加标点，取最后一个出现的大写字母 A-F
        matches = re.findall(r'[A-F]', response)
        if matches:
            return matches[-1]
            
        return "Failed"

    def _format_options(self, options):
        if isinstance(options, str):
            try:
                options = eval(options)
            except Exception:
                pass

        opt_str = ""
        for i, opt in enumerate(options):
            key = chr(65 + i)
            opt_str += f"{key}. {opt}\n"
        return opt_str

    def _build_prompt(self, example):
        question = example["question"]
        options = example["options"]
        question = question.replace("<image 1>", "").replace("<image 2>", "").replace("<image 3>", "")

        prompt = f"Question: {question.strip()}\n"
        prompt += "Options:\n"
        prompt += self._format_options(options)
        prompt += "Answer with the option letter directly."
        return prompt

    def _inference(self, conversation):
        text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
        if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
            assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
            text = text[0]
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        self._set_mquant_ctx(conversation, inputs)

        with torch.no_grad():
            text_ids = self.model.generate(**inputs, use_audio_in_video=False, thinker_max_new_tokens=20)

        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text[0]

    def _resolve_image_value(self, value):
        if isinstance(value, str):
            return value
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            if value.get("path"):
                return value["path"]
            if value.get("bytes"):
                return value
        if hasattr(value, "path") and getattr(value, "path"):
            return value.path
        return value

    def _inference_internvl(self, example, max_new_tokens=32, *, quant_stats_pass: bool = False, memory_override=None):
        image_entries = [self._resolve_image_value(entry) for entry in extract_image_entries(example)]
        image_entries = [entry for entry in image_entries if entry is not None]
        if not image_entries:
            raise ValueError(f"MMMU sample {example.get('id', 'unknown')} does not contain any image inputs.")

        if quant_stats_pass:
            max_im = getattr(self.config.data, "internvl_calib_max_images", None)
            if max_im is not None and int(max_im) >= 1:
                image_entries = image_entries[: int(max_im)]
            if not image_entries:
                raise ValueError(f"MMMU sample {example.get('id', 'unknown')} has no images after calib crop.")

            tile_cap = getattr(self.config.data, "internvl_calib_max_num_tiles", None)
            if tile_cap is None:
                tile_cap = getattr(self.config.data, "internvl_max_num_tiles", None)
            if tile_cap is None:
                max_num = 6
            else:
                max_num = int(tile_cap)
        else:
            ovr = memory_override if isinstance(memory_override, dict) else {}
            max_im = ovr.get("max_images")
            if max_im is None:
                max_im = getattr(self.config.data, "internvl_eval_max_images", None)
            if max_im is not None and int(max_im) >= 1:
                image_entries = image_entries[: int(max_im)]
            if not image_entries:
                raise ValueError(f"MMMU sample {example.get('id', 'unknown')} has no images after eval crop.")

            tile_cap = ovr.get("max_num_tiles")
            if tile_cap is None:
                tile_cap = getattr(self.config.data, "internvl_eval_max_num_tiles", None)
            if tile_cap is None:
                tile_cap = getattr(self.config.data, "internvl_max_num_tiles", None)
            if tile_cap is None:
                max_num = 12
            else:
                max_num = int(tile_cap)

        pixel_values_list = [load_internvl_image(entry, max_num=max_num) for entry in image_entries]
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [item.size(0) for item in pixel_values_list]

        options = self._format_options(example["options"])
        prompt = build_internvl_question(example["question"], options, len(image_entries))
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        model_device = next(self.model.parameters()).device
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        pixel_values = pixel_values.to(model_device).to(model_dtype)

        try:
            if len(num_patches_list) > 1:
                response = self.model.chat(
                    self.processor,
                    pixel_values,
                    prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=None,
                    return_history=False,
                )
            else:
                response = self.model.chat(
                    self.processor,
                    pixel_values,
                    prompt,
                    generation_config,
                    history=None,
                    return_history=False,
                )
        finally:
            del pixel_values_list
            del pixel_values

        if isinstance(response, tuple):
            return response[0]
        return response

    def _register_pca_hooks(self):
        activations = {
            "thinker_input": [],
            "visual_encoder": [],
            "text_embed": [],
        }

        def get_input_activation(name):
            def hook(module, inputs):
                hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
                activations[name].append(
                    hidden_states.detach().cpu().to(torch.float32).numpy().reshape(-1, hidden_states.shape[-1])
                )

            return hook

        def get_output_activation(name):
            def hook(module, inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                activations[name].append(
                    hidden_states.detach().cpu().to(torch.float32).numpy().reshape(-1, hidden_states.shape[-1])
                )

            return hook

        hook_handles = []

        thinker_layers = None
        if hasattr(self.model, "thinker") and hasattr(self.model.thinker, "model") and hasattr(self.model.thinker.model, "layers"):
            thinker_layers = self.model.thinker.model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            thinker_layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            thinker_layers = self.model.layers

        if thinker_layers is not None:
            h1 = thinker_layers[0].register_forward_pre_hook(get_input_activation("thinker_input"))
            hook_handles.append(h1)
        else:
            print("Warning: Could not find Transformer layers to hook.")

        if hasattr(self.model, "visual"):
            h2 = self.model.visual.register_forward_hook(get_output_activation("visual_encoder"))
            hook_handles.append(h2)
        elif hasattr(self.model, "thinker") and hasattr(self.model.thinker, "visual"):
            h2 = self.model.thinker.visual.register_forward_hook(get_output_activation("visual_encoder"))
            hook_handles.append(h2)

        embed_tokens = None
        if hasattr(self.model, "thinker") and hasattr(self.model.thinker, "model") and hasattr(self.model.thinker.model, "embed_tokens"):
            embed_tokens = self.model.thinker.model.embed_tokens
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embed_tokens = self.model.model.embed_tokens
        elif hasattr(self.model, "embed_tokens"):
            embed_tokens = self.model.embed_tokens

        if embed_tokens is not None:
            h3 = embed_tokens.register_forward_hook(get_output_activation("text_embed"))
            hook_handles.append(h3)
        else:
            print("Warning: Could not find embed_tokens to hook.")

        return hook_handles, activations

    def _process_and_save_pca(self, activations, sample_item, output_dir, quant_method):
        if len(activations["thinker_input"]) == 0:
            return

        x_mixed = np.concatenate(activations["thinker_input"], axis=0)

        final_output_dir = os.path.join(output_dir, quant_method)
        os.makedirs(final_output_dir, exist_ok=True)

        uniq_id = sample_item.get("id", "sample")

        max_fit_samples = 5000
        if x_mixed.shape[0] > max_fit_samples:
            indices = np.random.choice(x_mixed.shape[0], max_fit_samples, replace=False)
            x_fit = x_mixed[indices]
        else:
            x_fit = x_mixed

        pca = PCA(n_components=2)
        pca.fit(x_fit)
        x_mixed_pca = pca.transform(x_mixed)

        data_save_path = os.path.join(final_output_dir, f"pca_data_{uniq_id}.npz")
        save_dict = {
            "mixed": x_mixed,
            "mixed_pca": x_mixed_pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
        if len(activations["visual_encoder"]) > 0:
            save_dict["visual_encoder"] = np.concatenate(activations["visual_encoder"], axis=0)
        if len(activations["text_embed"]) > 0:
            save_dict["text_embed"] = np.concatenate(activations["text_embed"], axis=0)
        np.savez(data_save_path, **save_dict)

        plt.close("all")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.size": 24,
                "axes.linewidth": 2,
                "xtick.major.width": 2,
                "ytick.major.width": 2,
                "xtick.labelsize": 24,
                "ytick.labelsize": 24,
                "axes.labelsize": 24,
                "legend.fontsize": 24,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
            }
        )

        plt.figure(figsize=(8, 8))
        plt.scatter(
            x_mixed_pca[:, 0],
            x_mixed_pca[:, 1],
            alpha=0.25,
            color="#1f77b4",
            label="Visual Tokens",
            s=40,
            edgecolors="none",
            rasterized=True,
        )

        if len(activations["text_embed"]) > 0:
            x_text = np.concatenate(activations["text_embed"], axis=0)
            if x_text.shape[1] == x_mixed.shape[1]:
                x_text_pca = pca.transform(x_text)
                plt.scatter(
                    x_text_pca[:, 0],
                    x_text_pca[:, 1],
                    alpha=1.0,
                    color="#d62728",
                    label="Text Tokens",
                    marker="*",
                    s=200,
                    edgecolors="white",
                    linewidth=0.5,
                )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(frameon=False, fancybox=False, edgecolor="black", framealpha=1.0, loc="best")

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        save_path = os.path.join(final_output_dir, f"pca_analysis_{uniq_id}_{quant_method}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()

    def _resolve_subjects(self):
        if self.target_subject.lower() != "all":
            return [self.target_subject]

        try:
            return get_dataset_config_names("MMMU/MMMU", trust_remote_code=True)
        except Exception as e:
            print(f"警告：无法在线获取 subject 列表 ({e})，使用硬编码列表。")
            return [
                "Accounting",
                "Agriculture",
                "Architecture_and_Engineering",
                "Art",
                "Art_Theory",
                "Basic_Medical_Science",
                "Biology",
                "Chemistry",
                "Clinical_Medicine",
                "Computer_Science",
                "Design",
                "Diagnostics_and_Laboratory_Medicine",
                "Economics",
                "Electronics",
                "Energy_and_Power",
                "Finance",
                "Geography",
                "History",
                "Literature",
                "Manage",
                "Marketing",
                "Materials",
                "Math",
                "Mechanical_Engineering",
                "Music",
                "Pharmacy",
                "Physics",
                "Psychology",
                "Public_Health",
                "Sociology",
            ]

    def _load_subject_dataset(self, subject):
        dataset_root = os.path.join(self.dataset_dir, "MMMU___mmmu")
        subject_path = os.path.join(dataset_root, subject)

        print(f"Loading dataset from: {subject_path} (Local only)")
        dataset = None

        try:
            arrow_files = glob.glob(os.path.join(subject_path, "**", "*.arrow"), recursive=True)
            if arrow_files:
                print(f"Strategy 0: Detected {len(arrow_files)} .arrow files.")
                val_files = [f for f in arrow_files if "validation" in f]
                if not val_files:
                    print("  - No explicit 'validation' split in filenames. Using all found arrow files.")
                    val_files = arrow_files
                if val_files:
                    print(f"  - Loading via 'arrow' builder with {len(val_files)} files...")
                    dataset = load_dataset("arrow", data_files={"validation": val_files}, split="validation")
        except Exception as e:
            print(f"Strategy 0 (Arrow) failed: {e}")

        if dataset is None:
            try:
                parquet_files = glob.glob(os.path.join(subject_path, "**", "*.parquet"), recursive=True)
                if parquet_files:
                    val_files = [f for f in parquet_files if "validation" in os.path.basename(f)]
                    if not val_files:
                        val_files = parquet_files
                    if val_files:
                        print(f"Strategy 1: Detected {len(val_files)} parquet files. Loading...")
                        dataset = load_dataset("parquet", data_files={"validation": val_files}, split="validation")
            except Exception as e:
                print(f"Strategy 1 (Parquet) failed: {e}")

        if dataset is None:
            try:
                from datasets import load_from_disk

                if os.path.exists(os.path.join(subject_path, "dataset_info.json")):
                    print("  Detected Arrow cache, using load_from_disk...")
                    ds_dict = load_from_disk(subject_path)
                    if "validation" in ds_dict:
                        dataset = ds_dict["validation"]
                    else:
                        dataset = ds_dict
            except Exception:
                pass

        if dataset is None:
            print(f"Error: Failed to load subject '{subject}'. Check if path exists and contains .parquet/.json files.")

        return dataset

    def _example_to_qwen_calibration_conversation(self, example):
        """Build Qwen-style multimodal conversation for Morpho calibrate/search (non-InternVL)."""
        prompt = self._build_prompt(example)
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer the following question based on the provided information.",
                    }
                ],
            },
        ]
        # img_obj = example.get("image_1", None)
        # if img_obj:
        #     conversation[1]["content"].append({"type": "image", "image": img_obj})
        # 【修复 Bug 3】: 将所有图片全塞进 Qwen 对话里！
        image_entries = extract_image_entries(example)
        for img_obj in image_entries:
            if img_obj:
                conversation[1]["content"].append({"type": "image", "image": img_obj})
        conversation[1]["content"].append({"type": "text", "text": prompt})
        return conversation

    def _build_mmmu_stratified_quant_pool(self, target_n: int):
        """
        Round-robin across MMMU subjects so calibrate/search pools are not dominated by
        the first few categories in subject list order.

        Uses lazy per-subject loading: datasets are loaded one at a time and only
        iterated until enough examples have been collected, avoiding full-materialisation
        of all 30 subjects when only a small calibration/search set is needed.
        """
        if target_n <= 0:
            return []
        subjects = self._resolve_subjects()

        pool: list = []
        # Per-subject state: None = not loaded yet / failed / exhausted
        # tuple = (dataset, next_index)
        subject_state: dict = {s: None for s in subjects}
        rounds_without_progress = 0

        while len(pool) < target_n and rounds_without_progress < 2:
            progressed_this_round = False
            for subject in subjects:
                if len(pool) >= target_n:
                    break

                state = subject_state[subject]
                # Sentinel: already exhausted or failed → skip
                if state is False:
                    continue

                if state is None:
                    # First access — load dataset on demand
                    dataset = self._load_subject_dataset(subject)
                    if dataset is None or len(dataset) == 0:
                        subject_state[subject] = False  # mark exhausted
                        continue
                    state = (dataset, 0)
                    subject_state[subject] = state

                dataset, ptr = state
                # Find the next valid example from this subject
                found = False
                while ptr < len(dataset):
                    example = dataset[ptr]
                    ptr += 1
                    subject_state[subject] = (dataset, ptr)

                    if self._is_internvl():
                        entries = [self._resolve_image_value(e) for e in extract_image_entries(example)]
                        entries = [e for e in entries if e is not None]
                        if not entries:
                            continue
                        pool.append(example)
                    else:
                        pool.append(self._example_to_qwen_calibration_conversation(example))
                    found = True
                    break

                if not found:
                    # Exhausted this subject — no more valid examples
                    subject_state[subject] = False

                if found:
                    progressed_this_round = True

            if not progressed_this_round:
                rounds_without_progress += 1

        n_subj_used = sum(1 for v in subject_state.values() if v is not None and v is not False)
        print(
            f"MMMU 量化样本: 按 {n_subj_used} 个学科 round-robin 分层，池长 {len(pool)}/{target_n}"
            + ("（样本不足已用尽）" if len(pool) < target_n else "")
        )
        return pool

    def _batch_generate_for_quant_search(self, conversations, *, rearm_quantact_search_each_sample: bool = False):
        if len(conversations) == 0:
            return

        if self._is_internvl():
            for example in conversations:
                if rearm_quantact_search_each_sample:
                    for _, module in self.model.named_modules():
                        if isinstance(module, QuantAct):
                            module.set_search(search=True)
                with torch.no_grad():
                    self._inference_internvl(example, max_new_tokens=1, quant_stats_pass=True)
            return

        batch_texts = []
        batch_images = []
        batch_audios = []
        batch_videos = []

        for conversation in conversations:
            if getattr(self.config.quant, "calib_without_audio", False):
                for msg in conversation:
                    if msg.get("role") == "user" and "content" in msg:
                        msg["content"] = [c for c in msg["content"] if c.get("type") != "audio"]
            
            if getattr(self.config.quant, "calib_without_video", False):
                for msg in conversation:
                    if msg.get("role") == "user" and "content" in msg:
                        msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]
            
            text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
            if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
                assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
                text = text[0]
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

            batch_texts.append(text)
            if images:
                batch_images.extend(images)
            if audios:
                batch_audios.extend(audios)
            if videos:
                batch_videos.extend(videos)

        inputs = self.processor(
            text=batch_texts,
            images=batch_images if len(batch_images) > 0 else None,
            audio=batch_audios if len(batch_audios) > 0 else None,
            videos=batch_videos if len(batch_videos) > 0 else None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        if getattr(self.config.quant, "simulate_mbq", False):
            from modules.mbq_layers import set_mbq_modality
            modality = "multimodal" if (batch_images or batch_audios or batch_videos) else "text"
            set_mbq_modality(self.model, modality)
        if getattr(self.config.quant, "simulate_mquant", False) and len(conversations) == 1:
            self._set_mquant_ctx(conversations[0], inputs)
        with torch.no_grad():
            self.model.generate(**inputs, thinker_max_new_tokens=1, use_audio_in_video=False)

    def _collect_quant_samples(self, skip_count, sample_count):
        if sample_count <= 0:
            return []

        need = skip_count + sample_count
        pool = self._build_mmmu_stratified_quant_pool(need)
        if skip_count >= len(pool):
            return []
        return pool[skip_count : skip_count + sample_count]

    def calibrate(self):
        do_smoothquant = getattr(self.config.quant, 'simulate_smoothquant', False)
        do_awq = getattr(self.config.quant, 'simulate_awq', False)
        do_mbq = getattr(self.config.quant, 'simulate_mbq', False)
        do_mquant = getattr(self.config.quant, 'simulate_mquant', False)
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)

        if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8", "morpho_withhif4") and not do_smoothquant and not do_awq and not do_mbq and not do_mquant and not do_qlora_act:
            print(f"{self.config.model.quant_method}模式：跳过校准步骤。")
            return

        print("\n==> start batched calibrate")
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=True)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        if do_smoothquant:
            from modules.smoothquant_layers import set_smoothquant_observe
            set_smoothquant_observe(self.model, enabled=True)

        if do_awq:
            from modules.awq_layers import set_awq_observe
            set_awq_observe(self.model, enabled=True)

        if do_mbq:
            from modules.mbq_layers import set_mbq_observe
            set_mbq_observe(self.model, enabled=True)

        if do_mquant:
            from modules.mquant_layers import set_mquant_observe
            set_mquant_observe(self.model, enabled=True)

        calib_samples = self._collect_quant_samples(0, self.config.quant.calib_size)
        batch_size = max(1, self.config.quant.batch_size)
        print(f"Collected {len(calib_samples)} calibration samples.")

        for idx in tqdm(range(0, len(calib_samples), batch_size)):
            batch = calib_samples[idx : idx + batch_size]
            self._batch_generate_for_quant_search(batch)

        print("==> end calibrate")

        # 校准结束后计算 Dispersion Score，用于离群点检测
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                if hasattr(module, 'compute_dispersion_score'):
                    try:
                        module.compute_dispersion_score()
                        if hasattr(module, 'dispersion_score') and module.dispersion_score is not None:
                            # print(f"  [Dispersion] Layer {getattr(module, 'layer_name', 'Unknown')}: max score={module.dispersion_score.max().item():.4f}")
                            pass
                        else:
                            # print(f"  [Dispersion] Layer {getattr(module, 'layer_name', 'Unknown')}: skipped (no data)")
                            pass
                    except Exception as e:
                        # print(f"  [Dispersion] Layer {getattr(module, 'layer_name', 'Unknown')}: error ({type(e).__name__})")
                        pass

        if do_smoothquant:
            from modules.smoothquant_layers import finalize_smoothquant
            finalize_smoothquant(self.model, alpha=getattr(self.config.quant, 'smoothquant_alpha', 0.5))

        if do_awq:
            from modules.awq_layers import finalize_awq
            finalize_awq(self.model)

        if do_mbq:
            from modules.mbq_layers import finalize_mbq
            finalize_mbq(self.model)

        if do_mquant:
            from modules.mquant_layers import finalize_mquant
            finalize_mquant(self.model)
            print("[MQuant] 模态特化 scale 计算完成，权重已量化。")

    # def prepare(self):
    #     if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8"):
    #         return

    #     print("\n==> start prepare for inference")
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, QuantAct):
    #             module.set_search(search=True)
    #             module.set_calibrate(calibrate=False)
    #             if hasattr(module, 'set_sparse_buffer_ratio'):
    #                 module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))
    #             if hasattr(module, "dispersion_score") and module.dispersion_score is not None:
    #                 dispersion_score = module.dispersion_score
    #                 total_channels = dispersion_score.numel()
    #                 # 所有层统一按 threshold 确定离群补偿范围
    #                 # if _quantact_use_full_channel_outlier_mask(self.config, getattr(module, "layer_name", "") or ""):
    #                 #     outlier_mask = torch.ones_like(dispersion_score, dtype=torch.bool)
    #                 # else:
    #                 mean_ds = dispersion_score.mean()
    #                 std_ds = dispersion_score.std()
    #                 outlier_std_threshold = getattr(self.config.quant, 'outlier_std_threshold', 2.0)
    #                 threshold = mean_ds + outlier_std_threshold * std_ds
    #                 outlier_mask = dispersion_score >= threshold
    #                 num_outliers = outlier_mask.sum().item()
    #                 # print(f"Layer {getattr(module, 'layer_name', name)}: {num_outliers}/{total_channels} channels marked as outliers")
    #                 module.register_buffer("outlier_mask", outlier_mask)

    #     search_samples = self._collect_quant_samples(self.config.quant.calib_size, self.config.quant.search_size)
    #     print(f"Collected {len(search_samples)} search samples.")
    #     batch_size = max(1, self.config.quant.batch_size)
    #     for idx in range(0, len(search_samples), batch_size):
    #         batch = search_samples[idx : idx + batch_size]
    #         self._batch_generate_for_quant_search(
    #             batch,
    #             rearm_quantact_search_each_sample=self._is_internvl(),
    #         )

    #     for _, module in self.model.named_modules():
    #         if isinstance(module, QuantAct):
    #             module.set_search(search=False)
    #             if hasattr(module, 'set_sparse_buffer_ratio'):
    #                 module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

    #     print("==> end prepare for inference")

    # def prepare(self):
    #     if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8"):
    #         return

    #     print("\n==> start prepare for inference")
        
    #     # =====================================================================
    #     # [STAGE 1]: 收集全视野的绝对幅值，计算全局上帝视角阈值 (Global Quantile)
    #     # =====================================================================
    #     all_dispersion_scores = []
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, QuantAct):
    #             # 状态初始化
    #             module.set_search(search=True)
    #             module.set_calibrate(calibrate=False)
    #             if hasattr(module, 'set_sparse_buffer_ratio'):
    #                 module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))
                
    #             # 收集该层分数（安全起见，转为 CPU float 避免多卡 device mismatch）
    #             if hasattr(module, "dispersion_score") and module.dispersion_score is not None:
    #                 all_dispersion_scores.append(
    #                     module.dispersion_score.detach().cpu().float().flatten()
    #                 )
        
    #     # 计算全局统一阈值
    #     global_threshold = 0.0
    #     if len(all_dispersion_scores) > 0:
    #         global_scores_tensor = torch.cat(all_dispersion_scores)
    #         # 全局离群值比例，如果没有配置默认取 Top 5%
    #         global_outlier_ratio = getattr(self.config.quant, 'global_outlier_ratio', 0.05)
    #         # quantile 需要 float32 数据类型
    #         global_threshold = torch.quantile(global_scores_tensor, 1.0 - global_outlier_ratio).item()
    #         print(f"[*] Computed Global Dispersion Threshold: {global_threshold:.4f} (Top {global_outlier_ratio*100}%)")

    #     # =====================================================================
    #     # [STAGE 2]: 使用全局阈值，对全模型执行绝对幅值压制，生成 Outlier Mask
    #     # =====================================================================
    #     total_marked_outliers = 0
    #     total_channels_all = 0
        
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, QuantAct) and hasattr(module, "dispersion_score") and module.dispersion_score is not None:
    #             dispersion_score = module.dispersion_score
    #             total_channels = dispersion_score.numel()
    #             total_channels_all += total_channels
                
    #             # 直接用全局标量阈值切一刀，彻底尊重量级的绝对大小！
    #             outlier_mask = dispersion_score >= global_threshold
    #             num_outliers = outlier_mask.sum().item()
    #             total_marked_outliers += num_outliers
                
    #             # 选看：你可以取消下面这行注释，观察哪些大魔王层拿走了最多的预算
    #             # print(f"Layer {getattr(module, 'layer_name', name)}: {num_outliers}/{total_channels} channels marked as outliers")
                
    #             module.register_buffer("outlier_mask", outlier_mask)
        
    #     if total_channels_all > 0:
    #         print(f"[*] Total Outliers Marked: {total_marked_outliers}/{total_channels_all} ({(total_marked_outliers/total_channels_all)*100:.2f}%)")

    #     # =====================================================================
    #     # [STAGE 3]: 运行 Search 过程 (原逻辑不变，但现在有了完美的 Mask 保护)
    #     # =====================================================================
    #     search_samples = self._collect_quant_samples(self.config.quant.calib_size, self.config.quant.search_size)
    #     print(f"\nCollected {len(search_samples)} search samples.")
    #     batch_size = max(1, self.config.quant.batch_size)
    #     for idx in range(0, len(search_samples), batch_size):
    #         batch = search_samples[idx : idx + batch_size]
    #         self._batch_generate_for_quant_search(
    #             batch,
    #             rearm_quantact_search_each_sample=self._is_internvl(),
    #         )

    #     # =====================================================================
    #     # [STAGE 4]: 清理与恢复状态 (原逻辑不变)
    #     # =====================================================================
    #     for _, module in self.model.named_modules():
    #         if isinstance(module, QuantAct):
    #             module.set_search(search=False)
    #             if hasattr(module, 'set_sparse_buffer_ratio'):
    #                 module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

    #     print("==> end prepare for inference\n")

    # def prepare(self):
    #     if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8"):
    #         return

    #     print("\n==> start prepare for inference")
        
    #     # =====================================================================
    #     # [STAGE 1]: 收集全局幅值，计算“绝对幅值底线 (Global Floor)”
    #     # =====================================================================
    #     all_dispersion_scores = []
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, QuantAct):
    #             module.set_search(search=True)
    #             module.set_calibrate(calibrate=False)
    #             if hasattr(module, 'set_sparse_buffer_ratio'):
    #                 module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))
                
    #             if hasattr(module, "dispersion_score") and module.dispersion_score is not None:
    #                 # 拉到 CPU 计算全局分布，防多卡报错
    #                 all_dispersion_scores.append(
    #                     module.dispersion_score.detach().cpu().float().flatten()
    #                 )
        
    #     global_floor = 0.0
    #     if len(all_dispersion_scores) > 0:
    #         global_scores_tensor = torch.cat(all_dispersion_scores)
    #         # 【核心策略】：这里使用一个较宽松的比例作为“底线”，比如 Top 15% 或 20%
    #         # 含义：连全模型前 15% 都排不进去的数值，绝对没有资格当离群值！
    #         global_ratio_floor = getattr(self.config.quant, 'global_ratio_floor', 0.05) 
    #         global_floor = torch.quantile(global_scores_tensor, 1.0 - global_ratio_floor).item()
    #         print(f"[*] Computed Global Absolute Floor: {global_floor:.4f} (Top {global_ratio_floor*100}%)")

    #     # =====================================================================
    #     # [STAGE 2]: 层内相对分布 + 全局绝对幅值 双重过滤
    #     # =====================================================================
    #     total_marked_outliers = 0
    #     total_channels_all = 0
        
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, QuantAct) and hasattr(module, "dispersion_score") and module.dispersion_score is not None:
    #             dispersion_score = module.dispersion_score
    #             total_channels = dispersion_score.numel()
    #             total_channels_all += total_channels
                
    #             # 1. 计算层内的相对阈值 (保留原始的局部分布感知)
    #             mean_ds = dispersion_score.mean()
    #             std_ds = dispersion_score.std()
    #             outlier_std_threshold = getattr(self.config.quant, 'outlier_std_threshold', 2.0)
    #             local_threshold = mean_ds + outlier_std_threshold * std_ds
                
    #             # 2. 【神级融合】：用 clamp 将全局底线与局部阈值融合
    #             # 最终阈值 = max(层内局部阈值, 全局绝对底线)
    #             # final_threshold = torch.clamp(local_threshold, min=global_floor)
    #             global_floor_tensor = torch.tensor(global_floor, device=dispersion_score.device, dtype=dispersion_score.dtype)
    #             final_threshold = torch.minimum(local_threshold, global_floor_tensor)
                
    #             outlier_mask = dispersion_score >= final_threshold
                
    #             num_outliers = outlier_mask.sum().item()
    #             total_marked_outliers += num_outliers
                
    #             module.register_buffer("outlier_mask", outlier_mask)
        
    #     if total_channels_all > 0:
    #         print(f"[*] Total Outliers Marked: {total_marked_outliers}/{total_channels_all} ({(total_marked_outliers/total_channels_all)*100:.2f}%)")

    #     # =====================================================================
    #     # [STAGE 3]: 运行 Search 过程
    #     # =====================================================================
    #     search_samples = self._collect_quant_samples(self.config.quant.calib_size, self.config.quant.search_size)
    #     print(f"\nCollected {len(search_samples)} search samples.")
    #     batch_size = max(1, self.config.quant.batch_size)
    #     for idx in range(0, len(search_samples), batch_size):
    #         batch = search_samples[idx : idx + batch_size]
    #         self._batch_generate_for_quant_search(
    #             batch,
    #             rearm_quantact_search_each_sample=self._is_internvl(),
    #         )

    #     # =====================================================================
    #     # [STAGE 4]: 清理与恢复状态
    #     # =====================================================================
    #     for _, module in self.model.named_modules():
    #         if isinstance(module, QuantAct):
    #             module.set_search(search=False)
    #             if hasattr(module, 'set_sparse_buffer_ratio'):
    #                 module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

    #     print("==> end prepare for inference\n")
    def prepare(self):
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)
        if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8", "morpho_withhif4") and not do_qlora_act:
            return

        print("\n==> start prepare for inference")
        self._configure_quantact_for_prepare()
        # if len(plot_layer_names) > 0:
        #     import matplotlib.pyplot as plt
        #     import numpy as np
            
        #     print(f"[*] Drawing Dispersion Score Distribution for {len(plot_layer_names)} layers...")
            
        #     # 设置画布大小（层数多的话画布要够宽才不挤）
        #     plt.figure(figsize=(24, 8))
        #     x_pos = np.arange(len(plot_layer_names))
            
        #     # 画条形图：均值作为高度，标准差作为 error bar
        #     plt.bar(x_pos, plot_means, yerr=plot_stds, align='center', alpha=0.7, ecolor='red', capsize=3, color='steelblue')
            
        #     plt.ylabel('Dispersion Score (Dc)', fontsize=14)
        #     plt.xlabel('Layer Name', fontsize=14)
        #     plt.title('Distribution of Dispersion Scores Across Layers', fontsize=16)
            
        #     # 旋转 x 轴标签防遮挡，字体设小一点
        #     plt.xticks(x_pos, plot_layer_names, rotation=90, fontsize=6)
        #     plt.grid(axis='y', linestyle='--', alpha=0.5)
            
        #     # 自动调整布局防截断
        #     plt.tight_layout()
            
        #     # 存成高分辨率图片到当前目录
        #     plot_filename = "dispersion_distribution.pdf" # PDF格式放大不失真
        #     plt.savefig(plot_filename, dpi=300)
        #     plt.close()

        disable_boundary_coopt = getattr(self.config.quant, 'disable_boundary_cooptimization', False)

        if not disable_boundary_coopt:
            search_samples = self._collect_quant_samples(self.config.quant.calib_size, self.config.quant.search_size)
            print(f"Collected {len(search_samples)} search samples.")
            batch_size = max(1, self.config.quant.batch_size)
            for idx in range(0, len(search_samples), batch_size):
                print("Search Batch:", int(idx / batch_size + 1))
                batch = search_samples[idx : idx + batch_size]
                self._batch_generate_for_quant_search(
                    batch,
                    rearm_quantact_search_each_sample=self._is_internvl(),
                )

        self._finalize_quantact_prepare()

        print("==> end prepare for inference")


        mode_suffix = ""
        if self.config.quant.simulate_hif8:
            mode_suffix = "hif8_sim"
        if self.config.data.num_samples != -1:
            mode_suffix += f"_{self.config.data.num_samples}samples"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix += f"_{timestamp}"
        return mode_suffix

    def _evaluate_subject(self, subject, mode_suffix, save_dir):
        dataset = self._load_subject_dataset(subject)
        if dataset is None:
            return None

        correct = 0
        total = 0

        # Ensure QuantAct modules are in eval mode (calibrate=False, search=False)
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=False)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        print(f"Evaluating subject: {subject} ({len(dataset)} samples)...")
        limit = self.config.data.num_samples if self.config.data.num_samples != -1 else len(dataset)

        # Streaming write: open subject CSV once and write rows as produced
        subject_filename = f"{subject}_{mode_suffix}.csv"
        subject_csv_path = os.path.join(save_dir, subject_filename)
        keys = ["id", "subject", "question", "answer", "prediction", "is_correct", "raw_output"]
        f = open(subject_csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        try:
            for i in tqdm(range(min(limit, len(dataset)))):
                example = dataset[i]
                hook_handles = []
                activations = {}
                if self.run_pca:
                    hook_handles, activations = self._register_pca_hooks()

                try:
                    if self._is_internvl():
                        try:
                            output = self._inference_internvl(example)
                        except device_utils.oom_errors() as _oom:
                            if not device_utils.is_oom(_oom):
                                raise
                            device_utils.empty_cache()
                            sid = example.get("id", "unknown")
                            print(f"  Device OOM on {sid}; retry once with stricter vision (1 image, 4 tiles).")
                            output = self._inference_internvl(
                                example,
                                memory_override={"max_images": 1, "max_num_tiles": 4},
                            )
                    else:
                        prompt = self._build_prompt(example)

                        conversation = [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Answer the following question based on the provided information.",
                                    }
                                ],
                            },
                        ]

                        image_entries = extract_image_entries(example)
                        for img_obj in image_entries:
                            if img_obj:
                                conversation[1]["content"].append({"type": "image", "image": img_obj})
                        conversation[1]["content"].append({"type": "text", "text": prompt})

                        output = self._inference(conversation)
                    pred = self._extract_assistant_response(output)

                    if self.run_pca:
                        self._process_and_save_pca(activations, example, save_dir, self.config.model.quant_method)
                        for h in hook_handles:
                            h.remove()
                except Exception as e:
                    import traceback

                    sample_id = example.get("id", "unknown")
                    print(f"Error processing sample {i} ({sample_id}) in {subject}: {e}")
                    print(f"  Full traceback:\n{traceback.format_exc()}")
                    pred = "Error"
                    output = str(e)
                    if self.run_pca:
                        for h in hook_handles:
                            h.remove()

                answer = example.get("answer", "")
                is_correct = pred == answer
                if is_correct:
                    correct += 1
                total += 1

                writer.writerow({
                    "id": example["id"],
                    "subject": subject,
                    "question": example["question"],
                    "answer": answer,
                    "prediction": pred,
                    "is_correct": is_correct,
                    "raw_output": output,
                })
                # Periodic flush for crash-safety
                if total % 10 == 0:
                    f.flush()
        finally:
            f.close()

        device_utils.empty_cache()
        acc = correct / total if total > 0 else 0
        print(f"Subject: {subject} | Acc: {acc:.4f} ({correct}/{total})")

        return {"subject": subject, "acc": acc, "total": total, "correct": correct}

    def evaluate(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        mode_suffix = self._build_mode_suffix()
        subjects = self._resolve_subjects()

        print(f"Target Subjects: {len(subjects)}")
        agg_results = []

        for subject in subjects:
            res = self._evaluate_subject(subject, mode_suffix, save_dir)
            if res:
                agg_results.append(res)
            else:
                print(f"Skipping subject {subject} due to loading issues.")

        total_samples = sum(r["total"] for r in agg_results)
        total_correct = sum(r["correct"] for r in agg_results)
        avg_acc = total_correct / total_samples if total_samples > 0 else 0

        print(f"\n{'='*20} MMMU Evaluation Summary {'='*20}")
        print(f"Quant Method   : {self.config.model.quant_method}")
        print(f"Total Samples  : {total_samples}")
        print(f"Correct        : {total_correct}")
        print(f"Accuracy       : {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        for r in agg_results:
            print(f"  {r['subject']:<35} : {r['acc']:.4f}")
        print(f"{'='*70}")
        self._print_config_yaml()

        summary_path = os.path.join(save_dir, f"summary_report_{mode_suffix}.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "acc", "total", "correct", "quant_method"])
            writer.writeheader()

            for r in agg_results:
                row = dict(r)
                row["quant_method"] = self.config.model.quant_method
                writer.writerow(row)
            writer.writerow(
                {
                    "subject": "AVERAGE",
                    "acc": avg_acc,
                    "total": total_samples,
                    "correct": total_correct,
                    "quant_method": self.config.model.quant_method,
                }
            )

        print(f"Results saved to {save_dir}")
        return avg_acc

def load_videomme_dataset(data_dir):
    import pandas as pd
    import json
    import os
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    if parquet_files:
        print(f"Loading Parquet file: {parquet_files[0]}")
        df = pd.read_parquet(os.path.join(data_dir, parquet_files[0]))
        return df.to_dict('records')
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and 'test' in f]
    if json_files:
        print(f"Loading JSON file: {json_files[0]}")
        with open(os.path.join(data_dir, json_files[0]), 'r') as f:
            return json.load(f)
    for root, _, files in os.walk(data_dir):
         for f in files:
             if f.endswith('.parquet'):
                 return pd.read_parquet(os.path.join(root, f)).to_dict('records')
    raise FileNotFoundError("Video-MME dataset not found.")

def build_videomme_prompt(question, options):
    prompt = f"Question: {question}\nOptions:\n"
    for idx, opt in enumerate(options):
        prompt += f"({chr(65+idx)}) {opt}\n"
    prompt += "Answer with the option letter directly (e.g., A, B, C, D)."
    return prompt

class VideoMMEDataset(Dataset):
    def __init__(self, base_dir="/private/wy/datasets/Video-MME", max_samples=None, skip_samples=0, num_frames=32):
        self.base_dir = base_dir
        self.records = load_videomme_dataset(self.base_dir)
        self.num_frames = num_frames

        if skip_samples > 0:
            self.records = self.records[skip_samples:]
        if max_samples is not None:
            self.records = self.records[:max_samples]

        # Pre-compute video path cache: one os.walk instead of per-sample walk.
        # Search expected location first, then fall back to full tree scan.
        self._video_path_cache = {}
        expected_dir = os.path.join(self.base_dir, "videos", "data")
        search_roots = [expected_dir] if os.path.isdir(expected_dir) else []
        # Always include base_dir as fallback for any videos not in expected_dir
        search_roots.append(self.base_dir)
        seen_dirs = set()
        for root_dir in search_roots:
            for walk_root, _, files in os.walk(root_dir):
                real = os.path.realpath(walk_root)
                if real in seen_dirs:
                    continue
                seen_dirs.add(real)
                for f in files:
                    if f.endswith('.mp4'):
                        # Prefer first-found (expected_dir wins over base_dir fallback)
                        if f not in self._video_path_cache:
                            self._video_path_cache[f] = os.path.join(walk_root, f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        video_id = item.get('video_id', f'sample_{idx}')
        video_filename = item.get('videoID', f"{video_id}.mp4")
        question = item['question']
        options = item['options']
        answer_gt = item.get('answer', "")

        # Fix: avoid .mp4.mp4 double suffix
        video_name = video_filename if video_filename.endswith('.mp4') else f"{video_filename}.mp4"

        # O(1) cache lookup instead of per-sample os.walk
        video_path = self._video_path_cache.get(video_name)

        prompt = build_videomme_prompt(question, options)

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        if video_path:
            conversation[1]["content"].insert(0, {"type": "video", "video": video_path, "nframes": self.num_frames})

        return {
            "pid": video_id,
            "question": question,
            "answer": answer_gt,
            "options": options,
            "conversation": conversation,
            "video_path": video_path
        }

def collate_fn_videomme(batch):
    pids = [item["pid"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    conversations = [item["conversation"] for item in batch]
    return {
        "pids": pids,
        "questions": questions,
        "answers": answers,
        "conversations": conversations
    }

class VideoMMEEvaluator(BaseEvaluator):
    def __init__(self, model, processor, config, base_dir, data_dir="/private/wy/datasets/Video-MME"):
        super().__init__(model, processor, config, base_dir)
        self.data_dir = data_dir
        self.num_frames = getattr(config.data, "videomme_num_frames", 32)

    def _extract_assistant_response(self, text):
        response = text
        if "assistant" in text.lower():
            parts = text.split("assistant", 1)
            if len(parts) > 1:
                response = parts[1].strip()

        # 更严谨的正则匹配。先找 (A) 或 A. 这种，如果没有，从后往前找单独的字母。
        match = re.search(r'(?:\(([A-D])\)|([A-D])\.)', response)
        if match:
            return match.group(1) or match.group(2)
        
        # 兜底：找独立的 A-D，尽量避免匹配单词首字母
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)

        # 终极兜底：如果就是不加标点，取最后一个出现的大写字母 A-D
        matches = re.findall(r'[A-D]', response)
        if matches:
            return matches[-1]

        return "Failed"

    def _inference(self, conversation):
        text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
        if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
            assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
            text = text[0]
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        self._set_mquant_ctx(conversation, inputs)

        with torch.no_grad():
            text_ids = self.model.generate(**inputs, use_audio_in_video=False, thinker_max_new_tokens=32)

        text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return text[0]

    def _extract_video_frames(self, video_path, num_frames=8):
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames

    def _inference_internvl(self, data, max_new_tokens=32, *, quant_stats_pass: bool = False, memory_override=None):
        ovr = memory_override if isinstance(memory_override, dict) else {}

        num_frames = ovr.get("num_frames")
        if num_frames is None:
            num_frames = 8 if quant_stats_pass else 16

        tile_cap = ovr.get("max_num_tiles")
        if tile_cap is None:
            tile_cap = getattr(
                self.config.data,
                "internvl_calib_max_num_tiles" if quant_stats_pass else "internvl_eval_max_num_tiles",
                None,
            )
        if tile_cap is None:
            tile_cap = getattr(self.config.data, "internvl_max_num_tiles", None)
        tile_cap = int(tile_cap) if tile_cap is not None else 1

        video_path = data.get("video_path")
        frames = self._extract_video_frames(video_path, num_frames=num_frames) if video_path else []
        if not frames:
            raise ValueError(f"VideoMME sample {data.get('pid', 'unknown')} has no readable video frames.")

        pixel_values_list = [load_internvl_image(frame, max_num=tile_cap) for frame in frames]
        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [item.size(0) for item in pixel_values_list]

        model_device = next(self.model.parameters()).device
        model_dtype = getattr(self.model, "dtype", torch.bfloat16)
        pixel_values = pixel_values.to(model_device).to(model_dtype)

        video_prefix = ''.join([f"Frame{idx + 1}: <image>\n" for idx in range(len(frames))])
        prompt = video_prefix + build_videomme_prompt(data["question"], data["options"])
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

        try:
            response = self.model.chat(
                self.processor,
                pixel_values,
                prompt,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        finally:
            del pixel_values_list
            del pixel_values

        if isinstance(response, tuple):
            return response[0]
        return response

    def calibrate(self):
        do_smoothquant = getattr(self.config.quant, 'simulate_smoothquant', False)
        do_awq = getattr(self.config.quant, 'simulate_awq', False)
        do_mbq = getattr(self.config.quant, 'simulate_mbq', False)
        do_mquant = getattr(self.config.quant, 'simulate_mquant', False)
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)

        if self.config.model.quant_method not in ('morpho', 'morpho_withhif8', 'morpho_withhif4') and not do_smoothquant and not do_awq and not do_mbq and not do_mquant and not do_qlora_act:
            print(f"{self.config.model.quant_method}模式：跳过校准步骤。")
            return

        print('\n==> start batched calibrate')
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=True)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        if do_smoothquant:
            from modules.smoothquant_layers import set_smoothquant_observe
            set_smoothquant_observe(self.model, enabled=True)

        if do_awq:
            from modules.awq_layers import set_awq_observe
            set_awq_observe(self.model, enabled=True)

        if do_mbq:
            from modules.mbq_layers import set_mbq_observe
            set_mbq_observe(self.model, enabled=True)

        if do_mquant:
            from modules.mquant_layers import set_mquant_observe
            set_mquant_observe(self.model, enabled=True)

        dataset = VideoMMEDataset(
            base_dir=self.data_dir,
            max_samples=self.config.quant.calib_size,
            num_frames=self.num_frames
        )

        if self._is_internvl():
            for idx in tqdm(range(len(dataset))):
                sample = dataset[idx]
                with torch.no_grad():
                    self._inference_internvl(sample, max_new_tokens=1, quant_stats_pass=True)

            for _, module in self.model.named_modules():
                if isinstance(module, QuantAct):
                    module.compute_dispersion_score()

            if do_smoothquant:
                from modules.smoothquant_layers import finalize_smoothquant
                finalize_smoothquant(self.model, alpha=getattr(self.config.quant, 'smoothquant_alpha', 0.5))

            if do_awq:
                from modules.awq_layers import finalize_awq
                finalize_awq(self.model)

            if do_mbq:
                from modules.mbq_layers import finalize_mbq
                finalize_mbq(self.model)

            print('==> end calibrate')
            return

        dataloader = DataLoader(dataset, batch_size=self.config.quant.batch_size, shuffle=False, collate_fn=collate_fn_videomme)

        batch_count = 0

        for batch_data in tqdm(dataloader, total=len(dataloader)):
            conversations = batch_data["conversations"]
            
            batch_texts = []
            batch_images = []
            batch_audios = []
            batch_videos = []
            
            for conversation in conversations:
                if getattr(self.config.quant, "calib_without_audio", False):
                    for msg in conversation:
                        if msg.get("role") == "user" and "content" in msg:
                            msg["content"] = [c for c in msg["content"] if c.get("type") != "audio"]

                if getattr(self.config.quant, "calib_without_video", False):
                    for msg in conversation:
                        if msg.get("role") == "user" and "content" in msg:
                            msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]
                
                text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
                if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
                    assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
                    text = text[0]
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                
                batch_texts.append(text)
                if images: batch_images.extend(images)
                if audios: batch_audios.extend(audios)
                if videos: batch_videos.extend(videos)

            # NOTE: 不需要在 calibrate() 中设置 set_search(True)，因为 _calibrate=True 时 QuantAct.forward()
            # 始终走 _calibrate 分支，search 标志在此时无效。search 会在 prepare() 中正确启用。
            
            final_images = batch_images if len(batch_images) > 0 else None
            final_audios = batch_audios if len(batch_audios) > 0 else None
            final_videos = batch_videos if len(batch_videos) > 0 else None
            
            inputs = self.processor(
                text=batch_texts,
                images=final_images,
                audio=final_audios,
                videos=final_videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False
            )
            
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            if do_mbq:
                from modules.mbq_layers import set_mbq_modality
                modality = "multimodal" if (batch_images or batch_audios or batch_videos) else "text"
                set_mbq_modality(self.model, modality)

            if do_mquant and len(conversations) == 1:
                self._set_mquant_ctx(conversations[0], inputs)

            with torch.no_grad():
                self.model.generate(**inputs, thinker_max_new_tokens=1, use_audio_in_video=False)

            batch_count += 1

        print('==> end calibrate')
        if device_utils.is_available():
            device_utils.synchronize()
            print(f'[DEBUG] {device_utils.get_device_type().upper()} memory after calibrate: '
                  f'{device_utils.memory_allocated() / 1024**3:.2f} GB allocated, '
                  f'{device_utils.memory_reserved() / 1024**3:.2f} GB reserved')

        # 校准结束后计算 Dispersion Score，用于离群点检测
        quantact_count = 0
        quantact_with_data = 0
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                quantact_count += 1
                module.compute_dispersion_score()
                if module.dispersion_score is not None:
                    quantact_with_data += 1
        print(f'[DEBUG] compute_dispersion_score done: {quantact_with_data}/{quantact_count} QuantAct modules have calibration data')

        if do_smoothquant:
            from modules.smoothquant_layers import finalize_smoothquant
            finalize_smoothquant(self.model, alpha=getattr(self.config.quant, 'smoothquant_alpha', 0.5))

        if do_awq:
            from modules.awq_layers import finalize_awq
            finalize_awq(self.model)

        if do_mbq:
            from modules.mbq_layers import finalize_mbq
            finalize_mbq(self.model)

        if do_mquant:
            from modules.mquant_layers import finalize_mquant
            finalize_mquant(self.model)
            print("[MQuant] 模态特化 scale 计算完成，权重已量化。")

    def prepare(self):
        import torch
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)
        if self.config.model.quant_method not in ('morpho', 'morpho_withhif8', 'morpho_withhif4') and not do_qlora_act:
            return

        print('\n==> start prepare for inference')
        print('[DEBUG] Setting up QuantAct modules for search...')

        plot_layer_names = []
        plot_means = []
        plot_stds = []

        quantact_setup = 0
        quantact_mask = 0
        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                quantact_setup += 1
                module.set_search(search=True)
                module.set_calibrate(calibrate=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))
                
                if hasattr(module, 'set_search_ratio_lower_bound'):
                    module.set_search_ratio_lower_bound(getattr(self.config.quant, 'search_ratio_lower_bound', 0.6))
                    
                if hasattr(module, 'dispersion_score') and module.dispersion_score is not None:
                    dispersion_score = module.dispersion_score
                    total_channels = dispersion_score.numel()

                    mean_ds_val = dispersion_score.mean().item()
                    std_ds_val = dispersion_score.std().item()
                    
                    layer_str = getattr(module, 'layer_name', name)
                    if not layer_str: layer_str = name
                    
                    plot_layer_names.append(layer_str)
                    plot_means.append(mean_ds_val)
                    plot_stds.append(std_ds_val)

                    if getattr(self.config.quant, 'disable_sparse_compensation', False):
                        outlier_mask = torch.zeros_like(dispersion_score, dtype=torch.bool)
                    elif _quantact_use_full_channel_outlier_mask(self.config, getattr(module, "layer_name", "") or ""):
                        outlier_mask = torch.ones_like(dispersion_score, dtype=torch.bool)
                    else:
                        outlier_std_threshold = getattr(self.config.quant, 'outlier_std_threshold', 2.0)
                        threshold = dispersion_score.mean() + outlier_std_threshold * dispersion_score.std()
                        outlier_mask = dispersion_score >= threshold
                    num_outliers = outlier_mask.sum().item()
                    module.register_buffer("outlier_mask", outlier_mask)
                    quantact_mask += 1

        print(f'[DEBUG] QuantAct setup done: {quantact_setup} modules, {quantact_mask} have outlier_mask')

        dataset = VideoMMEDataset(
            base_dir=self.data_dir,
            max_samples=self.config.quant.search_size,
            skip_samples=self.config.quant.calib_size,
            num_frames=self.num_frames
        )

        if self._is_internvl():
            for idx in range(len(dataset)):
                sample = dataset[idx]
                print("Search Sample:", idx + 1)
                with torch.no_grad():
                    self._inference_internvl(sample, max_new_tokens=1, quant_stats_pass=True)

            self._finalize_quantact_prepare()
            print('==> end prepare for inference')
            return

        dataloader = DataLoader(dataset, batch_size=self.config.quant.batch_size, shuffle=False, collate_fn=collate_fn_videomme)

        print(f'[DEBUG] Search dataset: {len(dataset)} samples, batch_size={self.config.quant.batch_size}, {len(dataloader)} batches')

        batch_idx = 0
        for batch_data in dataloader:
            print("Search Batch:", int(batch_idx + 1))
            batch_idx += 1
            conversations = batch_data["conversations"]
            
            batch_texts = []
            batch_images = []
            batch_audios = []
            batch_videos = []
            
            for conversation in conversations:
                if getattr(self.config.quant, "calib_without_video", False):
                    for msg in conversation:
                        if msg.get("role") == "user" and "content" in msg:
                            msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]

                text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
                if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
                    assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
                    text = text[0]
                audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
                
                batch_texts.append(text)
                if images: batch_images.extend(images)
                if audios: batch_audios.extend(audios)
                if videos: batch_videos.extend(videos)

            final_images = batch_images if len(batch_images) > 0 else None
            final_audios = batch_audios if len(batch_audios) > 0 else None
            final_videos = batch_videos if len(batch_videos) > 0 else None

            if len(batch_texts) > 0:
                inputs = self.processor(
                    text=batch_texts,
                    images=final_images,
                    audio=final_audios,
                    videos=final_videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=False
                )
                inputs = inputs.to(self.model.device).to(self.model.dtype)
                
                with torch.no_grad():
                    import time as _time
                    _t0 = _time.time()
                    self.model.generate(
                        **inputs,
                        thinker_max_new_tokens=1,
                        use_audio_in_video=False,
                        return_audio=False,
                    )
                    print(f'  [DEBUG] Search batch {batch_idx} generate took {_time.time() - _t0:.1f}s')

        print('[DEBUG] Search loop complete, finalizing...')
        self._finalize_quantact_prepare()
        
        print('==> end prepare for inference')

    def evaluate(self, save_path):
        num_samples = self.config.data.num_samples if self.config.data.num_samples != -1 else None
        dataset = VideoMMEDataset(
            base_dir=self.data_dir,
            max_samples=num_samples,
            num_frames=self.num_frames
        )

        total, correct = 0, 0

        if self.config.quant.debug_quant_act:
            for name, module in self.model.named_modules():
                if isinstance(module, QuantAct):
                    module.set_debug(debug=True)
            print("✅ QuantAct debug mode enabled.")

        # Ensure QuantAct modules are in eval mode (calibrate=False, search=False)
        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=False)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        # Streaming write: open CSV once and write rows as produced
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        keys = ["video_id", "question", "answer", "prediction", "raw_response", "correct"]
        f = open(save_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        print(f"Start Evaluation on {len(dataset)} samples...")
        try:
            for i in tqdm(range(len(dataset))):
                data = dataset[i]

                pid = data["pid"]
                question = data["question"]
                answer = data["answer"]
                conversation = data["conversation"]
                output = ""

                start_time = time.time()
                try:
                    if self._is_internvl():
                        try:
                            output = self._inference_internvl(data, max_new_tokens=32)
                        except device_utils.oom_errors() as _oom:
                            if not device_utils.is_oom(_oom):
                                raise
                            device_utils.empty_cache()
                            print(f"  Device OOM on {pid}; retry once with stricter vision (fewer frames, 1 tile).")
                            output = self._inference_internvl(
                                data, max_new_tokens=32, memory_override={"num_frames": 6, "max_num_tiles": 1}
                            )
                    else:
                        output = self._inference(conversation)

                    pred = self._normalize_mcq_answer(
                        self._extract_assistant_response(output),
                        data.get("options"),
                    )
                except Exception as e:
                    import traceback
                    print(f"\n========== 样本 {pid} 报错详情 ==========")
                    traceback.print_exc()
                    print("==========================================\n")
                    pred = "Failed"

                answer = self._normalize_mcq_answer(answer, data.get("options"))
                is_correct = (pred == answer)
                if is_correct:
                    correct += 1
                total += 1

                writer.writerow({
                    "video_id": pid,
                    "question": question,
                    "answer": answer,
                    "prediction": pred,
                    "raw_response": output if 'output' in locals() else "",
                    "correct": is_correct
                })
                # Periodic flush for crash-safety
                if total % 10 == 0:
                    f.flush()
                # 每 100 个样本报告一次running accuracy（tqdm.write 以免打断进度条）
                if total % 100 == 0:
                    tqdm.write(
                        f"  [{total}/{len(dataset)}] running accuracy: "
                        f"{correct / total:.4f} ({correct}/{total})"
                    )
        finally:
            f.close()

        acc = correct / total if total > 0 else 0.0

        print(f"\n{'='*20} Video-MME Evaluation Summary {'='*20}")
        print(f"Quant Method   : {self.config.model.quant_method}")
        print(f"Total Samples  : {total}")
        print(f"Correct        : {correct}")
        print(f"Accuracy       : {acc:.4f} ({acc*100:.2f}%)")
        print(f"{'='*70}")
        self._print_config_yaml()

        print(f"Results saved to {save_path}")
        return acc


class AirBenchEvaluator(BaseEvaluator):
    def __init__(self, model, processor, config, base_dir):
        super().__init__(model, processor, config, base_dir)
        self.data_root = config.data.data_root
        self.output_file = config.data.output_file
        self.run_pca = config.data.run_pca

    def _extract_assistant_response(self, text):
        response = text
        if "assistant" in text.lower():
            parts = text.split("assistant", 1)
            if len(parts) > 1:
                response = parts[1].strip()

        # 更严谨的正则匹配。先找 (A) 或 A. 这种，如果没有，从后往前找单独的字母。
        match = re.search(r'(?:\(([A-D])\)|([A-D])\.)', response)
        if match:
            return match.group(1) or match.group(2)
        
        # 兜底：找独立的 A-D，尽量避免匹配单词首字母
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return match.group(1)
            
        # 终极兜底：如果就是不加标点，取最后一个出现的大写字母 A-D
        matches = re.findall(r'[A-D]', response)
        if matches:
            return matches[-1]
            
        return "Failed"

    def _inference(self, conversation):
        text = self.processor.apply_chat_template(
            _normalize_conversation_for_template(conversation, self.processor),
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
            assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
            text = text[0]
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )

        inputs = inputs.to(self.model.device).to(self.model.dtype)
        self._set_mquant_ctx(conversation, inputs)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, thinker_max_new_tokens=256, use_audio_in_video=False)
            output_ids = output_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def _meta_file_path(self):
        return os.path.join(self.data_root, "Foundation_meta.json")

    def _load_foundation_meta(self, max_samples=None, skip_samples=0):
        meta_path = self._meta_file_path()
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found at {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as fin:
            data = json.load(fin)

        if skip_samples > 0:
            data = data[skip_samples:]

        if max_samples is not None and max_samples != -1:
            return data[:max_samples]
        return data

    def _build_audio_path(self, item):
        wav_file = item["path"]
        task_name = item["task_name"]
        dataset_name = item["dataset_name"]
        if task_name == "Audio_Grounding":
            return f"{self.data_root}/{task_name}_{dataset_name}/{wav_file}"[:-3] + "flac"
        return f"{self.data_root}/{task_name}_{dataset_name}/{wav_file}"

    def _build_conversation(self, item, audio_path):
        question = item["question"]
        choice_a = item["choice_a"]
        choice_b = item["choice_b"]
        choice_c = item.get("choice_c", "N/A")
        choice_d = item.get("choice_d", "N/A")

        choices_str = f"A. {choice_a}\nB. {choice_b}"
        if item.get("choice_c"):
            choices_str += f"\nC. {choice_c}"
        if item.get("choice_d"):
            choices_str += f"\nD. {choice_d}"

        system_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )
        user_instruction = (
            "Please listen to the audio and answer the question.\n"
            f"Question: {question}\n"
            f"Choices:\n{choices_str}\n"
            "Please choose the most suitable answer from options A, B, C, and D. "
            "Only output the option letter (e.g., A)."
        )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": user_instruction}
                ]
            }
        ]

    def _batch_generate_for_quant_search(self, sample_items):
        if len(sample_items) == 0:
            return

        batch_texts = []
        batch_images = []
        batch_audios = []
        batch_videos = []
        
        calib_without_audio = getattr(self.config.quant, "calib_without_audio", False)
        calib_without_video = getattr(self.config.quant, "calib_without_video", False)
        
        # # Open a log file for debugging
        # log_file = None
        # if calib_without_audio or calib_without_video:
        #     log_file = open("calib_debug.log", "a")
        #     log_file.write("[DEBUG] ============ 校准开始 ============\n")
        #     if calib_without_audio:
        #         log_file.write(f"[DEBUG] 校准模式：移除音频输入\n")
        #     if calib_without_video:
        #         log_file.write(f"[DEBUG] 校准模式：移除视频输入\n")
        #     log_file.flush()
        #     print("[DEBUG] 校准模式：移除音频输入" if calib_without_audio else "[DEBUG] 校准模式：移除视频输入")

        for idx, item in enumerate(sample_items):
            audio_path = self._build_audio_path(item)
            if not os.path.exists(audio_path):
                continue
            conversation = self._build_conversation(item, audio_path)
            
            # if calib_without_audio:
            #     # conversation[0] is system, conversation[1] is user message with audio
            #     user_msg_content = conversation[1].get('content', [])
            #     audio_count_before = sum(1 for c in user_msg_content if c.get("type") == "audio")
            #     
            #     log_msg = f"[样本 {idx}] 修改前: 包含 {audio_count_before} 个音频，内容类型 = {[c.get('type') for c in user_msg_content]}\n"
            #     log_file.write(log_msg)
            #     print(log_msg.strip())
            #     
            #     for msg in conversation:
            #         if msg.get("role") == "user" and "content" in msg:
            #             msg["content"] = [c for c in msg["content"] if c.get("type") != "audio"]
            #     
            #     user_msg_content_after = conversation[1].get('content', [])
            #     audio_count_after = sum(1 for c in user_msg_content_after if c.get("type") == "audio")
            #     
            #     log_msg = f"[样本 {idx}] 修改后: 包含 {audio_count_after} 个音频，内容类型 = {[c.get('type') for c in user_msg_content_after]}\n"
            #     log_file.write(log_msg)
            #     print(log_msg.strip())

            # if calib_without_video:
            #     video_name = conversation[1]["content"][0]["video"] if len(conversation[1]["content"]) > 0 and "video" in conversation[1]["content"][0] else None
            #     if video_name:
            #         log_msg = f"[样本 {idx}] 移除视频输入: {video_name}\n"
            #         log_file.write(log_msg)
            #         print(log_msg.strip())
            #         for msg in conversation:
            #             if msg.get("role") == "user" and "content" in msg:
            #                 msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]

            text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
            if isinstance(text, (list, tuple)):  # transformers 4.5x 返回 List[str]
                assert len(text) == 1, f"apply_chat_template returned {len(text)} prompts, expected 1"
                text = text[0]
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            
            # if calib_without_audio:
            #     log_msg = f"[样本 {idx}] process_mm_info 提取的 audios 数量: {len(audios) if audios else 0}\n"
            #     log_file.write(log_msg)
            #     print(log_msg.strip())

            batch_texts.append(text)
            if images:
                batch_images.extend(images)
            if audios:
                batch_audios.extend(audios)
            if videos:
                batch_videos.extend(videos)

        if len(batch_texts) == 0:
            # if log_file:
            #     log_file.close()
            return

        # if calib_without_audio or calib_without_video:
        #     log_msg = f"[DEBUG] 校准数据统计:\n  - batch_texts 数量: {len(batch_texts)}\n  - batch_images 数量: {len(batch_images)}\n  - batch_audios 数量: {len(batch_audios)}\n  - batch_videos 数量: {len(batch_videos)}\n"
        #     log_file.write(log_msg)
        #     print(log_msg.strip())

        inputs = self.processor(
            text=batch_texts,
            images=batch_images if len(batch_images) > 0 else None,
            audio=batch_audios if len(batch_audios) > 0 else None,
            videos=batch_videos if len(batch_videos) > 0 else None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        
        # if calib_without_audio or calib_without_video:
        #     log_msg = f"[DEBUG] inputs 字段: {list(inputs.keys())}\n"
        #     if 'input_features' in inputs:
        #         log_msg += f"  - input_features shape: {inputs['input_features'].shape if hasattr(inputs['input_features'], 'shape') else 'None'}\n"
        #     log_file.write(log_msg)
        #     print(log_msg.strip())
        #     log_file.write("[DEBUG] ============ 校准结束 ============\n\n")
        #     log_file.flush()
        #     log_file.close()
        
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        if getattr(self.config.quant, "simulate_mbq", False):
            from modules.mbq_layers import set_mbq_modality
            modality = "multimodal" if (batch_images or batch_audios or batch_videos) else "text"
            set_mbq_modality(self.model, modality)
        with torch.no_grad():
            self.model.generate(**inputs, thinker_max_new_tokens=1, use_audio_in_video=False)

    def calibrate(self):
        do_smoothquant = getattr(self.config.quant, 'simulate_smoothquant', False)
        do_awq = getattr(self.config.quant, 'simulate_awq', False)
        do_mbq = getattr(self.config.quant, 'simulate_mbq', False)
        do_mquant = getattr(self.config.quant, 'simulate_mquant', False)
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)

        if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8", "morpho_withhif4") and not do_smoothquant and not do_awq and not do_mbq and not do_mquant and not do_qlora_act:
            print(f"{self.config.model.quant_method}模式：跳过校准步骤。")
            return

        print("\n==> start batched calibrate")
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=True)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        if do_smoothquant:
            from modules.smoothquant_layers import set_smoothquant_observe
            set_smoothquant_observe(self.model, enabled=True)

        if do_awq:
            from modules.awq_layers import set_awq_observe
            set_awq_observe(self.model, enabled=True)

        if do_mbq:
            from modules.mbq_layers import set_mbq_observe
            set_mbq_observe(self.model, enabled=True)

        if do_mquant:
            from modules.mquant_layers import set_mquant_observe
            set_mquant_observe(self.model, enabled=True)

        calib_size = self.config.quant.calib_size
        calib_batch_size = self.config.quant.batch_size
        calib_data = self._load_foundation_meta(max_samples=calib_size)

        for start in tqdm(range(0, len(calib_data), calib_batch_size)):
            batch = calib_data[start:start + calib_batch_size]
            # NOTE: 不需要在 calibrate() 中设置 set_search(True)，因为 _calibrate=True 时 QuantAct.forward()
            # 始终走 _calibrate 分支，search 标志在此时无效。search 会在 prepare() 中正确启用。
            self._batch_generate_for_quant_search(batch)

        print("==> end calibrate")

        # 校准结束后计算 Dispersion Score，用于离群点检测
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.compute_dispersion_score()
                # print(f"  [Dispersion] Layer {getattr(module, 'layer_name', 'Unknown')}: max score={module.dispersion_score.max().item():.4f}" if module.dispersion_score is not None else f"  [Dispersion] Layer {getattr(module, 'layer_name', 'Unknown')}: skipped (no data)")

        if do_smoothquant:
            from modules.smoothquant_layers import finalize_smoothquant
            finalize_smoothquant(self.model, alpha=getattr(self.config.quant, 'smoothquant_alpha', 0.5))

        if do_awq:
            from modules.awq_layers import finalize_awq
            finalize_awq(self.model)

        if do_mbq:
            from modules.mbq_layers import finalize_mbq
            finalize_mbq(self.model)

        if do_mquant:
            from modules.mquant_layers import finalize_mquant
            finalize_mquant(self.model)
            print("[MQuant] 模态特化 scale 计算完成，权重已量化。")

    def prepare(self):
        do_qlora_act = (self.config.model.quant_method == 'qlora' and getattr(self.config.quant, 'activation_bitwidth', 16) < 16)
        if self.config.model.quant_method not in ("morpho", "qvlm", "morpho_withhif8", "morpho_withhif4") and not do_qlora_act:
            return

        print("\n==> start prepare for inference")

        plot_layer_names = []
        plot_means = []
        plot_stds = []

        for name, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_search(search=True)
                module.set_calibrate(calibrate=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))
                
                if hasattr(module, 'set_search_ratio_lower_bound'):
                    module.set_search_ratio_lower_bound(getattr(self.config.quant, 'search_ratio_lower_bound', 0.6))
                    
                if hasattr(module, "dispersion_score") and module.dispersion_score is not None:
                    dispersion_score = module.dispersion_score
                    total_channels = dispersion_score.numel()

                    mean_ds_val = dispersion_score.mean().item()
                    std_ds_val = dispersion_score.std().item()
                    
                    layer_str = getattr(module, 'layer_name', name)
                    if not layer_str: layer_str = name
                    
                    plot_layer_names.append(layer_str)
                    plot_means.append(mean_ds_val)
                    plot_stds.append(std_ds_val)

                    if getattr(self.config.quant, 'disable_sparse_compensation', False):
                        outlier_mask = torch.zeros_like(dispersion_score, dtype=torch.bool)
                    elif _quantact_use_full_channel_outlier_mask(self.config, getattr(module, "layer_name", "") or ""):
                        outlier_mask = torch.ones_like(dispersion_score, dtype=torch.bool)
                    else:
                        outlier_std_threshold = getattr(self.config.quant, 'outlier_std_threshold', 2.0)
                        threshold = dispersion_score.mean() + outlier_std_threshold * dispersion_score.std()
                        outlier_mask = dispersion_score >= threshold
                    num_outliers = outlier_mask.sum().item()
                    module.register_buffer("outlier_mask", outlier_mask)

        start_idx = self.config.quant.calib_size
        search_data = self._load_foundation_meta(max_samples=self.config.quant.search_size, skip_samples=start_idx)
        
        batch_size = max(1, self.config.quant.batch_size)
        batch_idx = 0
        for start in range(0, len(search_data), batch_size):
            print("Search Batch:", int(batch_idx + 1))
            batch_idx += 1
            batch = search_data[start:start + batch_size]
            self._batch_generate_for_quant_search(batch)

        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                if hasattr(module, 'finalize_search'):
                    module.finalize_search()
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        print("==> end prepare for inference")

    def _register_pca_hooks(self):
        activations = {
            "thinker_input": [],
            "audio_encoder": [],
            "text_embed": []
        }

        def get_input_activation(name):
            def hook(module, input):
                hidden_states = input[0] if isinstance(input, tuple) else input
                activations[name].append(
                    hidden_states.detach().cpu().to(torch.float32).numpy().reshape(-1, hidden_states.shape[-1])
                )
            return hook

        def get_output_activation(name):
            def hook(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                activations[name].append(
                    hidden_states.detach().cpu().to(torch.float32).numpy().reshape(-1, hidden_states.shape[-1])
                )
            return hook

        hook_handles = []

        thinker_layers = None
        if hasattr(self.model, "thinker") and hasattr(self.model.thinker, "model") and hasattr(self.model.thinker.model, "layers"):
            thinker_layers = self.model.thinker.model.layers
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            thinker_layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            thinker_layers = self.model.layers

        if thinker_layers is not None:
            h1 = thinker_layers[2].register_forward_pre_hook(get_input_activation("thinker_input"))
            hook_handles.append(h1)

        if hasattr(self.model, "visual"):
            h2 = self.model.visual.register_forward_hook(get_output_activation("audio_encoder"))
            hook_handles.append(h2)
        elif hasattr(self.model, "thinker") and hasattr(self.model.thinker, "visual"):
            h2 = self.model.thinker.visual.register_forward_hook(get_output_activation("audio_encoder"))
            hook_handles.append(h2)

        embed_tokens = None
        if hasattr(self.model, "thinker") and hasattr(self.model.thinker, "model") and hasattr(self.model.thinker.model, "embed_tokens"):
            embed_tokens = self.model.thinker.model.embed_tokens
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            embed_tokens = self.model.model.embed_tokens
        elif hasattr(self.model, "embed_tokens"):
            embed_tokens = self.model.embed_tokens

        if embed_tokens is not None:
            h3 = embed_tokens.register_forward_hook(get_output_activation("text_embed"))
            hook_handles.append(h3)

        return hook_handles, activations

    def _process_and_save_pca(self, activations, sample_item, output_dir):
        if len(activations["thinker_input"]) == 0:
            return

        x_mixed = np.array(np.concatenate(activations["thinker_input"], axis=0))
        os.makedirs(output_dir, exist_ok=True)
        uniq_id = sample_item.get("uniq_id", "sample")

        max_fit_samples = 5000
        if x_mixed.shape[0] > max_fit_samples:
            indices = np.random.choice(x_mixed.shape[0], max_fit_samples, replace=False)
            x_fit = x_mixed[indices]
        else:
            x_fit = x_mixed

        pca = PCA(n_components=2)
        pca.fit(x_fit)
        x_mixed_pca = pca.transform(x_mixed)

        data_save_path = os.path.join(output_dir, f"pca_data_{uniq_id}.npz")
        save_dict = {
            "mixed": x_mixed,
            "mixed_pca": x_mixed_pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
        if len(activations["audio_encoder"]) > 0:
            save_dict["audio_encoder"] = np.concatenate(activations["audio_encoder"], axis=0)
        if len(activations["text_embed"]) > 0:
            save_dict["text_embed"] = np.concatenate(activations["text_embed"], axis=0)
        np.savez(data_save_path, **save_dict)

        plt.close("all")
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 24,
            "axes.linewidth": 2,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "axes.labelsize": 24,
            "legend.fontsize": 24,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })

        plt.figure(figsize=(8, 8))
        plt.scatter(
            x_mixed_pca[:, 0],
            x_mixed_pca[:, 1],
            alpha=0.25,
            color="#1f77b4",
            label="Audio Tokens",
            s=40,
            edgecolors="none",
            rasterized=True,
        )

        if len(activations["text_embed"]) > 0:
            x_text = np.array(np.concatenate(activations["text_embed"], axis=0))
            if x_text.shape[1] == x_mixed.shape[1]:
                x_text_pca = pca.transform(x_text)
                plt.scatter(
                    x_text_pca[:, 0],
                    x_text_pca[:, 1],
                    alpha=1.0,
                    color="#d62728",
                    label="Text Tokens",
                    s=200,
                    edgecolors="white",
                    linewidth=0.5,
                )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(frameon=False, fancybox=False, edgecolor="black", framealpha=1.0, loc="best")

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        quant_dir = os.path.join(output_dir, self.config.model.quant_method)
        os.makedirs(quant_dir, exist_ok=True)
        save_path = os.path.join(quant_dir, f"pca_analysis_{uniq_id}_{self.config.model.quant_method}.pdf")
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        plt.close()

    def evaluate(self, save_path=None):
        output_file = save_path if save_path is not None else self.output_file
        data = self._load_foundation_meta(max_samples=self.config.data.num_samples)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        pca_output_dir = "."
        if self.run_pca:
            pca_output_dir = os.path.dirname(os.path.abspath(output_file))
            if not pca_output_dir:
                pca_output_dir = "."
            print(f"PCA results will be saved to: {pca_output_dir}")

        # Ensure QuantAct modules are in eval mode (calibrate=False, search=False)
        for _, module in self.model.named_modules():
            if isinstance(module, QuantAct):
                module.set_calibrate(calibrate=False)
                module.set_search(search=False)
                if hasattr(module, 'set_sparse_buffer_ratio'):
                    module.set_sparse_buffer_ratio(getattr(self.config.quant, 'sparse_buffer_ratio', 0.8))

        total_count = 0
        correct_count = 0

        print(f"Reading data from {self._meta_file_path()}")
        print(f"Processing {len(data)} samples...")

        with open(output_file, "w", encoding="utf-8") as fout:
            for item in tqdm(data):
                audio_path = self._build_audio_path(item)
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file missing {audio_path}")
                    continue

                conversation = self._build_conversation(item, audio_path)

                raw_response = ""
                final_choice = ""
                hook_handles = []
                activations = {}

                try:
                    if self.run_pca:
                        hook_handles, activations = self._register_pca_hooks()

                    raw_response = self._inference(conversation)
                    final_choice = self._extract_assistant_response(raw_response)

                    if self.run_pca:
                        self._process_and_save_pca(activations, item, pca_output_dir)
                        for h in hook_handles:
                            h.remove()

                except Exception as e:
                    print(f"Inference failed for {item['uniq_id']}: {e}")
                    raw_response = "Error"
                    final_choice = "Error"
                    if self.run_pca:
                        for h in hook_handles:
                            h.remove()

                choice_a = item["choice_a"]
                choice_b = item["choice_b"]
                choice_c = item.get("choice_c", "N/A")
                choice_d = item.get("choice_d", "N/A")
                choices = [choice_a, choice_b, choice_c, choice_d]

                answer_gt = str(item.get("answer_gt", "")).strip()
                gt_letter = self._normalize_mcq_answer(answer_gt, choices)
                pred_letter = self._normalize_mcq_answer(final_choice, choices)

                if pred_letter == gt_letter:
                    correct_count += 1
                total_count += 1

                result_item = {
                    "path": item["path"],
                    "question": item["question"],
                    "choice_a": choice_a,
                    "choice_b": choice_b,
                    "choice_c": item.get("choice_c"),
                    "choice_d": item.get("choice_d"),
                    "answer_gt": item["answer_gt"],
                    "task_name": item["task_name"],
                    "dataset_name": item["dataset_name"],
                    "response": pred_letter,
                    "raw_response": raw_response,
                    "uniq_id": item["uniq_id"],
                }
                fout.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                fout.flush()

        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"\n{'='*20} AIR-Bench Evaluation Summary {'='*20}")
            print(f"Quant Method   : {self.config.model.quant_method}")
            print(f"Total Samples  : {total_count}")
            print(f"Correct        : {correct_count}")
            print(f"Accuracy       : {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"{'='*70}\n")
        else:
            accuracy = 0.0

        self._print_config_yaml()
        print(f"Done! Results saved to {output_file}")
        return accuracy
