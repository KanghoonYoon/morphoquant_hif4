import re

with open("modules/evaluator.py", "r", encoding="utf-8") as f:
    eval_code = f.read()

base_methods = [
    "def _is_internvl(self):",
    "def _build_mode_suffix(self):",
    "def _print_config_yaml(self):"
]

base_class_code = """class BaseEvaluator:
    def __init__(self, model, processor, config, base_dir=None):
        self.model = model
        self.processor = processor
        self.config = config
        self.base_dir = base_dir
        self.model_type = getattr(config.model, "model_type", "qwen2_5_omni").lower()

    def _is_internvl(self):
        return self.model_type.startswith("internvl")

    def _build_mode_suffix(self):
        \"\"\"Build a generic prefix suffix based on config method and size setup for output filename logs.\"\"\"
        return f"{self.config.model.quant_method}_calib{self.config.data.calib_size}_search{getattr(self.config.quant, 'search_size', 0)}"

    def _print_config_yaml(self):
        \"\"\"打印当前使用的 YAML 配置文件完整内容。\"\"\"
        print(f"\\n{'='*50}")
        print(f"  YAML 配置文件: {self.config.config_path}")
        print(f"{'='*50}")
        print(self.config.config_raw)
        print(f"{'='*50}\\n")

    def _extract_assistant_response(self, text):
        raise NotImplementedError
        
    def _inference(self, conversation):
        raise NotImplementedError

    def calibrate(self):
        raise NotImplementedError

    def prepare(self):
        raise NotImplementedError

    def evaluate(self, save_dir):
        raise NotImplementedError
"""

# Replace ScienceQAEvaluator definition
class_decl = "class ScienceQAEvaluator:"
new_class_decl = base_class_code + "\n\nclass ScienceQAEvaluator(BaseEvaluator):"

new_code = eval_code.replace(class_decl, new_class_decl)

# Clean up ScienceQAEvaluator __init__
sci_init_old = """    def __init__(self, model, processor, config, base_dir):
        self.model = model
        self.processor = processor
        self.config = config
        self.base_dir = base_dir
        
        self.problems_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/problems.json"
        self.split_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/pid_splits.json"
        self.model_type = getattr(config.model, "model_type", "qwen2_5_omni").lower()"""

sci_init_new = """    def __init__(self, model, processor, config, base_dir):
        super().__init__(model, processor, config, base_dir)
        self.problems_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/problems.json"
        self.split_path = f"{base_dir}/datasets/ScienceQA/data/scienceqa/pid_splits.json\"\"\"
"""
# Oops wait, the above had some issues with quotes. I will use regex.
