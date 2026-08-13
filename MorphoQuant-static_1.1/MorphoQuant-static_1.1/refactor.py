import re

def refactor():
    with open("modules/evaluator.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Need to regex remove duplicated methods from existing subclasses FIRST
    methods_to_remove = [
        r"    def _is_internvl\(self\):\n(?:        .*?\n)+",
        r"    def _build_mode_suffix\(self\):\n(?:        .*?\n)+",
        r"    def _print_config_yaml\(self\):\n(?:        .*?\n)+",
        r"    def _search_batch_size\(self\):\n(?:        .*?\n)+",
        r"    def _reset_quantact_search_best\(self\):\n(?:        .*?\n)+",
        r"    def _set_quantact_search_state\(self, enabled: bool\):\n(?:        .*?\n)+"
    ]
    for pattern in methods_to_remove:
        content = re.sub(pattern, "", content)
        
    def fix_init(match):
        return '    def __init__(self, model, processor, config, base_dir):\n        super().__init__(model, processor, config, base_dir)'

    content = re.sub(r'    def __init__\(self, model, processor, config, base_dir\):(\n\s+self\.(model|processor|config|base_dir|model_type) = .*)+', fix_init, content)

    # 1. Insert BaseEvaluator before ScienceQAEvaluator
    base_evaluator_code = """class BaseEvaluator:
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

    def _search_batch_size(self):
        return max(1, int(getattr(self.config.data, "calib_batch_size", 1)))

    def _reset_quantact_search_best(self):
        \"\"\"重置所有 QuantAct 的最佳搜索缓存\"\"\"
        for _, module in self.model.named_modules():
            if hasattr(module, 'reset_search_best'):
                module.reset_search_best()

    def _set_quantact_search_state(self, enabled: bool):
        \"\"\"Enable or disable the quantization search state for QuantAct modules.\"\"\"
        for _, module in self.model.named_modules():
            if type(module).__name__ == "QuantAct" or hasattr(module, "set_search"):
                if hasattr(module, "set_search"):
                    module.set_search(search=enabled)

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
"""
    content = content.replace("class ScienceQAEvaluator:\n", base_evaluator_code + "\n\nclass ScienceQAEvaluator(BaseEvaluator):\n")

    # update inherited classes
    content = content.replace("class MMMUEvaluator(ScienceQAEvaluator):", "class MMMUEvaluator(BaseEvaluator):")
    content = content.replace("class VideoMMEEvaluator(ScienceQAEvaluator):", "class VideoMMEEvaluator(BaseEvaluator):")
    content = content.replace("class AirBenchEvaluator(ScienceQAEvaluator):", "class AirBenchEvaluator(BaseEvaluator):")

    with open("modules/evaluator.py", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    refactor()
