import re

with open('modules/evaluator.py', 'r') as f:
    evaluator_code = f.read()

# Fix ScienceQA prepare

sci_qa_old = """
        dataset = ScienceQADataset(
            base_dir=self.base_dir,
            problems_path=self.problems_path,
            split_path=self.split_path,
            split="val",
            only_samples_with_images=self.config.data.only_samples_with_images,
            max_samples=self.config.data.calib_batch_size,
            skip_samples=self.config.data.calib_size  # 跳过前面用于校准的样本，保证数据隔离
        )
        
        batch_texts = []
        batch_images = []
        batch_audios = []
        batch_videos = []
        
        for i in range(len(dataset)):
            data = dataset[i]
            conversation = data["conversation"]

            if getattr(self.config.quant, "calib_without_video", False):
                for msg in conversation:
                    if msg.get("role") == "user" and "content" in msg:
                        msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]

            conv_for_template = _normalize_conversation_for_template(conversation, self.processor)
            text = self.processor.apply_chat_template(conv_for_template, add_generation_prompt=True, tokenize=False)
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
                self.model.generate(
                    **inputs,
                    thinker_max_new_tokens=1,
                    use_audio_in_video=False,
                    return_audio=False,
                )
"""

sci_qa_new = """
        dataset = ScienceQADataset(
            base_dir=self.base_dir,
            problems_path=self.problems_path,
            split_path=self.split_path,
            split="val",
            only_samples_with_images=self.config.data.only_samples_with_images,
            max_samples=self.config.quant.search_size,
            skip_samples=self.config.data.calib_size  # 跳过前面用于校准的样本，保证数据隔离
        )
        
        dataloader = DataLoader(dataset, batch_size=self.config.data.batch_size, shuffle=False, collate_fn=collate_fn_scienceqa)

        for batch_data in dataloader:
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
                    self.model.generate(
                        **inputs,
                        thinker_max_new_tokens=1,
                        use_audio_in_video=False,
                        return_audio=False,
                    )
"""

if sci_qa_old in evaluator_code:
    evaluator_code = evaluator_code.replace(sci_qa_old, sci_qa_new)
else:
    print("Warning: Could not find sci_qa_old exact match")


mmmu_old = """
        search_samples = self._collect_quant_samples(self.config.data.calib_size, self.config.data.calib_batch_size)
        print(f"Collected {len(search_samples)} search samples.")
        self._batch_generate_for_quant_search(
            search_samples,
            rearm_quantact_search_each_sample=self._is_internvl(),
        )
"""

mmmu_new = """
        search_samples = self._collect_quant_samples(self.config.data.calib_size, self.config.quant.search_size)
        print(f"Collected {len(search_samples)} search samples.")
        batch_size = max(1, self.config.data.batch_size)
        for idx in range(0, len(search_samples), batch_size):
            batch = search_samples[idx : idx + batch_size]
            self._batch_generate_for_quant_search(
                batch,
                rearm_quantact_search_each_sample=self._is_internvl(),
            )
"""

if mmmu_old in evaluator_code:
    evaluator_code = evaluator_code.replace(mmmu_old, mmmu_new)
else:
    print("Warning: Could not find mmmu_old exact match")

videomme_old = """
        dataset = VideoMMEDataset(
            base_dir=self.data_dir,
            max_samples=self.config.data.calib_batch_size,
            skip_samples=self.config.data.calib_size
        )
        
        batch_texts = []
        batch_images = []
        batch_audios = []
        batch_videos = []
        
        for i in range(len(dataset)):
            data = dataset[i]
            conversation = data["conversation"]

            if getattr(self.config.quant, "calib_without_video", False):
                for msg in conversation:
                    if msg.get("role") == "user" and "content" in msg:
                        msg["content"] = [c for c in msg["content"] if c.get("type") != "video"]

            text = self.processor.apply_chat_template(_normalize_conversation_for_template(conversation, self.processor), add_generation_prompt=True, tokenize=False)
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
                self.model.generate(
                    **inputs,
                    thinker_max_new_tokens=1,
                    use_audio_in_video=False,
                    return_audio=False,
                )
"""

videomme_new = """
        dataset = VideoMMEDataset(
            base_dir=self.data_dir,
            max_samples=self.config.quant.search_size,
            skip_samples=self.config.data.calib_size
        )
        
        dataloader = DataLoader(dataset, batch_size=self.config.data.batch_size, shuffle=False, collate_fn=collate_fn_videomme)

        for batch_data in dataloader:
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
                    self.model.generate(
                        **inputs,
                        thinker_max_new_tokens=1,
                        use_audio_in_video=False,
                        return_audio=False,
                    )
"""

if videomme_old in evaluator_code:
    evaluator_code = evaluator_code.replace(videomme_old, videomme_new)
else:
    print("Warning: Could not find videomme_old exact match")


airbench_old = """
        start_idx = self.config.data.calib_size
        search_data = self._load_foundation_meta(max_samples=self.config.data.calib_batch_size, skip_samples=start_idx)
        self._batch_generate_for_quant_search(search_data)
"""

airbench_new = """
        start_idx = self.config.data.calib_size
        search_data = self._load_foundation_meta(max_samples=self.config.quant.search_size, skip_samples=start_idx)
        
        batch_size = max(1, self.config.data.batch_size)
        for start in range(0, len(search_data), batch_size):
            batch = search_data[start:start + batch_size]
            self._batch_generate_for_quant_search(batch)
"""

if airbench_old in evaluator_code:
    evaluator_code = evaluator_code.replace(airbench_old, airbench_new)
else:
    print("Warning: Could not find airbench_old exact match")

# Replace remaining `calib_batch_size` with `batch_size` across the remaining files
evaluator_code = evaluator_code.replace('self.config.data.calib_batch_size', 'self.config.data.batch_size')

with open('modules/evaluator.py', 'w') as f:
    f.write(evaluator_code)

print("Replacement complete.")
