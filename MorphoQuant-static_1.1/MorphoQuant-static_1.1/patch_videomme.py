with open('modules/evaluator.py', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "dataset = VideoMMEDataset(" in line and "max_samples=self.config.data.batch_size," in lines[i+2]:
        start_idx = i
    if "inputs = inputs.to(self.model.device).to(self.model.dtype)" in line and start_idx != -1 and i > start_idx:
        end_idx = i + 10 # approximate to after model.generate call
        break

if start_idx != -1 and end_idx != -1:
    old_snippet = "".join(lines[start_idx:end_idx])
    
    new_snippet = """        dataset = VideoMMEDataset(
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
    with open('modules/evaluator.py', 'w') as f:
        f.write("".join(lines[:start_idx]))
        f.write(new_snippet)
        f.write("".join(lines[end_idx:]))
    print("VideoMME replaced successfully!")
else:
    print("Indices not found")
