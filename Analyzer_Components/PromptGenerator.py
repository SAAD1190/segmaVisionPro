from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PromptGenerator:
    
    def __init__(self, image_directory="./data/", images_extensions=['jpg', 'jpeg', 'png'], models_to_use=None):
        # Initialize available models
        self.available_models = {
            "blip": self.generate_blip_prompts,
            "vit_gpt2": self.generate_vit_gpt2_prompts
        }
        
        # Set the models to use (if None, use all available models)
        self.models_to_use = models_to_use if models_to_use else list(self.available_models.keys())
        
        # Initialize the processors and models only for selected ones
        self.models = {
            "blip": BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device),
            "vit_gpt2": VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
        }
        
        self.processors = {
            "blip": BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
            "vit_gpt2": ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        }

        self.tokenizers = {
            "vit_gpt2": AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        }

        self.image_directory = image_directory
        self.images_extensions = images_extensions
        self.prompt_pool = {}

    def load_images(self):
        images = []
        image_names = []
        for image_name in os.listdir(self.image_directory):
            if image_name.split('.')[-1].lower() in self.images_extensions:
                image_path = os.path.join(self.image_directory, image_name)
                try:
                    i_image = Image.open(image_path)
                    if i_image.mode != "RGB":
                        i_image = i_image.convert(mode="RGB")
                    images.append(i_image)
                    image_names.append(image_name)
                except Exception as e:
                    print(f"Error opening image {image_name}: {e}")
                    continue

        return images, image_names

    def generate_blip_prompts(self, images, num_prompts, max_length):
        prompts_dict = {}
        for idx, image in enumerate(images):
            inputs = self.processors["blip"](images=image, return_tensors="pt").to(device)
            pixel_values = inputs['pixel_values']
            attention_mask = torch.ones(pixel_values.shape[:-1], dtype=torch.long).to(device)  # Add attention mask
            prompts = []
            for _ in range(num_prompts):
                output_ids = self.models["blip"].generate(
                    pixel_values,
                    attention_mask=attention_mask,  # Pass attention mask
                    max_length=max_length,
                    num_beams=1,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0
                )
                pred = self.processors["blip"].decode(output_ids[0], skip_special_tokens=True).strip()
                prompts.append(pred)
            prompts_dict[idx] = prompts
        return prompts_dict

    def generate_vit_gpt2_prompts(self, images, num_prompts, max_length):
        prompts_dict = {}
        for idx, image in enumerate(images):
            pixel_values = self.processors["vit_gpt2"](images=image, return_tensors="pt").pixel_values.to(device)
            attention_mask = torch.ones(pixel_values.shape[:-1], dtype=torch.long).to(device)  # Add attention mask
            prompts = []
            for _ in range(num_prompts):
                output_ids = self.models["vit_gpt2"].generate(
                    pixel_values,
                    attention_mask=attention_mask,  # Pass attention mask
                    max_length=max_length,
                    num_beams=1,
                    do_sample=True,
                    top_k=50,
                    temperature=1.0
                )
                pred = self.tokenizers["vit_gpt2"].decode(output_ids[0], skip_special_tokens=True).strip()
                prompts.append(pred)
            prompts_dict[idx] = prompts
        return prompts_dict

    def generate_prompts(self, num_prompts=5, max_length=15):
        images, image_names = self.load_images()
        final_prompt_pool = {}

        # For each selected model, generate prompts and concatenate the results
        for model_name in self.models_to_use:
            if model_name in self.available_models:
                model_prompts = self.available_models[model_name](images, num_prompts, max_length)
                for idx, prompts in model_prompts.items():
                    if image_names[idx] not in final_prompt_pool:
                        final_prompt_pool[image_names[idx]] = []
                    final_prompt_pool[image_names[idx]].extend(prompts)

        return final_prompt_pool

# Example usage:
# To use all models (BLIP and ViT-GPT2):
# prompt_gen = PromptGenerator(image_directory="./data/")
# final_prompts = prompt_gen.generate_prompts(num_prompts=5)
# To use only selected models (e.g., 'blip'):
# prompt_gen = PromptGenerator(image_directory="./data/", models_to_use=['blip'])
