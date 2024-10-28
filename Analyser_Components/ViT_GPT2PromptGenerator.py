from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ViT_GPT2PromptGenerator():
        
        def __init__(self, image_directory="./data/", images_extensions=['jpg', 'jpeg', 'png']):
            self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
            self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

            self.image_directory = image_directory
            self.images_extensions = images_extensions
            self.prompts_dict = {}

        def generate_prompts(self, max_length=15, num_prompts=20):
            images = []
            image_names = []

            # Iterate over each file in the image directory
            for image_name in os.listdir(self.image_directory):
                # Check if the file has a valid image extension
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

            if not images:
                print("No valid images found!")
                return

            # Extract pixel values for all images
            pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values

            # Loop through each image and generate prompts
            for idx, image_name in enumerate(image_names):
                image_pixel_values = pixel_values[idx].unsqueeze(0).to(device)
                prompts = []
                for _ in range(num_prompts):
                    output_ids = self.model.generate(
                        image_pixel_values,
                        max_length=max_length,
                        num_beams=1,  # Disable beam search
                        do_sample=True,  # Enable sampling
                        top_k=50,  # Top-k sampling
                        temperature=1.0  # Sampling temperature
                    )
                    pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                    prompts.append(pred)

                # Store the generated prompts for the image
                self.prompts_dict[image_name] = prompts
                print(f"Generated {len(prompts)} prompts for {image_name}")

            return self.prompts_dict