from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClipPromptGenerator:
    def __init__(self, image_directory="./data/", images_extensions=['jpg', 'jpeg', 'png'], predefined_prompts=None):
        # Initialize CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.image_directory = image_directory
        self.images_extensions = images_extensions
        self.prompts_dict = {}

        # Predefined prompts can be a list of possible prompt options (single words or phrases)
        self.predefined_prompts = predefined_prompts if predefined_prompts else [
            "a cat", "a dog", "a person", "a car", "a building", "nature", "a beach"
        ]  # You can adjust this list based on your task

    def generate_prompts(self, top_k=5):
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

        # Loop through each image and generate ranked prompts based on similarity
        for idx, image_name in enumerate(image_names):
            image = images[idx]

            # Get similarity scores for each prompt against the image
            prompts = self.rank_prompts(image)
            
            # Keep only the top_k prompts based on similarity scores
            self.prompts_dict[image_name] = prompts[:top_k]
            print(f"Generated {len(prompts[:top_k])} prompts for {image_name}")

        return self.prompts_dict

    def rank_prompts(self, image):
        # Encode the image and the predefined prompts
        inputs = self.processor(text=self.predefined_prompts, images=image, return_tensors="pt", padding=True).to(device)
        outputs = self.model(**inputs)

        # Get the similarity scores between image and text (prompts)
        logits_per_image = outputs.logits_per_image  # Image to text similarity
        probs = logits_per_image.softmax(dim=1)      # Convert to probabilities

        # Create a list of prompts with their respective probabilities
        prompt_probs = list(zip(self.predefined_prompts, probs[0].tolist()))

        # Sort prompts by the probability of relevance (highest to lowest)
        ranked_prompts = sorted(prompt_probs, key=lambda x: x[1], reverse=True)

        # Return only the sorted prompts (ignoring scores for now)
        return [prompt for prompt, score in ranked_prompts]

