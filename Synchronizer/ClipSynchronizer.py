from transformers import CLIPProcessor, CLIPModel
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ClipSynchronizer:
    def __init__(self, grounding_sam, prompt_generator, threshold=0.3):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.grounding_sam = grounding_sam
        self.prompt_generator = prompt_generator
        self.threshold = threshold  # Threshold to keep prompts with significant class representation

    def synchronize(self):
        image_prompts = self.prompt_generator.generate_prompts()  # Generated prompts
        object_detections = self.grounding_sam.get_detections()   # Detected objects and their classes
        
        synchronized_data = {}
        
        for image_name, prompts in image_prompts.items():
            detections = object_detections.get(image_name, None)
            if detections:
                # Extract the detected classes (single words) from GroundingSAM detections
                detected_classes = [self.grounding_sam.classes[class_id] for class_id in detections.class_id]
                
                # Measure class-prompt similarity using CLIP
                relevant_prompts = self.get_relevant_prompts(image_name, prompts, detected_classes)
                
                synchronized_data[image_name] = {
                    'prompts': relevant_prompts,
                    'objects': detections
                }

        return synchronized_data

    def get_relevant_prompts(self, image_name, prompts, detected_classes):
        # List of relevant prompts to keep
        relevant_prompts = []
        
        # Use CLIP to calculate similarity for each prompt with the detected classes
        for prompt in prompts:
            scores = self.calculate_similarity(prompt, detected_classes)
            # Keep the prompt if any class is well-represented (exceeds the threshold)
            if any(score > self.threshold for score in scores):
                relevant_prompts.append(prompt)

        return relevant_prompts

    def calculate_similarity(self, prompt, detected_classes):
        # Prepare inputs for CLIP: the prompt and detected classes as "text"
        inputs = self.clip_processor(text=detected_classes, images=prompt, return_tensors="pt", padding=True).to(DEVICE)
        
        # Forward pass through CLIP
        outputs = self.clip_model(**inputs)
        
        # Calculate similarity between image (prompt) and text (classes)
        logits_per_image = outputs.logits_per_image  # Image-to-text similarity scores
        probs = logits_per_image.softmax(dim=1)      # Convert to probabilities
        
        return probs[0].tolist()  # Return the similarity scores as a list