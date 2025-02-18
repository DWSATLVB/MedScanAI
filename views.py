import torch
from PIL import Image
from django.shortcuts import render, redirect
from django.core.files.storage import default_storage
from django.conf import settings
from .models import MedicalImage
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import CLIPProcessor, CLIPModel
from .utils import analyze_image_claude
import numpy as np
import os

# Ensure the correct device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP model and processor (for image feature extraction)
clip_model_name = "openai/clip-vit-base-patch16"  # Pretrained CLIP model
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

# Load GPT-Neo model and tokenizer
gpt_neo_model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(gpt_neo_model_name)
model = GPTNeoForCausalLM.from_pretrained(gpt_neo_model_name).to(device)


def get_image_features(image_path):
    """Process the image and extract features using CLIP"""
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


def generate_diagnosis_with_gpt_neo(image_features):
    """Generate diagnosis using GPT-Neo based on extracted image features"""
    try:
        # Prepare a prompt based on image features
        prompt = f"Given the following features of a medical image, provide a diagnosis: {image_features.mean().item()}"

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate text from GPT-Neo
        output = model.generate(**inputs, max_length=100, num_return_sequences=1)

        # Decode and return the diagnosis text
        diagnosis = tokenizer.decode(output[0], skip_special_tokens=True)
        return diagnosis
    except Exception as e:
        print(f"Error generating diagnosis with GPT-Neo: {e}")
        return "Error generating diagnosis."


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get image from form submission
            image = request.FILES['image']
            image_path = default_storage.save(image.name, image)
            full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

            # Optionally, analyze the image with Claude
            diagnosis_claude = analyze_image_claude(full_image_path)  # Optional: use if needed

            # Extract image features with CLIP and generate diagnosis with GPT-Neo
            image_features = get_image_features(full_image_path)
            if image_features is None:
                return render(request, 'upload.html', {'error': 'Failed to process the image.'})

            diagnosis_gpt_neo = generate_diagnosis_with_gpt_neo(image_features)

            # Combine the results from Claude and GPT-Neo (or you can choose one)
            final_diagnosis = f"{diagnosis_claude} | GPT-Neo Diagnosis: {diagnosis_gpt_neo}"

            # Save result to database
            medical_image = MedicalImage.objects.create(image=image, diagnosis=final_diagnosis)
            return redirect('result', pk=medical_image.pk)

        except Exception as e:
            print(f"Error during image upload or analysis: {e}")
            return render(request, 'upload.html', {'error': 'An error occurred during image upload or analysis.'})

    return render(request, 'upload.html')


def result(request, pk):
    try:
        medical_image = MedicalImage.objects.get(pk=pk)
        return render(request, 'results.html', {'image': medical_image})
    except MedicalImage.DoesNotExist:
        return render(request, 'results.html', {'error': 'Image not found.'})

