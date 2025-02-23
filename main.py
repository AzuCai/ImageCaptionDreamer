import os
import torch
from torchvision import models, transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import requests
from io import BytesIO

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Check PyTorch version and handle dtype compatibility
def to_with_dtype(tensor, device, dtype=None):
    if dtype is not None and hasattr(tensor, 'to'):
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device)  # Fallback for older PyTorch versions


# Load image feature extractor (lightweight ResNet18 for keywords)
def load_image_feature_extractor():
    model = models.resnet18(pretrained=True)
    model = model.eval()
    # Convert model weights to FloatTensor initially, handle FP16 later if needed
    model = to_with_dtype(model, device, torch.float32)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform


# Extract keywords from image (simplified for demo)
def extract_keywords_from_image(image_path_or_url, model, transform):
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    image = transform(image).unsqueeze(0)
    # Convert image to FloatTensor for consistency
    image = to_with_dtype(image, device, torch.float32)
    with torch.no_grad():
        features = model(image).flatten()  # Simplified feature extraction
    # Dummy keywords based on features (simplified for demo, replace with actual logic)
    keywords = ["dog", "grass", "toy"]  # Updated for the dog image
    return ", ".join(keywords)


# Load BLIP for image captioning
def load_blip_captioner():
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).eval()
    # Convert BLIP model to HalfTensor for FP16
    model = to_with_dtype(model, device, torch.float16)
    return processor, model


# Load FLAN-T5 for story/creative generation
def load_generator():
    model_name = "google/flan-t5-base"  # Lightweight, instruction-tuned for text generation
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    # Convert FLAN-T5 model to HalfTensor for FP16
    model = to_with_dtype(model, device, torch.float16)
    return tokenizer, model


# Generate initial caption using BLIP
def generate_caption(image_path_or_url, blip_processor, blip_model):
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')

    inputs = blip_processor(image, return_tensors="pt")
    # Convert inputs to HalfTensor to match BLIP's FP16 weights, but keep indices as LongTensor
    inputs = {k: to_with_dtype(v, device, torch.float16) if k != "input_ids" else v.to(device, dtype=torch.long) for
              k, v in inputs.items()}
    with torch.no_grad():
        outputs = blip_model.generate(**inputs, max_length=50, num_beams=5)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True).strip()
    return caption


# Generate detailed text (description, story, or creativity) using FLAN-T5
def generate_text(image_path_or_url, output_type, blip_processor, blip_model, tokenizer, generator_model):
    if not image_path_or_url:
        return "Please upload an image or provide a URL."

    try:
        # Generate initial caption using BLIP
        caption = generate_caption(image_path_or_url, blip_processor, blip_model)
        # Extract keywords (simplified)
        keywords = extract_keywords_from_image(image_path_or_url, image_feature_model, image_transform)

        if output_type == "description":
            prompt = f"Describe this image in detail: {caption}, including keywords: {keywords}."
        elif output_type == "story":
            prompt = f"Generate a 10-15 sentence story based on this image: {caption}, with keywords: {keywords} and a creative, engaging tone."
        elif output_type == "creativity":
            prompt = f"Provide 5 creative ideas or prompts inspired by this image: {caption}, including keywords: {keywords}."
        else:
            return "Invalid output type."

        inputs = tokenizer(prompt, return_tensors="pt", max_length=150, truncation=True, padding=True)
        # Convert inputs to HalfTensor for FP16, but keep input_ids as LongTensor
        inputs = {k: to_with_dtype(v, device, torch.float16) if k != "input_ids" else v.to(device, dtype=torch.long) for
                  k, v in inputs.items()}

        with torch.no_grad():
            outputs = generator_model.generate(
                **inputs,
                max_length=250,  # Allow longer outputs for detailed stories/ideas
                num_beams=8,  # Increase beams for better quality
                early_stopping=True,
                temperature=0.8,  # More creativity for stories/ideas
                top_k=50,  # Increase variety
                top_p=0.95,  # Broader sampling
                no_repeat_ngram_size=3,  # Prevent repetition
                repetition_penalty=2.0,  # Penalize repeated phrases
                length_penalty=2.0  # Favor longer outputs
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if len(text.split()) < 5 or text.isspace():
            text = "Sorry, I couldn't generate a clear output. Please try another image or adjust your request."
        return text
    except Exception as e:
        return f"Error generating text: {e}"


# Gradio interface
def gradio_interface(image, output_type):
    if not image:
        return "", "Please upload an image or provide a URL."
    try:
        if isinstance(image, str):  # URL
            text = generate_text(image, output_type, blip_processor, blip_model, tokenizer, generator_model)
        else:  # File
            text = generate_text(image.name, output_type, blip_processor, blip_model, tokenizer, generator_model)
        return "", text  # Empty first output for consistency
    except Exception as e:
        return "", f"Error processing image: {e}"


# Main function
def main():
    global image_feature_model, image_transform, blip_processor, blip_model, tokenizer, generator_model

    print("Loading image feature extractor...")
    image_feature_model, image_transform = load_image_feature_extractor()
    print("Loading BLIP captioner...")
    blip_processor, blip_model = load_blip_captioner()
    print("Loading generator model...")
    tokenizer, generator_model = load_generator()

    interface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Image(type="filepath", label="Upload Image or Provide URL"),
            gr.Dropdown(choices=["description", "story", "creativity"], label="Select Output Type")
        ],
        outputs=[
            gr.Textbox(visible=False),  # Hidden error output
            gr.Textbox(label="Generated Text")
        ],
        title="ImageCaptionDreamer",
        description="Generate detailed descriptions, stories, or creative ideas from images using LLMs and lightweight vision models. Upload an image or provide a URL, select an output type, and explore the magic of multi-modal NLP!"
    )
    interface.launch(debug=True)


if __name__ == "__main__":
    main()