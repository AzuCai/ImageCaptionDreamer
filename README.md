# ImageCaptionDreamer
A fun and innovative LLM-based tool that generates detailed descriptions, stories, or creative ideas from images, optimized for 6G VRAM. Built with Python, it uses lightweight models like BLIP and FLAN-T5 to create engaging text based on user-uploaded images or URLs. Ideal for artists, writers, gamers, and NLP enthusiasts exploring multi-modal LLMs.

## Features
- **Image Processing**: Accepts image uploads or URLs, extracts lightweight features using ResNet18, and generates keywords.
- **Text Generation**: Uses LLMs (e.g., `Salesforce/blip-image-captioning-base`, `google/flan-t5-base`) to produce 10-15 sentence descriptions, stories, or creative prompts.
- **Interactive Interface**: Offers a Gradio-based UI with image upload/URL input and output type selection.
- **Lightweight Design**: Optimized for 6G VRAM GPUs with FP16 precision, ensuring accessibility on limited hardware.
- **Creative Output**: Generates coherent, emotionally rich text without retrieval, focusing on LLMs’ generative power.

## Knowledge Points
This project showcases expertise in:
- **Natural Language Processing (NLP)**: Text generation, prompt engineering, and multi-modal learning.
- **Large Language Models (LLMs)**: Utilizes lightweight models like FLAN-T5 for creative text generation.
- **Image Processing**: Leverages lightweight vision models (e.g., ResNet18) for feature extraction.
- **Model Optimization**: Demonstrates optimization for low-resource environments (6G VRAM).
- **Web Deployment**: Integrates Gradio for an interactive user interface.

## Prerequisites
- **OS**: Windows (tested), Linux, or macOS.
- **Hardware**: GPU with 6G VRAM (optional, CPU fallback available).
- **Anaconda**: Recommended for environment management.
- **Python**: Version 3.9.

## Installation and Deployment
Follow these steps to set up and run the project locally.

### 1. Clone the Repository
git clone https://github.com/yourusername/ImageCaptionDreamer.git
cd ImageCaptionDreamer

### 2. Prepare Images
Create a folder named images in the project root (optional).
Add example image files (e.g., .jpg, .png) or prepare image URLs for testing.

### 3. Set Up the Environment
Open Anaconda Prompt (Windows) or terminal (Linux/macOS) and run:
#### Create a new environment
conda create -n image_env python=3.9
conda activate image_env

#### Install PyTorch with CUDA (for GPU) or CPU-only
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
#### For CPU-only: conda install pytorch torchvision torchaudio cpuonly -c pytorch

#### Install remaining dependencies
pip install transformers gradio torch torchvision PyMuPDF sentencepiece

### 4. Run the Image Caption Generator
Run the application: 
python image_caption_dreamer.py

A Gradio interface will launch in your browser. Upload an image or provide a URL, select an output type (description, story, or creativity), and generate text (e.g., upload a forest image, select "story" for a narrative).


## Future Improvements
Fine-tune the generation model with multi-modal datasets for richer image-text outputs.
Add support for multi-language text generation.
Incorporate advanced vision models (e.g., CLIP) for better keyword extraction (if resources allow).
Enhance image processing for complex or low-quality images.

## License
MIT License - feel free to use, modify, and share!

## Contact
For questions or suggestions, open an issue or reach out to chaoquancai2019@gmail.com.
