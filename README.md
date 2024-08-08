
# Text-to-Image Generation with DALL-E

Welcome to the Text-to-Image Generation with DALL-E project! This project focuses on generating images from textual descriptions using the DALL-E model.

## Introduction

Text-to-image generation involves creating images based on textual input. In this project, we leverage the power of DALL-E to generate images using a dataset of text-image pairs.

## Dataset

For this project, we will use a custom dataset of text descriptions and their corresponding images. You can create your own dataset and place it in the `data/text_image_pairs.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- OpenAI API
- Torch
- Pandas

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/asimsultan/text_to_image_dalle.git
cd text_to_image_dalle

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes text descriptions and their corresponding images. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: text and image.

# To fine-tune the DALL-E model for text-to-image generation, run the following command:
python scripts/train.py --data_path data/text_image_pairs.csv --api_key YOUR_OPENAI_API_KEY

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/api_key.txt --data_path data/text_image_pairs.csv --api_key YOUR_OPENAI_API_KEY
