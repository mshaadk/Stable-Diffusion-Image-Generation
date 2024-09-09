# Stable Diffusion Image Generation
## Overview
This project demonstrates how to generate images using a pre-trained Stable Diffusion model in a Jupyter Notebook environment. The notebook covers the installation of necessary libraries, configuration of model parameters, and the process of generating images from text prompts.

## Project Structure
- **Installation and Updation of Libraries:** Ensures that all required packages are installed and updated.
- **Importing Libraries:** Imports necessary Python modules for data processing, machine learning, and visualization.
- **Configuration Settings:** Defines a configuration class to centralize important parameters.
- **Image Generation:** Sets up and uses the Stable Diffusion model to generate images based on text prompts.

## Installation
To get started, you need to install the required libraries. Run the following command in a Jupyter Notebook cell:

```python
!pip install --upgrade diffusers transformers tensorflow -q
```

## Usage
1. **Configuration Settings:** Modify the CFG class to set parameters for image generation.
```python
class CFG:
    device = "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12
```

2. **Load the Model:** Create an instance of the Stable Diffusion model.
```python
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token="YOUR_HUGGINGFACE_AUTH_TOKEN", guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)
```

3. **Generate an Image:** Use the function generate_image to create images from prompts.
```python
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

image = generate_image("Burj Khalifa in Kerala", image_gen_model)
image.show()  # Display the generated image
```

## Notes
- **Hugging Face Token:** Replace `"YOUR_HUGGINGFACE_AUTH_TOKEN"` with your actual Hugging Face authentication token to access the pre-trained model.
- **Device Configuration:** Ensure that the `device` parameter is set appropriately. Use `"cuda"` for GPU acceleration if available.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

## Contributing
Feel free to fork the repository and submit pull requests. If you have any issues or suggestions, please open an issue on GitHub.
