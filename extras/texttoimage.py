from diffusers import StableDiffusionPipeline
import torch
def text_to_image(prompt, output_path):
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float)
    pipe = pipe.to(device)

    image = pipe(prompt).images[0]
    image.save(output_path)

if __name__ == "__main__":
    prompt = "A white cat with red yarn."
    filename = "output2.png"
    output_path = "genai_outputs/"+filename 
    text_to_image(prompt, output_path)
