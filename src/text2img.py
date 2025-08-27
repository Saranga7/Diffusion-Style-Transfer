import argparse
from peft import PeftModel
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--infer_steps', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='output.jpg')
    parser.add_argument('--strength', type=float, default = 0.7)
    args = vars(parser.parse_args())

    # Load pretrained model
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to('cuda')

    # Load LoRA
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args['lora_path'])

    # Inference
    style_images = pipeline(prompt=args['prompt'], 
                           num_inference_steps=args['infer_steps'],
                           strength=args['strength'],
                           num_images_per_prompt = 3).images
    
    out = make_image_grid(style_images, rows = 1, cols = 3)
    out.save(args['save_path'])

if __name__ == "__main__":
    main()

