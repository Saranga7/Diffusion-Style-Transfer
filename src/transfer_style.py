import argparse
from PIL import Image
from peft import PeftModel
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from diffusers.utils import make_image_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--strength', type=float, required=True)
    parser.add_argument('--infer_steps', type=int, required=True)
    parser.add_argument('--save_path', type=str, default='styled_image.jpg')
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = vars(parser.parse_args())

    # Load pretrained model
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                              safety_checker = None)
    pipeline.to('cuda')


    # Load LoRA
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args['lora_path'])



    # Image Transform
    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )

    # Load content image
    im = Image.open(args['image_path'])
    im_init = image_transforms(im)
   

    generator = torch.Generator("cuda").manual_seed(args['seed'])

    # Inference
    style_images = pipeline(
        prompt=args['prompt'], 
        image=im_init,
        num_inference_steps=args['infer_steps'],
        strength=args['strength'],
        num_images_per_prompt = 3,
        generator = generator,
    ).images

    images = [(im.resize((512, 512)))] + style_images
    out = make_image_grid(images, rows = 1, cols = 4)
    out.save(args['save_path'])

if __name__ == "__main__":
    main()

