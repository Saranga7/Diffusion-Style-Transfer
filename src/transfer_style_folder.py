import os
import torch
import argparse
from PIL import Image
from peft import PeftModel
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import make_image_grid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_path', type = str, required = True)
    parser.add_argument('--prompt', type = str, required = True)
    parser.add_argument('--folder_path', type = str, required = True)
    parser.add_argument('--strength', type = float, required = True)
    parser.add_argument('--infer_steps', type = int, required = True)
    parser.add_argument('--save_path', type = str, default='out')
    parser.add_argument('--seed', type = int, default = 7)
    args = vars(parser.parse_args())

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                              safety_checker = None)
    pipeline.to('cuda')

 

    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, args['lora_path'])


    image_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )

    folder_path = args['folder_path']
    image_names = os.listdir(folder_path)

    gen = torch.Generator(device="cuda").manual_seed(args['seed'])

    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)

        # try:
        im = Image.open(image_path)
        im_init = image_transforms(im)


        style_images = pipeline(
            prompt=args['prompt'], 
            image=im_init,
            num_inference_steps=args['infer_steps'],
            strength=args['strength'],
            num_images_per_prompt = 3,
            generator=gen,
        ).images


        if not os.path.exists(args['save_path']):
            os.makedirs(args['save_path'])

        save_path = os.path.join(args['save_path'], image_name)
        images = [(im.resize((512, 512)))] + style_images

        out = make_image_grid(images, rows = 1, cols = 4)

        out.save(save_path)
        # except:
        #     pass

if __name__ == "__main__":
    main()

