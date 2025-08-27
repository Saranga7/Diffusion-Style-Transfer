import os
import glob
import argparse
import random
import shutil
from tqdm import tqdm

class CaptionProcessor:
    """Style reference pictures and save captions with same basename in out_path.

    For each image in data_path/*.jpg this:
      - copies the image to out_path/
      - writes <basename>.txt containing a caption variation

    Args:
        data_path: directory containing .jpg images
        out_path: directory where images + caption files will be written
        style_label: textual label for the style used in caption templates (e.g. "Impressionism")
    """

    def __init__(self, data_path: str, out_path: str, style_label: str):
        self.data_path = data_path
        self.out_path = out_path
        self.style_label = style_label

        self.image_paths = glob.glob(f"{data_path}/*.jpg")

        print(len(self.image_paths), "images found in", data_path)
        os.makedirs(out_path, exist_ok=True)  


    def _default_templates(self):
        """Templates with placeholders: {style}"""
        return [
            "A painting in the style of {style}",
            "An expressive {style} painting",
            "An artwork in the style of {style}",
            "A photo in the style of {style}",
            "An image in the style of {style}",
            "A photo in {style} style",
            "{style} style art",
        ]


    def generate_caption(self):
        caption = ""
        for img_path in tqdm(self.image_paths):
            image_name = os.path.basename(img_path)                   
            base_name = os.path.splitext(image_name)[0] 

            # Copy image to out_path
            out_image_path = os.path.join(self.out_path, image_name)
            shutil.copy2(img_path, out_image_path)        
                       
            caption_filename = base_name + ".txt"
            caption_path = os.path.join(self.out_path, caption_filename)
            template = random.choice(self._default_templates())
            caption = template.format(style=self.style_label)

            with open(caption_path, "w") as f:
                f.write(caption)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--vangogh_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str, required=True)
    # args = vars(parser.parse_args())
    args = {
        'input_path': 'data/rayonism',  
        'output_path': 'data/rayonismDataset',
        'style_label': 'Rayonism'
    }


    cp = CaptionProcessor(
        args['input_path'], 
        args['output_path'], 
        args['style_label'],
    )

    cp.generate_caption()
