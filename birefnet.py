import torch, os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch.nn.functional as F
from PIL import Image, ImageOps, ImageSequence
from models.baseline import BiRefNet
from config import config
from torchvision.transforms.functional import normalize
import numpy as np
import folder_paths
import node_helpers
import cv2
# config = Config()

device = "cuda" if torch.cuda.is_available() else "cpu"
folder_paths.folder_names_and_paths["BiRefNet"] = ([os.path.join(folder_paths.models_dir, "BiRefNet")], folder_paths.supported_pt_extensions)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


class BiRefNet_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "birefnet_model": (folder_paths.get_filename_list("BiRefNet"), ),
            }
        }

    RETURN_TYPES = ("BRNMODEL",)
    RETURN_NAMES = ("birefnetmodel",)
    FUNCTION = "load_model"
    CATEGORY = "🧹BiRefNet"
  
    def load_model(self, birefnet_model):
        if birefnet_model in ['BiRefNet-massive-epoch_240.pth', 
                              'BiRefNet-portrait-TR_P3M_10k-epoch_120.pth',
                              'BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.pth']:
            config.official = True
        else:
            config.official = False
        net = BiRefNet(bb_pretrained=False)
        model_path = folder_paths.get_full_path("BiRefNet", birefnet_model)
        #print(model_path)
        state_dict = torch.load(model_path, map_location='cpu')
        unwanted_prefix = '_orig_mod.'
        # unwanted_prefix = ''
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        net.load_state_dict(state_dict)
        net.to(device)
        net.eval() 
        return [net]


class BiRefNet_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "birefnetmodel": ("BRNMODEL",),
                "image": ("IMAGE",),
                "background":(['RGBA', 'BLACK', 'WHITE', 'RED'],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "remove_background"
    CATEGORY = "🧹BiRefNet"
  
    def remove_background(self, birefnetmodel, image, background):
        processed_images = []
        processed_masks = []
        assert background in ['RGBA', 'BLACK', 'WHITE', 'RED']
        for image in image:
            orig_image = tensor2pil(image)
            w,h = orig_image.size
            image = resize_image(orig_image)
            im_np = np.array(image)
            im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
            im_tensor = torch.unsqueeze(im_tensor,0)
            im_tensor = torch.divide(im_tensor,255.0)
            im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
            if torch.cuda.is_available():
                im_tensor=im_tensor.cuda()

            result = birefnetmodel(im_tensor)[-1].sigmoid()
            #print(result.shape)
            
            result = torch.squeeze(F.interpolate(result, size=(h,w), mode='bilinear') ,0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)    
            im_array = (result*255).cpu().data.numpy().astype(np.uint8)
            pil_im = Image.fromarray(np.squeeze(im_array))
            if background == "RGBA":
                new_im = Image.new("RGBA", pil_im.size, (0,0,0,0))
            elif background == "BLACK":
                new_im = Image.new("RGBA", pil_im.size, (0,0,0,255))
            elif background == "WHITE":
                new_im = Image.new("RGBA", pil_im.size, (255,255,255,255))
            elif background == "RED":
                new_im = Image.new("RGBA", pil_im.size, (255,0,0,255))
            new_im.paste(orig_image, mask=pil_im)

            new_im_tensor = pil2tensor(new_im)  # 将PIL图像转换为Tensor
            pil_im_tensor = pil2tensor(pil_im)  # 同上

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)

        return new_ims, new_masks
'''
class Load_Image_from_path:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image_path": ("STRING", {"default": ""})}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_path"

    def load_image_from_path(self, image_path):
        if not os.path.exists(image_path):
            raise ValueError(f"Directory {image_path} does not exist.") 
        image_files = self.get_image_files(image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for image_file in image_files:
            img = node_helpers.pillow(Image.open, image_file)

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]

                if image.size[0] != w or image.size[1] != h:
                    continue

                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    def get_image_files(self, directory):
        """
        Get all valid image files in the directory.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions) and self.is_image_file(os.path.join(root, file)):
                    image_files.append(os.path.join(root, file))
        return sorted(image_files)

    def is_image_file(self, file_path):
        """
        Check if the file is a valid image.
        """
        try:
            img = Image.open(file_path)
            img.verify()  # Verify that it is, in fact, an image
            return True
        except (IOError, SyntaxError) as e:
            return False
'''

NODE_CLASS_MAPPINGS = {
    "BiRefNet_ModelLoader_Zho": BiRefNet_ModelLoader_Zho,
    "BiRefNet_Zho": BiRefNet_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_ModelLoader_Zho": "🧹BiRefNet Model Loader",
    "BiRefNet_Zho": "🧹BiRefNet",
}
