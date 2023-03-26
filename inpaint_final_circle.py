import PIL.Image as Image
from PIL import ImageOps
import os
from itertools import islice, cycle
from diffusers import StableDiffusionInpaintPipeline
import torch
import math
import numpy as np
import cv2
IMAGES_FORMAT = ['.jpg', '.JPG','jpeg']  # 图片格式
 
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to('cuda')
def dummy_checker(images, **kwargs):
    return images, False
pipe.safety_checker = dummy_checker

generator = torch.Generator("cuda").manual_seed(34212322233333)
from PIL import Image, ImageDraw, ImageEnhance
def blend_gt2pt(old_image, new_image, sigma=0.15, steps=100):
    new_size = new_image.size
    old_size = old_image.size
    easy_img = np.array(new_image)
    gt_img_array = np.array(old_image)
    pos_w = (new_size[0] - old_size[0]) // 2
    pos_h = (new_size[1] - old_size[1]) // 2

    kernel_h = cv2.getGaussianKernel(old_size[1], old_size[1] * sigma)
    kernel_w = cv2.getGaussianKernel(old_size[0], old_size[0] * sigma)
    kernel = np.multiply(kernel_h, np.transpose(kernel_w))

    kernel[steps:-steps, steps:-steps] = 1
    kernel[:steps, :steps] = kernel[:steps, :steps] / kernel[steps - 1, steps - 1]
    kernel[:steps, -steps:] = kernel[:steps, -steps:] / kernel[steps - 1, -(steps)]
    kernel[-steps:, :steps] = kernel[-steps:, :steps] / kernel[-steps, steps - 1]
    kernel[-steps:, -steps:] = kernel[-steps:, -steps:] / kernel[-steps, -steps]
    kernel = np.expand_dims(kernel, 2)
    kernel = np.repeat(kernel, 3, 2)

    weight = np.linspace(0, 1, steps)
    top = np.expand_dims(weight, 1)
    top = np.repeat(top, old_size[0] - 2 * steps, 1)
    top = np.expand_dims(top, 2)
    top = np.repeat(top, 3, 2)

    weight = np.linspace(1, 0, steps)
    down = np.expand_dims(weight, 1)
    down = np.repeat(down, old_size[0] - 2 * steps, 1)
    down = np.expand_dims(down, 2)
    down = np.repeat(down, 3, 2)

    weight = np.linspace(0, 1, steps)
    left = np.expand_dims(weight, 0)
    left = np.repeat(left, old_size[1] - 2 * steps, 0)
    left = np.expand_dims(left, 2)
    left = np.repeat(left, 3, 2)

    weight = np.linspace(1, 0, steps)
    right = np.expand_dims(weight, 0)
    right = np.repeat(right, old_size[1] - 2 * steps, 0)
    right = np.expand_dims(right, 2)
    right = np.repeat(right, 3, 2)

    kernel[:steps, steps:-steps] = top
    kernel[-steps:, steps:-steps] = down
    kernel[steps:-steps, :steps] = left
    kernel[steps:-steps, -steps:] = right

    pt_gt_img = easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]]
    gaussian_gt_img = kernel * gt_img_array + (1 - kernel) * pt_gt_img  # gt img with blur img
    gaussian_gt_img = gaussian_gt_img.astype(np.int64)
    easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]] = gaussian_gt_img
    gaussian_img = Image.fromarray(easy_img)
    return gaussian_img

def resize_image(image, max_size=1000000, multiple=8):
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(math.sqrt(max_size * aspect_ratio))
    new_height = int(new_width / aspect_ratio)
    new_width = new_width - (new_width % multiple)
    new_height = new_height - (new_height % multiple)
    return image.resize((new_width, new_height))

def dowhile(original_img, tosize, prompt):
    old_img = original_img
    while(old_img.size != tosize):
        crop_w = 20 if old_img.size[0] != tosize[0] else 0
        crop_h = 20 if old_img.size[1] != tosize[1] else 0
        old_img = ImageOps.crop(old_img, (crop_w, crop_h, crop_w, crop_h))
        temp_canvas_size = (4*old_img.width if 4*old_img.width < tosize[0] else tosize[0], 4*old_img.height if 4*old_img.height < tosize[1] else tosize[1])
        temp_canvas, temp_mask = Image.new("RGB", temp_canvas_size, color="white"), Image.new("L", temp_canvas_size, color="white")            
        x, y = (temp_canvas.width - old_img.width) // 2, (temp_canvas.height - old_img.height) // 2
        temp_canvas.paste(old_img, (x, y)) 
        #temp_canvas.save('./test/test1.jpg')
        temp_mask.paste(0, (x, y, x+old_img.width, y+old_img.height))
        resized_temp_canvas, resized_temp_mask = resize_image(temp_canvas), resize_image(temp_mask)
        #resized_temp_mask.save('./test/test2.jpg')
        image = pipe(prompt=prompt, image=resized_temp_canvas, mask_image=resized_temp_mask, height=resized_temp_canvas.height, width=resized_temp_canvas.width, num_inference_steps=100, generator=generator).images[0].resize((temp_canvas.width, temp_canvas.height),Image.ANTIALIAS)
        draw = ImageDraw.Draw(temp_mask)
        mask_box = (x,y,x+old_img.width,y+old_img.height)
        draw.rectangle(mask_box, fill=0)
        image.putalpha(temp_mask)
        image.paste(old_img, (x,y))
        image = blend_gt2pt(old_img.convert('RGB'), image.convert('RGB'))
        old_img = image
        #old_img.convert('RGB').save('./test/1done.jpg')
    return old_img

def crop(image, border_size = 10):
    left = border_size
    top = border_size
    right = border_size
    bottom = border_size
    return ImageOps.crop(image, (left, top, right, bottom))


img_dir = './draw/'  # Set the path to the directory containing the images
text_dir = './imagine/'
output_dir = './after_outpainting/'  # Set the path to the directory to store the output files
img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']  # List of image file extensions to include
img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if os.path.splitext(filename)[1].lower() in img_extensions]  # Get a list of all image files in the directory with the specified extensions

if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the output directory if it doesn't already exist

for img_path in img_paths:
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # Get the base name of the image file without extension
    image = Image.open(img_path)
    txt_path = os.path.join(text_dir, img_name + '.txt')  # Set the path to the output text file
    with open(txt_path, 'r') as f:
        prompt = ""
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespaces and line breaks
            if line != "":  # Check if line is not empty
                prompt = line
                break  # Stop reading file after the first non-empty line
    prompt += " child painting style"
    print(prompt)
    image = crop(image)
    final = dowhile(image,(5800,1200),prompt = prompt)
    output_path = os.path.join(output_dir, img_name + os.path.splitext(img_path)[1])
    final.convert('RGB').save(output_path)

