from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
import os
# choose the base model here
# base_model_path = "hngan/brushnet_juggernautXL_juggernautX"
base_model_path = "/lustre/scratch/client/vinai/users/ngannh9/data/brushnet_ckpt/realisticVisionV60B1_v51VAE"
IMG_SIZE = 1024
# input brushnet ckpt path
brushnet_path = "/lustre/scratch/client/vinai/users/ngannh9/data/brushnet_ckpt/random_mask_brushnet_ckpt"
captions = ["a 8K portrait of beautiful luxury elegant lady, posing with her hands",
            "a beautiful woman, waving hands, smiling, looking at viewer",
            "a close-up picture of human hand",
            "a potrait of an elderly Serbian man eating banana",
            "image of hand shaking",
            "a picture of a person, drawing on the paper",
            "A picture of Will Smith eating spaghetti",
            "a portrait of young woman in a professional stance, arms crossed, captured from the waist up"]
# choose whether using blended operation
OUTPUT_DIR = '/lustre/scratch/client/vinai/users/ngannh9/hand_improve/BrushNet/output/brushnet_realisticVision/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
blended = False 

# input source image / mask image path and the text prompt
folder="/lustre/scratch/client/vinai/users/ngannh9/hand_improve/BrushNet/output/jug"

negative_prompt=["ugly, low quality, incorrect and unrealistic hands, bad atonomy"]*len(captions)

# conditioning scale
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)

pipe = StableDiffusionBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, loơw_cpu_mem_usage=False, safety_checker=None
)
# change to sdxl-vae-fp16-fix to avoid nan in VAE encoding when using fp16
 
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

init_images = []
mask_images = []
image_path = os.path.join(folder, "images")
mask_path = os.path.join(folder, "masks")
for i in range(len(captions)):
    image_file = os.path.join(image_path, str(i) + ".png")
    mask_file = os.path.join(mask_path, str(i) + ".png")
    init_image = cv2.imread(image_file)[:,:,::-1]
    mask_image = 1.*(cv2.imread(mask_file).sum(-1)>255)[:,:,np.newaxis]
    # init_image=cv2.resize(init_images,(IMG_SIZE,IMG_SIZE))
    init_image = init_image * (1-mask_image)
    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")
    init_images.append(init_image)
    mask_images.append(mask_image)
# resize image
# h,w,_ = init_images[0].shape
# if w<h:
#     scale=1024/w
# else:
#     scale=1024/h
# new_h=int(h*scale)
# new_w=int(w*scale)

generator = torch.Generator("cuda").manual_seed(1234)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
# pipe.enable_model_cpu_offload()

images = pipe(
    prompt=captions, 
    image=init_images,
    mask=mask_images, 
    num_inference_steps=50,
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale,
    negative_prompt=negative_prompt, 
).images

# if blended:
#     image_np=np.array(images)
#     init_image_np=cv2.imread(image_path)[:,:,::-1]
#     mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]

#     # blur, you can adjust the parameters for better performance
#     mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
#     mask_blurred = mask_blurred[:,:,np.newaxis]
#     mask_np = 1-(1-mask_np) * (1-mask_blurred)
    
#     image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
#     image_pasted=image_pasted.astype(image_np.dtype)
#     image=Image.fromarray(image_pasted)

for i, output in enumerate(images):
    output.save(OUTPUT_DIR + str(i) + '.png')
