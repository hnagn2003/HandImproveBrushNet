from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image

# choose the base model here
base_model_path = "hngan/brushnet_juggernautXL_juggernautX"
# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

# input brushnet ckpt path
brushnet_path = "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/BrushNet/ckpt/random_mask_brushnet_ckpt_sdxl_v0"

# choose whether using blended operation
blended = False

# input source image / mask image path and the text prompt
image_path="/lustre/scratch/client/vinai/users/ngannh9/hand_improve/BrushNet/output/jug/images/1.png"
mask_path="/lustre/scratch/client/vinai/users/ngannh9/hand_improve/BrushNet/output/jug/masks/1.png"
caption="a 8K portrait of beautiful luxury elegant lady, posing with her hands"
negative_prompt="ugly, low quality, incorrect and unrealistic hands, bad atonomy"

# conditioning scale
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path, brushnet=brushnet, torch_dtype=torch.float16, low_cpu_mem_usage=False
)
# change to sdxl-vae-fp16-fix to avoid nan in VAE encoding when using fp16
pipe.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
 
# speed up diffusion process with faster scheduler and memory optimization
# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

init_image = cv2.imread(image_path)[:,:,::-1]
mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)

# resize image
h,w,_ = init_image.shape
if w<h:
    scale=1024/w
else:
    scale=1024/h
new_h=int(h*scale)
new_w=int(w*scale)

init_image=cv2.resize(init_image,(new_w,new_h))
mask_image=cv2.resize(mask_image,(new_w,new_h))[:,:,np.newaxis]

init_image = init_image * (1-mask_image)

init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

generator = torch.Generator("cuda").manual_seed(6849302)
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
image = pipe(
    prompt=caption, 
    image=init_image, 
    mask=mask_image, 
    num_inference_steps=50, 
    generator=generator,
    brushnet_conditioning_scale=brushnet_conditioning_scale,
    # negative_prompt=negative_prompt
).images[0]

if blended:
    image_np=np.array(image)
    init_image_np=cv2.imread(image_path)[:,:,::-1]
    mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]

    # blur, you can adjust the parameters for better performance
    mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
    mask_blurred = mask_blurred[:,:,np.newaxis]
    mask_np = 1-(1-mask_np) * (1-mask_blurred)

    image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
    image_pasted=image_pasted.astype(image_np.dtype)
    image=Image.fromarray(image_pasted)

image.save("tuned/output_1.png")