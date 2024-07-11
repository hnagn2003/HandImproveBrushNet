from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image
import torch
import os
def get_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read all lines from the file and store them as a list of strings
            prompts = file.readlines()
            # Remove newline characters from each line
            prompts = [prompt.strip() for prompt in prompts]
            # Print the list of lines
            return prompts[:10]
    except FileNotFoundError:
        print("File not found. Please provide the correct file path.")
        
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
OUTDIR = '/lustre/scratch/client/vinai/users/ngannh9/hand_improve/BrushNet/output/hand_prompt/'
BATCH_SIZE = 3
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)
# prompt = ["a 8K portrait of beautiful luxury elegant lady, posing with her hands",
#         "a beautiful woman, waving hands, smiling, looking at viewer",
#         "a close-up picture of human hand",
#         "a potrait of an elderly Serbian man eating banana",
#         "image of hand shaking",
#         "a picture of a person, drawing on the paper",
#         "A picture of Will Smith eating spaghetti",
#         "a portrait of young woman in a professional stance, arms crossed, captured from the waist up"]
prompt = get_from_file(file_path="./hand_prompts.txt")
negative_prompt="ugly, low quality, incorrect and unrealistic hands, bad atonomy"
generator = torch.Generator(device="cuda").manual_seed(1234)
# pipeline_text2image.enable_vae_slicing()
start = 0
end = BATCH_SIZE
for _ in range(len(prompt) // BATCH_SIZE + 1):
    print(prompt[start:end])
    outputs = pipeline_text2image(prompt=prompt[start:end], negative_prompt=negative_prompt, generator=generator).images
    for i, output in enumerate(outputs):
        image = output
        image.save(OUTDIR + str(start+i) + '.png') 
    start += BATCH_SIZE
    end += BATCH_SIZE