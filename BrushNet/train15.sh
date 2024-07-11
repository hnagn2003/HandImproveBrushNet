accelerate launch examples/brushnet/train_brushnet_coco.py \
--pretrained_model_name_or_path /lustre/scratch/client/vinai/users/ngannh9/data/brushnet_ckpt/realisticVisionV60B1_v51VAE \
--output_dir runs/logs/brushnet_segmentationmask \
--train_data_dir /lustre/scratch/client/vinai/users/ngannh9/.cache/huggingface/hub/datasets--random123123--BrushData/snapshots/333cdc6dc928b6efdbe25c0b3943d1c67a85fc8f \
--resolution 512 \
--learning_rate 1e-5 \
--train_batch_size 2 \
--gradient_accumulation_steps 4 \
--tracker_project_name brushnet \
--report_to tensorboard \
--resume_from_checkpoint latest \
--validation_steps 300 \
--checkpointing_steps 10000 