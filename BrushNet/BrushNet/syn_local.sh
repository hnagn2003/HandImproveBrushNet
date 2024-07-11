sshpass -p 'Ngan_07102001089082' rsync -aPvrz \
--exclude .git \
--exclude .github \
--exclude env \
--exclude ckpt \
--exclude runs \
--exclude segment-anything \
. spp:/lustre/scratch/client/vinai/users/ngannh9/hand/BrushNet