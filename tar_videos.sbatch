#!/bin/bash
#SBATCH --partition=normal
#SBATCH --qos=group3
#SBATCH --output=tar_videos_%j.out
#SBATCH --error=tar_videos_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

echo "Starting tar of videos directory"
echo "Source: /home/au182598/old_bullying_project/videos/"
echo "Node: $(hostname)"
date

tar -cf /home/au182598/aggressive_behavior_project/videos.tar -C /home/au182598/old_bullying_project videos/

echo "Tar complete"
ls -lh /home/au182598/aggressive_behavior_project/videos.tar
date
