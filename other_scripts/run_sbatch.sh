#!/bin/bash

dos2unix $1 && sbatch $1 &&  sleep 2 && ./monitor_job.sh $(squeue --me -h -o %i | head -n 1)
