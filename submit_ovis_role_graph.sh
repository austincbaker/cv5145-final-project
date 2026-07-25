#!/bin/bash
# Submit 4 independent Ovis role_graph eval jobs + merge job

J1=$(sbatch --parsable --export=START_INDEX=0,END_INDEX=667 run_ovis_role_graph.sbatch)
J2=$(sbatch --parsable --export=START_INDEX=667,END_INDEX=1334 run_ovis_role_graph.sbatch)
J3=$(sbatch --parsable --export=START_INDEX=1334,END_INDEX=2001 run_ovis_role_graph.sbatch)
J4=$(sbatch --parsable --export=START_INDEX=2001 run_ovis_role_graph.sbatch)

echo "Submitted 4 eval jobs: $J1 $J2 $J3 $J4"

MERGE=$(sbatch --parsable --dependency=afterok:$J1:$J2:$J3:$J4 merge_ovis_role_graph.sbatch)
echo "Submitted merge job: $MERGE (depends on $J1 $J2 $J3 $J4)"
