#! /bin/bash

#Batch Job Paremeters
#SBATCH --account=MST107266
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gp4d

python generate_raw_data_only_raw.py --input_dir /home/cyan8877/coco_archived/train2017    --result_dir /work/cyan8877/COCO2017_CycleISP_RAW/train2017
# python generate_raw_data_only_raw.py --input_dir /home/cyan8877/coco_archived/train2017       --result_dir /work/cyan8877/COCO2017_CycleISP_RAW/val2017
# python generate_raw_data_only_raw.py --input_dir /home/cyan8877/coco_archived/train2017      --result_dir /work/cyan8877/COCO2017_CycleISP_RAW/test2017

