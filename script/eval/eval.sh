#!/bin/bash
#SBATCH --job-name=float32-llava-more-3.8B-reproduction-eval
#SBATCH --output=/leonardo_scratch/large/userexternal/dcaffagn/logs/%x-%j
#SBATCH --error=/leonardo_scratch/large/userexternal/dcaffagn/logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrB_FM4TGS
#SBATCH --array=1
#SBATCH --time=01:30:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate mm
cd ~/git/jeppetto
export PYTHONPATH=./lmms-eval:.

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models"
export HF_DATASETS_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/mllm_evaluation/cvprw"
export HF_OFFLINE=1

output_dir="/leonardo_scratch/large/userexternal/dcaffagn/checkpoints/jeppetto/${run_name}"

task_list=(pope mme gqa scienceqa_img mmmu_val seedbench ai2d textvqa_val)
echo ${task_list[$SLURM_ARRAY_TASK_ID]}

checkpoint_path="/leonardo_scratch/large/userexternal/dcaffagn/checkpoints/jeppetto/llava_more_3.8B__clip_336__stage_2/checkpoint-5197"

srun -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
python -u lmms-eval/lmms_eval/__main__.py \
--verbosity=DEBUG \
--task ${task_list[$SLURM_ARRAY_TASK_ID]} \
--model mllm \
--model_args "name_or_path=${checkpoint_path},dtype=float32,conv_mode=phi4" \
--device cuda:0 \
--batch_size 1 \
--output /leonardo_scratch/large/userexternal/dcaffagn/logs/lmms_eval \
--log_samples_suffix j \
--log_samples \
--timezone Europe/Paris \
