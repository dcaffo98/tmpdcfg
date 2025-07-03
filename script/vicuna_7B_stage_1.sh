#!/bin/bash
#SBATCH --job-name=vicuna_7B_stage_1
#SBATCH --output=/leonardo_scratch/large/userexternal/dcaffagn/logs/%x-%j
#SBATCH --error=/leonardo_scratch/large/userexternal/dcaffagn/logs/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_RESI-WP8
#SBATCH --time=08:00:00

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8
module unload gcc 
module load gcc/11.3.0

source activate mm
cd ~/git/jeppetto

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_ENTITY=dcaffo98
export WANDB_PROJECT=jeppetto
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_CACHE="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models"

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

clip_model_name_or_path="openai/clip-vit-large-patch14"

learning_rate=2e-3
run_name="${SLURM_JOB_NAME}"
output_dir="/leonardo_scratch/large/userexternal/dcaffagn/checkpoints/jeppetto/${run_name}"

per_device_train_batch_size=32
gradient_accumulation_steps=1

language_model_name="lmsys/vicuna-7b-v1.5"
vision_model_name="openai/clip-vit-large-patch14"
train_data_path="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/first_stage_LLaVA/blip_laion_cc_sbu_558k.json"
train_image_folder="/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/first_stage_LLaVA"

((ws = $SLURM_NNODES * $SLURM_GPUS_PER_NODE))
export WORLD_SIZE=$ws

dataloader_num_workers=$(( $SLURM_CPUS_PER_TASK / 2 / $SLURM_GPUS_PER_NODE))

echo "Nodes: ${SLURM_NNODES}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "MASTER ADDR: ${MASTER_ADDR}"
echo "MASTER PORT: ${MASTER_PORT}"
echo "WORLD SIZE: ${WORLD_SIZE}"
echo "DATALOADER WORKERS: ${dataloader_num_workers}"

srun --exclusive -c $SLURM_CPUS_PER_TASK --mem $SLURM_MEM_PER_NODE \
torchrun \
--nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-endpoint=$MASTER_ADDR --master-port=$MASTER_PORT --rdzv-id=$SLURM_JOB_NAME --rdzv-backend=c10d \
src/train/train.py \
--deepspeed deepspeed/zero2.json \
--gradient_checkpointing True \
--seed 42 \
--iwm_loss \
--save_steps 10000 \
--output_dir $output_dir \
--run_name $run_name \
--report_to wandb \
--language_model_name $language_model_name \
--vision_model_name $vision_model_name \
--train_data_path $train_data_path \
--train_image_folder $train_image_folder \
--remove_unused_columns False \
--bf16 True \
--num_train_epochs 1 \
--per_device_train_batch_size $per_device_train_batch_size \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--learning_rate 2e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--dataloader_num_workers $dataloader_num_workers \
