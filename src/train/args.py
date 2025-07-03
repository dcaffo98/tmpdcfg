from dataclasses import dataclass, field
from typing import Optional
from src.models.conversations import CONV_TEMPLATES
from transformers import TrainingArguments
from src.models import ProjectorType


@dataclass
class DataArguments:
    train_data_path: str
    train_image_folder: str
    eval_data_path: Optional[str] = None
    eval_image_folder: Optional[str] = None
    conv_template: str = CONV_TEMPLATES.PLAIN
    iwm_captions: bool = False
    prompt_max_length: Optional[int] = None
    
    # IJEPA
    ijepa_aspect_ratio: float = field(default_factory=lambda: [0.75, 1.5])
    ijepa_enc_mask_scale: float = field(default_factory=lambda: [0.85, 1.0])
    ijepa_min_keep: int = 10
    ijepa_num_enc_masks: int = 1
    ijepa_num_pred_masks: int = 4
    ijepa_pred_mask_scale: float = field(default_factory=lambda: [0.15, 0.2])    


@dataclass
class ModelArguments:
    iwm_loss: bool = False
    instruction_tuning: bool = False
    train_proj_only: bool = False

    # from checkpoint
    model_name: Optional[str] = None

    # from scratch
    language_model_name: Optional[str] = None
    vision_model_name: Optional[str] = None
    vision_layer_idx: int = -2
    iwm_tgt_vision_layer_idx: int = -1
    iwm_tgt_vision_model_proj_head: bool = False
    iwm_tgt_proj_output_size: int = 0
    iwm_full_img_on_encoder: bool = False
    projector_type: Optional[ProjectorType] = ProjectorType.LINEAR
    projector_input_size: Optional[int] = None
    projector_output_size: Optional[int] = None
    projector_tie_weights: Optional[bool] = True
    full_mask_image_tokens: Optional[bool] = None


def postprocess_args(training_args: TrainingArguments, model_args: ModelArguments, data_args: DataArguments):
    training_args.iwm_loss = model_args.iwm_loss
    training_args.instruction_tuning = model_args.instruction_tuning
    training_args.train_proj_only = model_args.train_proj_only

    data_args.iwm_loss = model_args.iwm_loss
    data_args.instruction_tuning = model_args.instruction_tuning    

    return training_args, model_args, data_args