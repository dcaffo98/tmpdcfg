from src.models import MLLMForCausalLM
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
import torch
from transformers import LlamaModel

CHECKPOINT = '/leonardo_scratch/large/userexternal/dcaffagn/checkpoints/jeppetto/vicuna_7B_stage_1__lr_1e_3__epochs_10/checkpoint-6543'

if __name__ == '__main__':
    device = 'cuda'
    dtype = torch.bfloat16
    # model = MLLMForCausalLM.from_pretrained(CHECKPOINT, lm_kwargs=dict(attn_implementation='eager'))
    model = MLLMForCausalLM.from_pretrained_MLLMForIWM(CHECKPOINT, lm_kwargs=dict(attn_implementation='eager'))
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    image_processor = AutoImageProcessor.from_pretrained(model.config.vision_config.name_or_path)
    model.to(device, dtype=dtype)

    while True:
        prompt = 'Describe the image.'
        msg = [
            dict(role='user', content=[
                dict(type='text', text=prompt),
                dict(type='image')
            ]),
            # dict(role='assistant', content=
            #     dict(text='The image depicts a dog.'),
            # ),
            # dict(role='user', content=[
            #     dict(type='text', text='Are you sure that the image shows a dog?'),
            # ])
        ]
        input_ids = tokenizer.apply_chat_template(msg, add_generation_prompt=True, return_tensors='pt').to(device=device)
        attention_mask = torch.ones_like(input_ids, device=device, dtype=torch.long)
       
        # prompt = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        # prompt += "The image depicts "
        # prompt = "You are given the following image:<image>\nPlease tell me what do you see."
        
        inputs = tokenizer([prompt], return_tensors='pt').to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        img = Image.open('.scratch/images/pizza.jpg')
        pixel_values = image_processor([img], return_tensors='pt').to(dtype=dtype, device=device).pixel_values

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                num_beams=5,
                num_return_sequences=5,
                do_sample=True,
                max_new_tokens=30,
                # temperature=3.
            )
        from transformers import LlamaModel
        answers = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        for answer in answers:
            print(answer)
        ...
        
