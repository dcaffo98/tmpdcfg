from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List
from torch import Tensor
import torch
import src.custom_utils as utils

logger = utils.get_logger()


DEFAULT_IMAGE_TOKEN = '<image>'


class CONV_TEMPLATES(StrEnum):
    PLAIN = 'plain'
    PHI4 = 'phi4'
    LLAMA3_1 = 'llama3_1'


@dataclass
class Conversation:
    '''
    - A `role` is defined as `[<role_start>]ROLE[<role_end>]`.
    - A `sep` is defined as the string after a role + message: `[<role_start>]ROLE[<role_end>]Here is the message<sep>`. 
     Each role has its own separator.
    '''
    roles: Dict[str, str]
    seps: Dict[str, str] = field(default_factory=lambda: defaultdict(str))
    messages: List[str] = field(default_factory=list)
    sys_prompt: str = ''
    img_tok: str = DEFAULT_IMAGE_TOKEN
    bos_tok: str = ''
    eos_tok: str = ''
    eot_tok: str | None = None
    assistant_toks: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.eot_tok is None:
            self.eot_tok = self.seps['assistant']

    def new_empty(self):
        return Conversation(
            roles=self.roles,
            seps=self.seps,
            messages=[],
            sys_prompt=self.sys_prompt,
            img_tok=self.img_tok,
            bos_tok=self.bos_tok,
            eos_tok=self.eos_tok,
            assistant_toks=self.assistant_toks
        )
    
    def reset(self):
        self.messages = []

    def add_message(self, msg: Dict[str, str]):
        role = msg['role']
        ret = self.roles[role]
        content = msg.get('content')
        if content is None and role == 'system':
            content = self.sys_prompt
        if content:
            ret += content
            ret += self.seps[role]
        self.messages.append(ret)

    def add_sys_prompt(self):
        self.add_message(dict(role='system'))

    def get_prompt(self, add_generation_prompt=False):
        if add_generation_prompt:
            self.add_message(dict(role='assistant'))
            ret = self.bos_tok + ''.join(self.messages)
        else:
            ret = self.bos_tok + ''.join(self.messages) + self.eos_tok
        return ret
    
    def get_labels(self, input_ids: Tensor, tokenized_text: List[str], ignore_index: int = -100):
        labels = [ignore_index] * len(input_ids)
        assistant_len = len(self.assistant_toks)
        i = 0
        maxlen = len(tokenized_text)
        while i < maxlen:
            tok = tokenized_text[i]
            if tok == self.assistant_toks[0]:
                if tokenized_text[i:i + assistant_len] == self.assistant_toks:
                    i += assistant_len
                    while i < maxlen and tokenized_text[i] != self.eot_tok:
                        labels[i] = input_ids[i]
                        i += 1
                    if i < maxlen:
                        labels[i] = input_ids[i]
            i += 1
        if i != len(input_ids):
            logger.warning(f"Tokenization mismatch: expected {len(input_ids)} tokens, but found {i}. Setting all labels to `ignore_index` {ignore_index}.")
            labels = [ignore_index] * len(input_ids)
        return torch.tensor(labels, dtype=input_ids.dtype)


CONV_MAPPING = {
    CONV_TEMPLATES.PHI4: Conversation(
        roles={
            'system': '<|system|>',
            'user': '<|user|>',
            'assistant': '<|assistant|>'
        },
        seps=defaultdict(lambda: '<|end|>'),
        sys_prompt='You are a helpful AI assistant.',
        img_tok=DEFAULT_IMAGE_TOKEN,
        eos_tok='<|endoftext|>',
        assistant_toks=['<|assistant|>']
    ),

    CONV_TEMPLATES.LLAMA3_1: Conversation(
        roles={
            'system': '<|start_header_id|>system<|end_header_id|>\n\n',
            'user': '<|start_header_id|>user<|end_header_id|>\n\n',
            'assistant': '<|start_header_id|>assistant<|end_header_id|>\n\n'
        },
        seps=defaultdict(lambda: '<|eot_id|>'),
        sys_prompt='You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.',
        img_tok=DEFAULT_IMAGE_TOKEN,
        bos_tok='<|begin_of_text|>',
        eos_tok='<|end_of_text|>',
        assistant_toks=['<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ']
    )    
}


if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    MODEL_NAME_OR_PATHS = (
        'microsoft/Phi-4-mini-instruct',
        'meta-llama/Llama-3.1-8B-Instruct'
    )
    conv_modes = (
        'phi4', 
        'llama3_1'
    )

    for model_name_or_path, conv_mode in zip(MODEL_NAME_OR_PATHS, conv_modes):

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)

        conv = CONV_MAPPING[conv_mode].new_empty()

        msgs = [
            dict(role='system', content='sysprompt test.'),
            dict(role='user', content='<image>Please describe the image.'),
            dict(role='assistant', content='The image depicts a dog.'),
            dict(role='user', content='What is the color of the dog?'),
            dict(role='assistant', content='Brown.')
        ]

        for msg in msgs:
            conv.add_message(msg)

        prompt = conv.get_prompt()
        gt_prompt = tokenizer.apply_chat_template([msgs], add_generation_prompt=False, tokenize=False)[0]

        print(prompt)

        input_ids = tokenizer([prompt], return_tensors='pt').input_ids[0]
        gt_input_ids = tokenizer([gt_prompt], return_tensors='pt').input_ids[0]

        labels = conv.get_labels(input_ids, tokenizer.convert_ids_to_tokens(input_ids))
        gt_labels = conv.get_labels(gt_input_ids, tokenizer.convert_ids_to_tokens(gt_input_ids))

        # assert prompt == gt_prompt
        # assert (tokenizer([prompt], return_tensors='pt').input_ids == tokenizer([gt_prompt], return_tensors='pt').input_ids).all().item()

    