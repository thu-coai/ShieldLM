import torch

from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import Dataset

class SafetyDatasetDecoderOnly(Dataset):
    
    # for decoder-only models like gpt and llama

    def __init__(self,
                 args,
                 data,
                 ):
       
        self.max_ength = args.max_length

        self.args = args
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        if self.tokenizer.eos_token is None:
            print('eos token is None. set eos token !!!')
            # for qwen
            self.tokenizer.eos_token = '<|endoftext|>'
        

    def __len__(self) -> int:
        return len(self.data)

    def tokenize_text(self, text: str, max_length, padding='max_length', add_special_tokens=False, add_eos_token_id=False) -> tuple:
        encoded_inputs = self.tokenizer(text, add_special_tokens=add_special_tokens, max_length=max_length, padding=padding, truncation=True)
        if add_eos_token_id:
            input_ids = torch.LongTensor(encoded_inputs['input_ids'] + [self.tokenizer.eos_token_id])
            mask = torch.LongTensor(encoded_inputs['attention_mask'] + [1])
        else:
            input_ids = torch.LongTensor(encoded_inputs['input_ids'])
            mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        d = self.data[idx]
        
        text_input = d['input']
        
        output_text = d['output']
        
        input_ids, _  = self.tokenize_text(text_input, self.args.max_length, padding=False, add_special_tokens=False)
        output_ids, _ = self.tokenize_text(output_text, self.args.max_length, padding=False, add_special_tokens=False, add_eos_token_id=True)
        concat_input_ids = torch.cat([input_ids, output_ids], dim=-1)
        tot_max_len = self.args.max_length
        if len(concat_input_ids) < tot_max_len:
            padded_tokens = torch.full((tot_max_len - len(concat_input_ids), ), fill_value=self.tokenizer.eos_token_id)
            padded_input_ids = torch.cat([concat_input_ids, padded_tokens], dim=-1)
        else:
            padded_input_ids = concat_input_ids[:tot_max_len]
        
        output_ids = padded_input_ids.clone()
        concat_len = len(concat_input_ids)
        output_ids[concat_len:] = -100
        
        input_len = len(input_ids)
        output_ids[:input_len] = -100

        data = dict(
            input_ids=padded_input_ids,
            labels=output_ids
        )

        return data
    
