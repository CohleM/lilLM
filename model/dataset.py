from transformers import load_dataset
import torch


class SFTDataset:
    def __init__(self,tokenizer, max_seq_len, data_path = 'CohleM/lillm-sft-dataset'):
        self.data = load_dataset(data_path)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.tokenized_data = self.data.map(self._tokenize, num_proc = 8)
        
    def _add_chat_format(self, example):
        items = example['conversation']
        template = ""
        
        for item in items:

            if item['role'] == 'user':
                template += f"<r0>{item['role']}<r1>" + f"{item['content']}</r2>"
            elif item['role'] =='assistant':
                template += f"<r0>{item['role']}<r1>" + f"{item['content']}</s>"
        return template

    def _generate_loss_mask(self, tokenized_input):
        assistant_token = self.tokenizer.encode('<r0>assistant<r1>')
        end_token = self.tokenizer.encode('</s>')[0]

        assist_token_idx = [i+3 for i in range(len(tokenized_input)) if tokenized_input[i:i+3] == assistant_token]
        end_token_idx = [i for i,v in enumerate(tokenized_input) if v == end_token]

        loss_mask = [0]*len(tokenized_input)

        for i in range(len(assist_token_idx)):
            loss_mask[assist_token_idx[i]: end_token_idx[i] + 1] = [1]* (end_token_idx[i] - assist_token_idx[i] + 1)

        return loss_mask

        
    def _tokenize(self,example):
        template = self._add_chat_format(example)

        x = self.tokenizer.encode(template)
        x += (self.max_seq_len - len(x))* [0]
        
        X = torch.tensor(x[:-1], dtype=torch.long)
        Y = torch.tensor(x[1:], dtype=torch.long)
        
        loss_mask = self._generate_loss_mask(x[1:])
        
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        
        return {'X': X, 'Y': Y, 'loss_mask': loss_mask}
    
    
    def get_batch(self, split, batch_size):
        batches = torch.randint(0, self.data[split].num_rows, (batch_size,))
        out = self.tokenized_data[split][batches]
        
        return out['X'], out['Y'], out['loss_mask']
    
