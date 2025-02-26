import time
import torch
import argparse
import os
import gradio as gr
from transformers import AutoTokenizer
from model.model import LilLM
from model.config import Config

DEFAULT_TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/tokenizer')
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_sft_4epoch_with_hard_coded.pt')
DEFAULT_MODEL_TYPE = 'sft'

class ChatInterface:
    def __init__(self, tokenizer_path, model_path, model_type):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = LilLM(Config(flash=False))
        self.model.eval()
        self.model.to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model']
        
        # Remove unwanted prefix if present
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        self.model.load_state_dict(checkpoint['model'])
        self.eos = torch.tensor([[2]]).to(self.device)
    
    def add_chat_format(self, conversation):
        formatted_chat = ""
        for message in conversation:
            role = message["role"]
            content = message["content"]
            if role == "user":
                formatted_chat += f"<r0>user<r1>{content}</r2>"
            elif role == "assistant":
                formatted_chat += f"<r0>assistant<r1>{content}</s>"
        # Add the final assistant prompt to get model to generate a response
        formatted_chat += "<r0>assistant<r1>"
        return formatted_chat
    
    def generate_response(self, message, chat_history):
        # Add the new user message to the chat history
        if chat_history is None:
            chat_history = []
        
        # Format conversation for model input
        conversation = []
        for user_msg, assistant_msg in chat_history:
            conversation.append({"role": "user", "content": user_msg})
            conversation.append({"role": "assistant", "content": assistant_msg})
        
        # Add the new message
        conversation.append({"role": "user", "content": message})
        
        # Format the conversation for the model
        template_text = self.add_chat_format(conversation) if self.model_type == "sft" else message
        
        # Generate response
        t0 = time.time()
        start_prompt = torch.tensor(self.tokenizer.encode(template_text)).unsqueeze(dim=0).to(self.device)
        generated_output = self.model.generate(start_prompt, self.eos).squeeze()
        response = self.tokenizer.decode(generated_output)
        # Extract just the assistant's response from the full output
        print('conversation', conversation)
        if self.model_type == "sft":
            # The response should be after the last assistant tag
            last_assistant_tag_pos = response.rfind("<r0> assistant<r1>") + len("<r0> assistant<r1>")
            response = response[last_assistant_tag_pos:].replace("</s>", "").strip()
            print('ggg response', response) 
        t1 = time.time()
        print(f'Generation completed in {t1-t0} seconds')
        
        # Update chat history
        chat_history.append((message, response))
        return "", chat_history

def create_chat_interface(tokenizer_path, model_path, model_type):
    chat_interface = ChatInterface(tokenizer_path, model_path, model_type)
    
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# LilLM Chat Interface")
        
        chatbot = gr.Chatbot(height=600)
        msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
        clear = gr.Button("Clear")
        
        msg.submit(
            chat_interface.generate_response, 
            [msg, chatbot], 
            [msg, chatbot]
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        
    return demo

if __name__ == '__main__':

    torch.serialization.add_safe_globals([Config])
    parser = argparse.ArgumentParser(description='LilLM Chat Interface')
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="Tokenizer path")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, help="sft model or pretrained")
    args = parser.parse_args()
    
    # Update the ChatInterface class with command line arguments

    # Launch the interface
    demo = create_chat_interface(args.tokenizer_path, args.model_path,args.model_type )
    demo.launch(share=False)
