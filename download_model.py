import os
import argparse

from huggingface_hub import hf_hub_download

DEFAULT_REPO_ID = "jonwondo/lilLM_40M_param_10B_tok"
DEFAULT_FILENAME = "lilLM_40M_params_10B_tok.pt"



if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Download model for inference/sft training')
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="Huggingface repo id exp: jonwondo/lilLM_40M_param_10B_tok")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="file name inside that repo id exp: lilLM_40M_params_10B_tok.pt")
    args = parser.parse_args()
# Define the repository ID and file name
    current_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=current_dir)

    print(f"Model downloaded to: {model_path}")
    os.rename(args.filename, 'best_model.pt')

    print(f"File renamed from '{args.filename}' to best_model.pt")


