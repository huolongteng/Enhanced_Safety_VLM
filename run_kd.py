import dataset, load_models, train
import random
import torch
import json


def same_seeds(seed=443):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None

if __name__ == '__main__':
    same_seeds(2025)
    with open("policy.json") as f:
        policy_data = json.load(f)
        policy = policy_data["policy"][0] # Use the first policy for all images.


