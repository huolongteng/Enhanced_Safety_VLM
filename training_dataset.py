import json
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class PolicyImageDataset(Dataset):
    """
    Dataset that pairs images with policy and response texts for supervised training.

    Each dataset item returns:
    - "image": a transformed PIL Image (RGB) resized to the specified image_size.
    - "conversation": a list with two messages:
        1) user message containing the image marker and the policy text.
        2) assistant message containing the response text.

    Designed for use with PyTorch DataLoader to feed image+text conversational examples
    into multimodal training or evaluation pipelines.
    """
    def __init__(self, image_paths, policy_lists, response_lists, image_size=256):
        self.image_paths = image_paths
        self.policy_lists = policy_lists
        self.response_lists = response_lists
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        policy = self.policy_lists[idx]
        response = self.response_lists[idx]
        return {
            "image": image,
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": policy},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response},
                    ],
                },
            ],
        }


def read_images_policies_responses_paths(json_path):
    """
    Read image paths, policy texts, and response texts from a JSON dataset file.

    The JSON file is expected to contain a list of objects, each with:
    - 'input.image': path to the image file
    - 'input.policy': policy text
    - 'output.response': response text

    Returns three lists: image paths, policy texts, and response texts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_paths = []
    policy_texts = []
    response_texts = []
    for obj in data:
        image_path = (obj.get("input") or {}).get("image")
        policy_text = (obj.get("input") or {}).get("policy")
        if isinstance(image_path, str) and image_path.strip():
            image_paths.append(image_path)
        if isinstance(policy_text, str) and policy_text.strip():
            policy_texts.append(policy_text)
        response_text = (obj.get("output") or {})
        if isinstance(response_text, str) and response_text.strip():
            response_texts.append(response_text)

    return image_paths, policy_texts, response_texts

def apply_chat_template_to_batch(conversations, processor, add_generation_prompt=False):
    """Apply the chat template to a batch of conversations."""
    prompts = []
    for conv in conversations:
        prompt = processor.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        prompts.append(prompt)
    return prompts

def build_policy_collate_fn(
        processor,
        add_generation_prompt=False,
        padding=False,
        return_tensors="pt",
        **processor_kwargs
):
    """Create a collate function that batches samples with the processor."""

    def collate_fn(batch):
        if not batch:
            raise ValueError("Received an empty batch.")

        images = [sample["image"] for sample in batch]
        conversations = [sample["conversation"] for sample in batch]
        prompts = apply_chat_template_to_batch(
            conversations,
            processor,
            add_generation_prompt=add_generation_prompt,
        )

        return processor(
            text=prompts,
            images=images,
            padding=padding,
            return_tensors=return_tensors,
            **processor_kwargs,
        )

    return collate_fn


def create_dataloader():
    return None

def rewrite_image_paths(json_path, image_dir_prefix):
    """
    Rewrite image paths for items in a JSON dataset.

    Reads a JSON file expected to contain a list of objects, ensures each object
    has an 'input' dictionary, and sets the 'input.image' field to the
    concatenation of image_dir_prefix and the object's 'id' with a '.jpg' suffix.
    Items without an 'id' are skipped. The modified list is written back to the
    same JSON file using UTF-8 encoding and pretty-printed JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    for obj in items:
        _id = str(obj.get("id", ""))
        if not _id:
            continue
        if "input" not in obj or not isinstance(obj["input"], dict):
            obj["input"] = {}
        obj["input"]["image"] = f"{image_dir_prefix}{_id}.jpg"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train_image_dir_prefix = "data/train/"
    test_image_dir_prefix = "data/test/"
    train_json_path = "data/train_dataset.json"
    test_json_path = "data/test_dataset.json"
    # rewrite_image_paths(train_json_path, train_image_dir_prefix)
    # rewrite_image_paths(test_json_path, test_image_dir_prefix)
    tarin_image_paths, train_policy_lists, train_response_lists = read_images_policies_responses_paths(train_json_path)
    test_image_paths, test_policy_lists, test_response_lists = read_images_policies_responses_paths(test_json_path)
    train_dataset = PolicyImageDataset(tarin_image_paths, train_policy_lists, train_response_lists, image_size=256)
    test_dataset = PolicyImageDataset(test_image_paths, test_policy_lists, test_response_lists, image_size=256)



