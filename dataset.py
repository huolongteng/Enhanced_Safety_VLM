# This block is about dataset handling and preprocessing for a machine learning project.
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Any, Callable, List, Optional, Sequence

def count_jpg_in_folder(folder_name):
    """
    Counts the number of .jpg files in the specified folder.

    Args:
        folder_name (str): The path to the folder.
    """
    base_dir = Path(__file__).parent
    tmp_dir = base_dir / folder_name
    if not tmp_dir.exists() or not tmp_dir.is_dir():
        print(f"`{tmp_dir} not exists or is not a directory.`")
        return 0
    count = sum(1 for p in tmp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    print(f"Found {count} .jpg files in `{tmp_dir}`.")
    return count

def get_dirs_with_jpg(base_folder_name):
    """
        Returns a list of the absolute paths of all '.jpg' or '.jpeg' image files
        (recursively) under `base_folder_name`.
        """
    base_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    target_dir = base_dir / base_folder_name
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"`{target_dir}` does not exist or is not a directory.")
        return []

    image_paths: List[str] = []
    for p in target_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"):
            image_paths.append(str(p.resolve()))

    print(f"Found {len(image_paths)} .jpg/.jpeg files under `{target_dir}`.")
    return image_paths

# This Dataset class loads images and prepares them with the given policy prompt.
class PolicyImageDataset(Dataset):
    def __init__(self, image_paths, policy, image_size=256):
        self.image_paths = image_paths
        self.policy = policy
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.5, 0.5, 0.5),  # Maybe use ImageNet mean and std instead.
            #     std=(0.5, 0.5, 0.5)
            # ),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.policy},
                    ],
                },
            ],
        }
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
