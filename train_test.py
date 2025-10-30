from pathlib import Path
from typing import List
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# This function counts the number of .jpg files in the indicated folder.
def count_jpg_in_folder(folder_name) -> int:
    base_dir = Path(__file__).parent
    tmp_dir = base_dir / folder_name
    if not tmp_dir.exists() or not tmp_dir.is_dir():
        print(f"`{tmp_dir} not exists or is not a directory.`")
        return 0
    count = sum(1 for p in tmp_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
    print(f"Found {count} .jpg files in `{tmp_dir}`.")
    return count

def get_dirs_with_jpg(base_folder_name: str = "tmp") -> List[str]:
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
    def __init__(self, image_paths, processor, policy, image_size=224):
        self.image_paths = image_paths
        self.processor = processor
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
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": policy},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        return inputs

if __name__ == "__main__":
    # Example usage
    folder_name = "tmp"
    count_jpg_in_folder(folder_name)
    image_paths = get_dirs_with_jpg(folder_name)

    policy = "Describe the content of the image."

    processor = AutoProcessor.from_pretrained("llava-onevision-model")
    dataset = PolicyImageDataset(image_paths, processor, policy)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        print(batch)
        break


