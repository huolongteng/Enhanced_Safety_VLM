# This block is about dataset handling and preprocessing for a machine learning project.
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _ensure_posix_path(path_str: str) -> str:
    """Normalize Windows style paths to POSIX style strings."""

    return path_str.replace("\\", "/")


def _format_label_text(output: Dict[str, Any]) -> str:
    """Create a human readable label string from the output dictionary."""

    parts: List[str] = []
    rating = output.get("rating")
    if rating:
        parts.append(f"Rating: {rating}")

    category = output.get("category")
    if category:
        parts.append(f"Category: {category}")

    rationale = output.get("rationale")
    if rationale:
        parts.append(f"Rationale: {rationale}")

    if not parts:
        # Fall back to the original representation when structured fields are missing.
        return json.dumps(output, ensure_ascii=False)

    return "\n".join(parts)


def load_split_entries(dataset_json: Path | str, image_root: Path | str) -> List[Dict[str, Any]]:
    """Load and normalize dataset entries from a JSON split file.

    Args:
        dataset_json: Path to the dataset JSON file containing the split entries.
        image_root: Base path containing the image assets for the split.

    Returns:
        A list of dictionaries with normalized fields used by :class:`PolicyImageDataset`.
    """

    dataset_path = Path(dataset_json)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as fp:
        raw_entries = json.load(fp)

    normalized_entries: List[Dict[str, Any]] = []
    base_root = Path(image_root)
    for record in raw_entries:
        input_payload = record.get("input", {}) or {}
        output_payload = record.get("output", {})

        image_field = str(input_payload.get("image", ""))
        image_rel = _ensure_posix_path(image_field)

        policy_text = input_payload.get("policy", "")

        if isinstance(output_payload, str):
            try:
                output_payload = json.loads(output_payload)
            except json.JSONDecodeError:
                output_payload = {"response": output_payload}
        elif not isinstance(output_payload, dict):
            output_payload = {"response": output_payload}

        label_text = _format_label_text(output_payload)

        normalized_entries.append(
            {
                "id": record.get("id"),
                "image": image_rel,
                "policy": policy_text,
                "output": output_payload,
                "label_text": label_text,
                "image_root": base_root,
            }
        )

    return normalized_entries

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


def load_policy_text(policy_file: Path | str, index: int = 0) -> str:
    """Load the policy string located at ``index`` from ``policy_file``."""

    policy_path = Path(policy_file)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    with policy_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    policies: Sequence[str] = []
    if isinstance(data, dict):
        if "policy" in data:
            policies = data["policy"]
        elif "policies" in data:
            policies = data["policies"]
    elif isinstance(data, Sequence):
        policies = data

    if not policies:
        raise ValueError(f"No policies found in policy file: {policy_path}")

    if not 0 <= index < len(policies):
        index = 0

    return policies[index]


def gather_image_paths(base_folder_name: str, limit: Optional[int] = None) -> List[str]:
    """Collect the image paths under ``base_folder_name`` with an optional limit."""

    count_jpg_in_folder(base_folder_name)
    image_paths = get_dirs_with_jpg(base_folder_name)
    if not image_paths:
        raise ValueError(
            f"No .jpg/.jpeg files were found under directory: {base_folder_name}"
        )

    if limit is None or limit <= 0 or len(image_paths) <= limit:
        return image_paths

    return random.sample(image_paths, limit)

# This Dataset class loads images and prepares them with the given policy prompt.
class PolicyImageDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        image_root: Path | str,
        image_size: int = 256,
    ):
        self.samples = list(samples)
        self.image_root = Path(image_root)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.5, 0.5, 0.5),  # Maybe use ImageNet mean and std instead.
            #     std=(0.5, 0.5, 0.5)
            # ),
        ])

    def __len__(self):
        return len(self.samples)

    def _resolve_image_path(self, relative_path: str) -> Path:
        path_obj = Path(relative_path)
        if path_obj.is_absolute():
            return path_obj
        return self.image_root / path_obj

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self._resolve_image_path(sample["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample.get("policy", "")},
                ],
            }
        ]

        return {
            "image": image,
            "conversation": conversation,
            "label_text": sample.get("label_text", ""),
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
    add_generation_prompt: bool = False,
    padding: bool | str = False,
    return_tensors: str = "pt",
    **processor_kwargs,
):
    """Create a collate function that batches samples with the processor."""

    def collate_fn(batch):
        if not batch:
            raise ValueError("Received an empty batch.")

        images = [sample["image"] for sample in batch]
        conversations = [sample["conversation"] for sample in batch]
        targets = [sample.get("label_text", "") for sample in batch]

        prompts = apply_chat_template_to_batch(
            conversations,
            processor,
            add_generation_prompt=add_generation_prompt,
        )

        return processor(
            text=prompts,
            images=images,
            text_target=targets,
            padding=padding,
            return_tensors=return_tensors,
            **processor_kwargs,
        )

    return collate_fn


def create_policy_dataloader(
    dataset_json: Path | str,
    image_root: Path | str,
    processor,
    *,
    batch_size: int,
    image_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a dataloader for policy-conditioned image datasets with labels."""

    samples = load_split_entries(dataset_json, image_root)
    dataset = PolicyImageDataset(samples, image_root=image_root, image_size=image_size)

    collate_fn = build_policy_collate_fn(
        processor,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
