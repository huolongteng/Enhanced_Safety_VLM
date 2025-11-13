import json
from typing import List, Sequence, Tuple

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

        transform_steps = []
        if image_size is not None:
            transform_steps.append(transforms.Resize((image_size, image_size)))
        self.transform = transforms.Compose(transform_steps) if transform_steps else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
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

    image_paths: List[str] = []
    policy_texts: List[str] = []
    response_texts: List[str] = []

    for obj in data:
        input_payload = obj.get("input") or {}
        output_payload = obj.get("output")

        image_path = input_payload.get("image") if isinstance(input_payload, dict) else None
        policy_text = input_payload.get("policy") if isinstance(input_payload, dict) else None

        if isinstance(output_payload, dict):
            response_text = output_payload.get("response")
        else:
            response_text = output_payload

        if not (isinstance(image_path, str) and image_path.strip()):
            continue
        if not (isinstance(policy_text, str) and policy_text.strip()):
            continue
        if isinstance(response_text, str) and response_text.strip():
            final_response = response_text
        elif response_text is not None:
            final_response = json.dumps(response_text, ensure_ascii=False)
        else:
            continue

        image_paths.append(image_path)
        policy_texts.append(policy_text)
        response_texts.append(final_response)

    if not (len(image_paths) == len(policy_texts) == len(response_texts)):
        raise ValueError("Dataset fields have mismatched lengths after filtering invalid entries.")
    if not image_paths:
        raise ValueError("No valid samples were found in the dataset file.")

    return image_paths, policy_texts, response_texts


def _count_image_slots(conversation: Sequence[dict]) -> int:
    count = 0
    for message in conversation:
        if not isinstance(message, dict):
            continue
        contents = message.get("content") or []
        for item in contents:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
    return count


def _ensure_image_sequence(image) -> List:
    if isinstance(image, (list, tuple)):
        return list(image)
    return [image] if image is not None else []


def _infer_image_size(image) -> Tuple[int, int]:
    if hasattr(image, "height") and hasattr(image, "width"):
        return int(image.height), int(image.width)

    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        # PIL images expose (width, height)
        width, height = size
        return int(height), int(width)

    shape = getattr(image, "shape", None)
    if isinstance(shape, Sequence):
        if len(shape) == 2:
            height, width = shape
            return int(height), int(width)
        if len(shape) == 3:
            # Handle both (H, W, C) and (C, H, W)
            if shape[0] in (1, 3, 4):
                height, width = shape[1], shape[2]
            else:
                height, width = shape[0], shape[1]
            return int(height), int(width)

    raise ValueError("Unable to infer image size for processor consumption.")

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

        conversations = [sample["conversation"] for sample in batch]

        flat_images: List = []
        image_sizes: List[Tuple[int, int]] = []

        for sample, conversation in zip(batch, conversations):
            required_slots = _count_image_slots(conversation)
            sample_images = _ensure_image_sequence(sample.get("image"))

            if required_slots == 0:
                if sample_images:
                    raise ValueError(
                        "Sample contains image data but the conversation does not request images."
                    )
                continue

            if not sample_images:
                raise ValueError(
                    "Conversation expects images, but none were provided for the sample."
                )

            if len(sample_images) == 1 and required_slots > 1:
                sample_images = sample_images * required_slots
            elif len(sample_images) != required_slots:
                raise ValueError(
                    "Number of provided images does not match the required image slots in the conversation."
                )

            for image in sample_images:
                flat_images.append(image)
                image_sizes.append(_infer_image_size(image))
        prompts = apply_chat_template_to_batch(
            conversations,
            processor,
            add_generation_prompt=add_generation_prompt,
        )

        processor_inputs = dict(processor_kwargs)
        provided_image_sizes = processor_inputs.pop("image_sizes", None)
        processor_inputs.pop("images", None)

        processor_inputs.update(
            {
                "text": prompts,
                "padding": padding,
                "return_tensors": return_tensors,
            }
        )

        if flat_images:
            processor_inputs["images"] = flat_images
            processor_inputs["image_sizes"] = provided_image_sizes or image_sizes
        elif provided_image_sizes is not None:
            processor_inputs["image_sizes"] = provided_image_sizes

        return processor(**processor_inputs)

    return collate_fn


def create_dataloader(
    json_path,
    processor,
    *,
    batch_size=1,
    shuffle=False,
    image_size=256,
    add_generation_prompt=False,
    padding=False,
    return_tensors="pt",
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    persistent_workers=False,
    **processor_kwargs,
):
    """Create a ``torch.utils.data.DataLoader`` ready for training or evaluation.

    Parameters
    ----------
    json_path: str or pathlib.Path
        Path to the JSON dataset file containing ``input.image``, ``input.policy``
        and ``output.response`` entries.
    processor: callable
        Processor (typically a HuggingFace processor) used to collate samples into
        tensors. It must expose an ``apply_chat_template`` method.
    batch_size: int, optional
        Number of samples to load per batch. Defaults to ``1``.
    shuffle: bool, optional
        Whether to shuffle the dataset at every epoch. Defaults to ``False``.
    image_size: int, optional
        Target square size for the images before feeding them to the processor.
    add_generation_prompt: bool, optional
        Forwarded to :func:`build_policy_collate_fn` so the processor can append a
        generation prompt when required.
    padding, return_tensors: optional
        Passed through to the processor when batching.
    num_workers, pin_memory, drop_last, persistent_workers: optional
        Standard ``DataLoader`` arguments forwarded as-is.
    **processor_kwargs
        Additional keyword arguments passed directly to the processor.

    Returns
    -------
    DataLoader
        A PyTorch dataloader yielding batches of processed multimodal examples.
    """

    image_paths, policy_texts, response_texts = read_images_policies_responses_paths(json_path)

    if not (len(image_paths) == len(policy_texts) == len(response_texts)):
        raise ValueError(
            "Dataset fields have mismatched lengths: "
            f"images={len(image_paths)}, policies={len(policy_texts)}, responses={len(response_texts)}"
        )

    dataset = PolicyImageDataset(
        image_paths,
        policy_texts,
        response_texts,
        image_size=image_size,
    )

    collate_fn = build_policy_collate_fn(
        processor,
        add_generation_prompt=add_generation_prompt,
        padding=padding,
        return_tensors=return_tensors,
        **processor_kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    return dataloader

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



