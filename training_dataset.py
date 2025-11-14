import json
from typing import List, Optional, Sequence, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class PolicyImageDataset(Dataset):
    """
    Dataset that pairs images with policy and response texts for supervised training.

    Each dataset item returns:
    - "image": a transformed PIL Image (RGB) resized to the specified image_size.
    - "policy": the user policy/instruction text paired with the image.
    - "response": the assistant response text that acts as the supervision label.

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
            "policy": policy,
            "response": response,
        }


def read_images_policies_responses_paths(json_path):
    """
    Read image paths, policy texts, and response texts from a JSON dataset file.

    The JSON file is expected to contain a list of objects, each with:
    - 'input.image': path to the image file
    - 'input.policy': policy text
    - 'output': response text (stored as a JSON-formatted string)

    Returns three lists: image paths, policy texts, and response texts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_paths = []
    policy_texts = []
    response_texts = []
    for obj in data:
        input_payload = obj.get("input") or {}
        response_payload = obj.get("output")

        image_path = input_payload.get("image")
        policy_text = input_payload.get("policy")
        if not (
            isinstance(image_path, str)
            and image_path.strip()
            and isinstance(policy_text, str)
            and policy_text.strip()
            and isinstance(response_payload, str)
            and response_payload.strip()
        ):
            continue

        image_paths.append(image_path)
        policy_texts.append(policy_text)
        response_texts.append(response_payload)

    return image_paths, policy_texts, response_texts

def apply_chat_template_to_batch(
    conversations,
    processor,
    add_generation_prompt=False,
):
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
    **processor_kwargs,
):
    """Create a collate function that batches samples with the processor.

    The returned function assembles policy/response pairs into chat prompts and
    produces tensors ready for supervised learning. When ``return_tensors`` is
    ``"pt"`` and ``add_generation_prompt`` is ``False``, a ``labels`` tensor
    masking the user portion with ``-100`` is attached to the batch encoding so
    it can be fed directly to ``LlavaOnevisionForConditionalGeneration`` style
    models.
    """

    def collate_fn(batch):
        if not batch:
            raise ValueError("Received an empty batch.")

        images = [sample["image"] for sample in batch]
        policies = [sample["policy"] for sample in batch]
        responses = [sample["response"] for sample in batch]

        user_messages: List[List[dict]] = []
        full_conversations: List[List[dict]] = []
        for policy_text, response_text in zip(policies, responses):
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": policy_text},
                ],
            }
            assistant_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response_text},
                ],
            }
            user_messages.append([user_message])
            full_conversations.append([user_message, assistant_message])

        if add_generation_prompt:
            text_prompts = apply_chat_template_to_batch(
                user_messages,
                processor,
                add_generation_prompt=True,
            )
            generation_prompts: Optional[List[str]] = None
        else:
            text_prompts = apply_chat_template_to_batch(
                full_conversations,
                processor,
                add_generation_prompt=False,
            )
            generation_prompts = apply_chat_template_to_batch(
                user_messages,
                processor,
                add_generation_prompt=True,
            )

        image_token = getattr(processor, "image_token", "<image>")
        aligned_images = []
        for sample_image, prompt in zip(images, text_prompts):
            token_uses = prompt.count(image_token)
            if token_uses == 0:
                raise ValueError(
                    "Chat template did not emit any image tokens; cannot align images."
                )
            if token_uses == 1:
                aligned_images.append(sample_image)
            else:
                aligned_images.append([sample_image.copy() for _ in range(token_uses)])

        # Pad tokenized prompts when tensors are requested to avoid shape
        # mismatches caused by variable text lengths across samples.
        effective_padding = padding
        if return_tensors == "pt" and padding is False:
            effective_padding = "longest"

        batch_encoding = processor(
            text=text_prompts,
            images=aligned_images,
            padding=effective_padding,
            return_tensors=return_tensors,
            **processor_kwargs,
        )

        if add_generation_prompt or return_tensors != "pt":
            return batch_encoding

        if not hasattr(processor, "tokenizer"):
            raise AttributeError(
                "Processor must expose a tokenizer to compute supervision labels."
            )

        labels = batch_encoding["input_ids"].clone()
        for index, prompt in enumerate(generation_prompts or []):
            prompt_ids = processor.tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
            prompt_length = min(len(prompt_ids), labels.size(1))
            labels[index, :prompt_length] = -100

        if "attention_mask" in batch_encoding:
            attention_mask = batch_encoding["attention_mask"]
            mask = attention_mask == 0
            if hasattr(labels, "masked_fill"):
                labels = labels.masked_fill(mask, -100)
            else:
                for row_idx in range(len(attention_mask)):
                    row_mask = mask[row_idx]
                    if isinstance(row_mask, (list, tuple)):
                        iterator = enumerate(row_mask)
                    else:
                        iterator = enumerate(getattr(row_mask, "data", row_mask))
                    for col_idx, should_mask in iterator:
                        if should_mask:
                            labels[row_idx, col_idx] = -100

        batch_encoding["labels"] = labels
        return batch_encoding

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
        and ``output`` entries.
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
        When ``return_tensors`` is ``"pt"`` and ``add_generation_prompt`` is
        ``False`` the batches also include ``labels`` tensors with the user
        prompt masked to ``-100`` for supervised learning.
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


