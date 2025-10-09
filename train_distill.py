# Import all needed packages.
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F


policy = """
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.

O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.

O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.

O4: Nudity Content: 
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.

O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.

O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.

O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.

O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.

O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe, referring to the specific policy category and its associated guidelines to illustrate any violations or compliance.
To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
    "rationale": str,
}
"""


def load_model_and_processor(teacher_path, student_path):
    T_ = LlavaOnevisionForConditionalGeneration.from_pretrained(teacher_path)
    S_ = LlavaOnevisionForConditionalGeneration.from_pretrained(student_path)
    P_ = AutoProcessor.from_pretrained(teacher_path)
    return T_, S_, P_

def generate_model_inputs(image_path, processor):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": policy},
                ],
        },
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    inputs = processor(text=text_prompt, images=image, return_tensors="pt")
    return inputs


def build_image_list(image_dir, allowed_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in allowed_extensions and path.is_file()
    )

    if not image_paths:
        raise ValueError(f"No images with extensions {allowed_extensions} found in {image_dir}")

    return image_paths


def cal_kl_loss(inputs, teacher_model, student_model, T, device):
    # Move inputs to the computation device.
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 教师模型前向传播，得到logits.
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits
        

    student_outputs = student_model(**inputs)
    student_logits = student_outputs.logits

    # Delete middle Vs.
    del student_outputs, inputs, teacher_outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_probs = (teacher_logits / T).softmax(dim=-1)
    student_probs = (student_logits / T).log_softmax(dim=-1)
    
    loss = F.kl_div(student_probs, teacher_probs, reduction='none') * (T * T)
    
    loss = loss.mean()

    del student_probs, teacher_probs, teacher_logits, student_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distill Llava safety model")
    parser.add_argument("--image_dir", required=True, help="Directory containing training images")
    parser.add_argument("--teacher_model", default="AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf")
    parser.add_argument("--student_model", default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None, help="Optional limit on number of steps for quick experiments")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model, student_model, processor = load_model_and_processor(
        args.teacher_model, args.student_model
    )
    teacher_model.eval()
    student_model.train()
    teacher_model.to(device)
    student_model.to(device)

    image_list = build_image_list(args.image_dir)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)

    step = 0
    optimizer.zero_grad()
    for idx, image_path in enumerate(image_list, start=1):
        if args.max_steps is not None and step >= args.max_steps:
            break

        inputs = generate_model_inputs(image_path, processor)
        loss = cal_kl_loss(inputs, teacher_model, student_model, args.temperature, device)
        loss.backward()
        step += 1

        if idx % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"Step {step}: loss={loss.item():.6f} (image: {image_path.name})")

    # Flush remaining gradients if the total number of images is not a multiple of grad_accum_steps.
    if step % args.grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    print("Distillation finished. Total steps:", step)




    