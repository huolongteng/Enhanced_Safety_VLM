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

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image
import torch

teacher_model = LlavaOnevisionForConditionalGeneration.from_pretrained('E:/Downloads/models/LlavaGuard-v1.2-0.5B-OV-hf')
student_model = LlavaOnevisionForConditionalGeneration.from_pretrained('E:/Downloads/models/llava-onevision-qwen2-0.5b-ov-hf')
processor = AutoProcessor.from_pretrained('E:/Downloads/models/LlavaGuard-v1.2-0.5B-OV-hf')
teacher_model.eval()
student_model.train()

image_path = ('E:/Downloads/tmp/3019.jpg')
image = Image.open(image_path)

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
teacher_model.to('cuda')
inputs = {k: v.to('cuda') for k, v in inputs.items()}
with torch.no_grad():
    teacher_outputs = teacher_model(**inputs)
    teacher_logits = teacher_outputs.logits

teacher_model.to('cpu')
del teacher_outputs
torch.cuda.empty_cache()

student_model.to('cuda')
student_outputs = student_model(**inputs)
student_logits = student_outputs.logits


# student_model.to('cpu')
del student_outputs, inputs
torch.cuda.empty_cache()

import torch.nn.functional as F

T = 2.0
teacher_probs = (teacher_logits / T).softmax(dim=-1)
student_probs = (student_logits / T).log_softmax(dim=-1)

del teacher_logits, student_logits
torch.cuda.empty_cache()

loss = F.kl_div(teacher_probs, student_probs, reduction='batchmean') * (T * T)

del student_probs, teacher_probs
torch.cuda.empty_cache()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("JIJIJI")





