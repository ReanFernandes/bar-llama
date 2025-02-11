# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model.to('cuda')
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Respond to the Multistate Bar Exam question with a detailed explanation in the following MARKDOWN format, considering the question and answer choices provided:

## Domain
*The relevant legal domain for this question.*

## Chosen Option
*The letter corresponding to the correct answer choice.*

## Explanation

### Legal Concept
*The legal principle or doctrine at the heart of the question.*

### Fact Analysis
*A breakdown of the relevant facts presented in the question.*

### Rule Application
*How the legal concept applies to the facts in the question.*

### Legal Conclusion
*A concise statement explaining why the chosen option is the correct answer.*

Example 1: 
Question:
A neighbor frequently visited a woman who lived in the house next to him. The woman was separated from her husband. The husband resided with his mother but jointly owned the house in which the woman resided. Late one night, the neighbor and the woman were sitting and drinking on the bed in the woman's bedroom when the husband burst through the door and told the neighbor, "Get out." When the neighbor refused, the husband challenged him to go outside and "fight it out." The neighbor again refused. The husband then pulled a knife from his pocket and lunged at the neighbor. The neighbor grabbed a lamp, struck the husband on the head, and killed him. The neighbor is charged with murder. On a charge of murder, the neighbor should be found

Answer Choices:
A. not guilty, because the neighbor had as much right as the husband to be in the house.
B. not guilty, because the husband attacked the neighbor with a deadly weapon.
C. guilty, because the neighbor's presence in the woman's bedroom prompted the husband's attack.
D. guilty, because the neighbor's failure to obey the husband's order to leave the house made him a trespasser.

Response:

## Chosen Domain
Criminal_law

## Chosen Option
B

## Explanation

### Legal concept
Self-defense in criminal law, particularly the justification of using deadly force in response to an imminent threat of death or serious bodily harm.

### Fact analysis
The husband entered the bedroom where the neighbor and the separated wife were sitting and aggressively demanded the neighbor leave, escalating the confrontation by inviting him to fight outside. When the neighbor refused, the husband escalated further by drawing a knife and lunging at him.

### Rule application
Self-defense allows an individual to use reasonable force, including deadly force, to protect themselves if they reasonably believe it is necessary to prevent imminent death or serious bodily harm. In this scenario, the neighbor's use of a lamp to strike the husband, who was armed with a knife and lunging towards him, would likely be considered a reasonable response under imminent threat.

### Legal Conclusion
Given that the neighbor was defending himself against an unlawful and immediate threat posed by the husband with a deadly weapon, he acted within the bounds of self-defense. Therefore, the correct verdict would be not guilty based on self-defense, supporting option B.<|eot_id|><|start_header_id|>user<|end_header_id|>

A shareholder of a car manufacturing company wants to sue the company in federal court for breach of fiduciary duty. What information about the company, besides its place of incorporation, does the shareholder's attorney need in order to determine whether the parties are diverse?

Answer Choices:
A. Where the company distributes its cars.
B. Where the company has sufficient minimum contacts.
C. Where the company's headquarters is located.
D. Where the company's major place of manufacturing is located.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# using manual generation for peft model
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to(model.device)
generation_kwargs = {
    "do_sample": True,
    "top_k": 10,
    "num_return_sequences": 1,
    "max_new_tokens": 600,
    "temperature": 0.6,
}
#generate model response 
output = model.generate(**inputs,**generation_kwargs)

sequences = tokenizer.batch_decode(output[:, inputs['input_ids'].shape[1]:])

print(sequences[0])