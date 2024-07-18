import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_path = r"C:\Users\prash\OneDrive\Desktop\chatbot\gpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

def generate_response(question):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    
    # Find the last full stop and truncate the answer there
    if '.' in answer:
        last_full_stop = answer.rfind('.')
        answer = answer[:last_full_stop + 1]
    
    return answer

# Read user message from the temporary file
with open("user_message.txt", "r") as file:
    user_message = file.read()

# Generate the response
response = generate_response(user_message)

# Print the response (will be captured by PHP)
print(response)
