# Step 1: Install required packages
# pip install transformers torch gradio

# Step 2: Import libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import gradio as gr

# Step 3: Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Step 4: Define the text generation function
def generate_text(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 5: Create the Gradio Interface
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Enter your prompt"),
        gr.Slider(20, 200, value=100, label="Max Length")
    ],
    outputs="text",
    title="GPT-2 Text Generator",
    description="Generate text using the GPT-2 model. Type a prompt and set a max length."
)

# Step 6: Launch the Interface
interface.launch()