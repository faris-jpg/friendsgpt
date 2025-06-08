# In app.py
import gradio as gr
from model_utils import Interface

try:
    model_interface = Interface(ver='best')
    model_loaded = True
except Exception as e:
    print(f"Error initializing model interface: {e}")
    model_loaded = False

def generate_text_for_gradio(prompt: str, max_tokens: int): # No -> str return type hint for generators
    if not model_loaded:
        yield "Error: Model failed to load during initialization. Please check your model files and definitions."
        return
    
    if not prompt.strip():
        yield "Please enter a prompt to generate text."
        return

    try:
        for text_chunk in model_interface.generate_text(prompt, max_tokens):
            yield text_chunk
    except Exception as e:
        yield f"An error occurred during text generation: {e}"

iface = gr.Interface(
    fn=generate_text_for_gradio,
    inputs=[
        gr.Textbox(lines=2, label="Enter your prompt (e.g., 'JOEY: How you doin')", placeholder="Start typing..."),
        gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Maximum New Tokens")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="FriendsGPT Text Generator",
    description="Generate Friends-style dialogue using your trained language model. Characters will appear as they are generated."
)

iface.launch()