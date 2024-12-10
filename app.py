
import gradio as gr
import requests
import pytesseract
from PIL import Image
import io
import speech_recognition as sr
from gtts import gTTS
from langdetect import detect
import tempfile
import os

# Set your Google API Key
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
API_URL = "https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText"

# Global variable to store the extracted medical report text
extracted_report_text = ""

# Function to analyze medical report
def analyze_report(image):
    global extracted_report_text
    try:
        if image is None:
            return "Please upload a valid image."

        # Convert image to a compatible format if necessary
        if not image.format or image.format.lower() not in ['png', 'jpeg', 'jpg']:
            image = image.convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        if not extracted_text.strip():
            return "No text detected in the uploaded image. Please upload a clearer image."

        # Save extracted text for chatbot context
        extracted_report_text = extracted_text

        return (
            "Medical report text has been successfully extracted. "
            "You can now ask questions about the report in the chatbot section."
        )
    except Exception as e:
        return f"Error: {str(e)}"

# Function to process voice input and return voice output
def process_voice(input_audio):
    global extracted_report_text
    try:
        # Convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(input_audio) as source:
            audio = recognizer.record(source)
        user_input = recognizer.recognize_google(audio)

        # Detect language
        detected_language = detect(user_input)

        # Check if medical report is uploaded
        if not extracted_report_text:
            return "No medical report has been uploaded yet. Please upload a report before asking questions.", None

        # Prepare prompt for chatbot
        prompt = (
            f"You are a friendly and knowledgeable doctor. Use the following medical report text to answer the user's "
            f"questions in a simple and polite manner. Respond in {detected_language}:\n"
            f"Medical Report:\n{extracted_report_text}\n\n"
            f"User Question: {user_input}\nDoctor:"
        )

        # Send the request to the PaLM API
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "text-bison-001",
            "prompt": prompt,
            "temperature": 0.7,
            "candidateCount": 1,
        }

        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            headers=headers,
            json=payload,
        )

        # Handle API response
        if response.status_code != 200:
            return f"Error: Received a bad response from the API. Status code: {response.status_code}", None

        response_data = response.json()
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            response_text = response_data["candidates"][0]["output"]
        else:
            return "Unable to process your question. Please try again.", None

        # Convert response text to audio in the same language
        tts = gTTS(response_text, lang=detected_language)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio_file.name)

        return response_text, temp_audio_file.name

    except sr.UnknownValueError:
        return "Sorry, I could not understand your speech. Please try again.", None
    except Exception as e:
        return f"Error: {str(e)}", None

# Create a Gradio Interface
with gr.Blocks() as doctor_app:
    gr.Markdown("# 🩺 Doctor's Assistant App")
    gr.Markdown("Analyze medical reports and chat with a virtual doctor using your voice. Speak in your preferred language!")

    # Section 1: Report Analysis
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Upload a Medical Report Image for Analysis")
            image_input = gr.Image(label="Upload Medical Report", type="pil")
            analyze_button = gr.Button("Analyze")
        with gr.Column():
            analysis_output = gr.Textbox(label="Analysis Result", lines=5)
    
    analyze_button.click(analyze_report, inputs=image_input, outputs=analysis_output)

    # Section 2: Voice Chatbot
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Ask Your Question via Voice")
            voice_input = gr.Audio(label="Record Your Question", type="filepath")
            chat_button = gr.Button("Send")
        with gr.Column():
            chat_output = gr.Textbox(label="Doctor's Reply", lines=5)
            voice_output = gr.Audio(label="Doctor's Voice Reply")

    chat_button.click(process_voice, inputs=voice_input, outputs=[chat_output, voice_output])

# Launch the app
doctor_app.launch()
