import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 480  # 30ms at 16kHz - optimal for VAD
VAD_MODE = 3  # Aggressiveness mode (1-3)
SILENCE_THRESHOLD = 0.3  # seconds (reduced for faster response)
MAX_SILENCE = 0.8  # Maximum silence duration before stopping (reduced for faster response)
MIN_SPEECH_DURATION = 0.3  # Minimum duration of speech to process

# Audio Device Configuration
DEFAULT_INPUT_DEVICE = None
FALLBACK_INPUT_DEVICES = []

# GPT Configuration
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise 
and natural, as they will be converted to speech."""

# TTS Configuration
TTS_RATE = 150  # Words per minute
TTS_VOLUME = 1.0  # Volume (0.0 to 1.0)

# UI Configuration
TITLE = "Voice Chatbot"
DESCRIPTION = """An interactive voice-based chatbot that can detect speech, 
process it with GPT, and respond verbally."""