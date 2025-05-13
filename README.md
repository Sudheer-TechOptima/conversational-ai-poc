# Voice-to-Voice AI Chatbot

An intelligent, real-time voice interaction system that enables natural conversations with AI using speech. This application combines efficient speech processing with GPT-powered responses, delivering a seamless voice-based chat experience through an intuitive Streamlit interface.

## Project Overview

The Voice-to-Voice AI Chatbot creates natural, flowing conversations between users and AI using voice interaction. The system processes voice input in real-time, converts it to text, generates intelligent responses using GPT models, and delivers these responses through synthesized speech.

### Core Pipeline
1. **Voice Activity Detection (VAD)**: Uses WebRTC VAD for efficient speech detection
2. **Speech-to-Text**: Employs OpenAI's Whisper model for accurate speech recognition
3. **AI Processing**: Leverages OpenAI's GPT models for intelligent responses
4. **Text-to-Speech**: Utilizes pyttsx3 for local speech synthesis

## Features

### Real-time Voice Interaction
- Automatic speech detection and processing
- Continuous conversation flow
- Low-latency response generation

### Advanced Speech Recognition
- State-of-the-art Whisper model
- Support for multiple languages
- High accuracy transcription

### Customizable GPT Integration
- Support for different GPT models
- Adjustable system prompts
- Conversation history management

### User-friendly Interface
- Clean, intuitive Streamlit UI
- Real-time status indicators
- Easy configuration options
- Chat history with download capability

## Technical Architecture

The application is built using Python and consists of several key components:

### Components
- **AudioHandler**: Manages audio input and VAD
- **SpeechToText**: Handles speech recognition using Whisper
- **ChatGPT**: Processes text through OpenAI's API
- **TextToSpeech**: Manages speech synthesis
- **Streamlit Interface**: Provides the user interface

### Data Flow
1. Audio input → VAD processing
2. Speech detection → Whisper transcription
3. Text → GPT processing
4. Response → Speech synthesis
5. Audio output to user

## Prerequisites

- Python 3.x
- Operating System: Windows/Linux/MacOS
- Microphone input device
- Audio output capability
- Internet connection (for GPT API)
- CUDA-capable GPU (optional, for faster transcription)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd voice-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up OpenAI API key:
- Create a `.env` file
- Add your API key: `OPENAI_API_KEY=your-api-key`

## Configuration

### Audio Settings
- Sample Rate: 16000 Hz
- VAD Mode: Adjustable (1-3)
- Silence Threshold: Configurable
- Speech Rate: Adjustable

### Model Settings
- Whisper Model Selection (small.en)
- GPT Model Selection
- System Prompt Customization
- Voice Detection Sensitivity

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configuration (via sidebar):
- Set OpenAI API key
- Choose GPT model
- Adjust voice settings
- Configure VAD parameters

3. Basic Operation:
- Click "Start" to begin listening
- Speak naturally when "Listening" is shown
- Wait for AI response
- Conversation continues automatically

4. Additional Features:
- Download chat history
- Clear conversation
- Adjust voice and VAD settings in real-time

## Known Limitations

- Requires internet connection for speech recognition
- Basic text-to-speech quality
- Requires consistent internet for GPT API
- Audio processing dependent on system capabilities

## Future Recommendations

1. Enhanced Features
- Custom TTS voices
- Advanced TTS with emotion
- Custom wake word detection
- Speech sentiment analysis
- Voice style customization

2. Technical Improvements
- Offline mode support
- WebSocket implementation
- Audio streaming optimization
- Advanced noise reduction
- Cloud deployment options

3. UI Enhancements
- Real-time audio visualization
- Voice activity graphs
- Response confidence indicators
- Theme customization
- Mobile responsiveness

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [OpenAI](https://openai.com) for GPT integration
- [Streamlit](https://streamlit.io) for the UI framework
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) for voice activity detection
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for text-to-speech

## Support

For issues, questions, or contributions:
- Create an issue in the repository
- Contact the maintainers
- Check the [Streamlit Community](https://discuss.streamlit.io) for UI-related questions
- Visit [OpenAI Documentation](https://platform.openai.com/docs) for API-related queries