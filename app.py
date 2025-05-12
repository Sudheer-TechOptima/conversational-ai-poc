import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from utils import AudioHandler, SpeechToText, TextToSpeech, ChatGPT
import config

def initialize_session_state():
    if 'audio_handler' not in st.session_state:
        st.session_state.audio_handler = AudioHandler()
    if 'stt' not in st.session_state:
        try:
            st.session_state.stt = SpeechToText()
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    if 'tts' not in st.session_state:
        st.session_state.tts = TextToSpeech()
    if 'chat' not in st.session_state:
        st.session_state.chat = ChatGPT()
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    if 'audio_visualization' not in st.session_state:
        st.session_state.audio_visualization = None
    if 'viz_key' not in st.session_state:
        st.session_state.viz_key = 0

def update_audio_visualization():
    if st.session_state.is_listening:
        audio_data = st.session_state.audio_handler.get_audio_data()
        if len(audio_data) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=audio_data,
                mode='lines',
                name='Audio Signal',
                line=dict(color='#00ff00')
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(visible=False),
                xaxis=dict(visible=False),
                showlegend=False
            )
            return fig
    return None

def process_audio_and_respond(audio_data, status_placeholder):
    """Process audio data and generate response"""
    if audio_data is not None and len(audio_data) > 0:
        status_placeholder.markdown("🔄 **Processing speech...**")
        
        # Transcribe audio
        text = st.session_state.stt.transcribe(audio_data)
        if text:
            # Add user message
            st.session_state.conversation.append({"role": "user", "content": text})
            with st.chat_message("user"):
                st.write(text)
            
            # Get GPT response
            status_placeholder.markdown("🤔 **Thinking...**")
            response = st.session_state.chat.get_response(text)
            st.session_state.conversation.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
            
            # Convert response to speech
            status_placeholder.markdown("🔊 **Speaking...**")
            try:
                st.session_state.tts.speak(response)
            except Exception as e:
                print(f"TTS Error: {e}")
    
    status_placeholder.markdown("⏸️ **Paused**")

def main():
    st.set_page_config(page_title=config.TITLE, layout="wide")
    
    # Initialize session state
    initialize_session_state()
    
    # Two-column layout for main content and settings
    col_main, col_settings = st.columns([2, 1])
    
    with col_main:
        st.title(config.TITLE)
        st.markdown(config.DESCRIPTION)
        
        # Status indicator and controls
        status_col, control_col = st.columns([3, 1])
        with status_col:
            status_placeholder = st.empty()
            if st.session_state.is_listening:
                status_placeholder.markdown("🎤 **Listening...**")
            else:
                status_placeholder.markdown("⏸️ **Paused**")
        
        with control_col:
            if st.button("🎤 Start" if not st.session_state.is_listening else "⏹️ Stop"):
                if not st.session_state.is_listening:
                    # Starting recording
                    st.session_state.is_listening = True
                    st.session_state.audio_handler.start_recording()
                    status_placeholder.markdown("🎤 **Listening...**")
                else:
                    # Manual stop
                    st.session_state.is_listening = False
                    audio_data = st.session_state.audio_handler.stop_recording()
                    if audio_data is not None:
                        process_audio_and_respond(audio_data, status_placeholder)
                st.rerun()

        # Chat history
        st.markdown("### Conversation")
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.conversation:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

    # Settings sidebar
    with col_settings:
        st.markdown("### Settings")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", 
                               type="password", 
                               value=config.OPENAI_API_KEY,
                               help="Enter your OpenAI API key")
        if api_key:
            config.OPENAI_API_KEY = api_key
            st.session_state.chat.client.api_key = api_key
        
        # Model selection
        model = st.selectbox("GPT Model", 
                           ["gpt-3.5-turbo", "gpt-4"], 
                           index=0,
                           help="Select the GPT model to use")
        if model != config.DEFAULT_MODEL:
            config.DEFAULT_MODEL = model
        
        # System prompt
        system_prompt = st.text_area("System Prompt", 
                                   value=config.DEFAULT_SYSTEM_PROMPT,
                                   help="Customize the chatbot's behavior")
        if system_prompt != config.DEFAULT_SYSTEM_PROMPT:
            config.DEFAULT_SYSTEM_PROMPT = system_prompt
        
        # Voice settings
        st.markdown("#### Voice Settings")
        new_rate = st.slider("Speech Rate", 
                           min_value=100, 
                           max_value=200, 
                           value=config.TTS_RATE,
                           help="Words per minute")
        if new_rate != config.TTS_RATE:
            config.TTS_RATE = new_rate
            st.session_state.tts.engine.setProperty('rate', new_rate)
        
        new_volume = st.slider("Volume", 
                             min_value=0.0, 
                             max_value=1.0, 
                             value=config.TTS_VOLUME,
                             step=0.1,
                             help="Speech volume")
        if new_volume != config.TTS_VOLUME:
            config.TTS_VOLUME = new_volume
            st.session_state.tts.engine.setProperty('volume', new_volume)
        
        # VAD settings
        st.markdown("#### Voice Detection Settings")
        new_vad_mode = st.selectbox("VAD Aggressiveness",
                                  [1, 2, 3],
                                  index=config.VAD_MODE-1,
                                  help="Higher values = more aggressive filtering")
        if new_vad_mode != config.VAD_MODE:
            config.VAD_MODE = new_vad_mode
            st.session_state.audio_handler.vad.set_mode(new_vad_mode)
        
        new_silence_threshold = st.slider("Silence Threshold",
                                        min_value=0.1,
                                        max_value=2.0,
                                        value=config.SILENCE_THRESHOLD,
                                        step=0.1,
                                        help="Seconds of silence before processing")
        if new_silence_threshold != config.SILENCE_THRESHOLD:
            config.SILENCE_THRESHOLD = new_silence_threshold
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.conversation = []
            st.session_state.chat.conversation_history = []
            st.rerun()
    
    # Check VAD state and process audio if silence is detected
    if st.session_state.is_listening and st.session_state.audio_handler.check_vad_state():
        print("VAD detected silence after speech, processing audio...")
        st.session_state.is_listening = False
        audio_data = st.session_state.audio_handler.stop_recording()
        if audio_data is not None:
            process_audio_and_respond(audio_data, status_placeholder)
        st.rerun()
    elif st.session_state.is_listening:
        time.sleep(0.1)
        st.rerun()

if __name__ == "__main__":
    main()