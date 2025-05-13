import numpy as np
import sounddevice as sd
import webrtcvad
from queue import Queue, Empty
from threading import Thread, Event
import time
import pyttsx3
from openai import OpenAI
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json
import config

class AudioHandler:
    def __init__(self):
        self.vad = webrtcvad.Vad(config.VAD_MODE)
        self.audio_queue = Queue()
        self.is_recording = False
        self.stop_event = Event()
        self.silence_start = None
        self.speech_buffer = []
        self.audio_processor = None
        self.audio_data = []
        self.device = None
        self.has_speech = False
        self.auto_stopped = False
        self.speech_detected = False
        self.silence_detected = False
        self.last_vad_check = 0  # Track last VAD check time
        self.continuous_mode = False
        self.init_audio_device()

    def init_audio_device(self):
        """Initialize audio device with fallback options"""
        try:
            # List all available devices
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    print(f"[{i}] {dev['name']} (inputs: {dev['max_input_channels']})")
                    config.FALLBACK_INPUT_DEVICES.append(i)
            
            # Try to use default input device first
            try:
                default_device = sd.query_devices(kind='input')
                self.device = default_device['index']
                print(f"\nUsing default input device: {default_device['name']}")
            except:
                if config.FALLBACK_INPUT_DEVICES:
                    self.device = config.FALLBACK_INPUT_DEVICES[0]
                    device_info = sd.query_devices(self.device)
                    print(f"\nUsing fallback input device: {device_info['name']}")
                else:
                    raise RuntimeError("No input devices available")
            
            # Update channel count if needed
            device_info = sd.query_devices(self.device)
            if device_info['max_input_channels'] < config.CHANNELS:
                config.CHANNELS = device_info['max_input_channels']
                
        except Exception as e:
            print(f"Error initializing audio device: {e}")
            raise

    def try_next_device(self):
        """Try to switch to the next available input device"""
        if not config.FALLBACK_INPUT_DEVICES:
            return False
            
        current_index = config.FALLBACK_INPUT_DEVICES.index(self.device)
        if current_index + 1 < len(config.FALLBACK_INPUT_DEVICES):
            self.device = config.FALLBACK_INPUT_DEVICES[current_index + 1]
            device_info = sd.query_devices(self.device)
            print(f"\nSwitching to next input device: {device_info['name']}")
            return True
        return False

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Error in audio callback: {status}")
        if self.is_recording:
            try:
                self.audio_queue.put(indata.copy())
                self.audio_data.extend(indata.flatten())
                # Keep only last 5 seconds for visualization
                max_samples = int(config.SAMPLE_RATE * 5)
                if len(self.audio_data) > max_samples:
                    self.audio_data = self.audio_data[-max_samples:]
            except Exception as e:
                print(f"Error in audio callback: {e}")

    def get_audio_data(self):
        """Get current audio data for visualization"""
        return np.array(self.audio_data)

    def check_vad_state(self):
        """
        Check if we should process audio based on VAD state
        """
        current_time = time.time()
        
        # Limit VAD checks to every 100ms
        if current_time - self.last_vad_check < 0.1:
            return False
            
        self.last_vad_check = current_time
        
        if not self.is_recording or not self.has_speech:
            return False
            
        if self.silence_start is not None:
            silence_duration = current_time - self.silence_start
            min_samples = int(config.SAMPLE_RATE * config.MIN_SPEECH_DURATION)
            total_samples = sum(len(chunk) for chunk in self.speech_buffer)
            
            if silence_duration >= config.MAX_SILENCE and total_samples >= min_samples:
                print(f"VAD State: Speech detected and silence threshold ({silence_duration:.2f}s) reached")
                return True
                
        return False

    def process_audio_chunk(self, audio_chunk):
        """Process audio chunk and detect speech/silence"""
        # Convert to mono if stereo
        if len(audio_chunk.shape) > 1 and audio_chunk.shape[1] > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)

        # Convert float32 audio to int16 for VAD
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Process in 30ms frames (optimal for VAD)
        frame_length = int(config.SAMPLE_RATE * 0.03)  # 30ms
        
        # Count speech frames
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_int16), frame_length):
            frame = audio_int16[i:i + frame_length]
            if len(frame) == frame_length:  # Only process complete frames
                total_frames += 1
                try:
                    frame_bytes = frame.tobytes()
                    if self.vad.is_speech(frame_bytes, config.SAMPLE_RATE):
                        speech_frames += 1
                except Exception as e:
                    print(f"VAD error: {e}")
                    continue
        
        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
        is_speech = speech_ratio > 0.1  # More than 10% of frames contain speech
        
        if is_speech:
            if self.silence_start is not None:
                print("Speech detected, resetting silence timer")
            self.silence_start = None
            self.speech_buffer.append(audio_chunk)
            self.has_speech = True
            return False
        elif self.has_speech:
            if self.silence_start is None:
                print("Starting silence timer")
                self.silence_start = time.time()
            return False
        return False

    def start_recording(self):
        """Start recording audio with automatic silence detection"""
        self.reset()  # Ensure clean state
        self.is_recording = True
        self.stop_event.clear()
        
        def audio_processing_thread():
            try:
                stream_settings = {
                    'device': self.device,
                    'channels': config.CHANNELS,
                    'samplerate': config.SAMPLE_RATE,
                    'callback': self.audio_callback,
                    'blocksize': config.CHUNK_SIZE,
                    'dtype': 'float32',
                }
                
                with sd.InputStream(**stream_settings) as self.stream:
                    print("Audio stream started successfully")
                    while not self.stop_event.is_set():
                        try:
                            audio_chunk = self.audio_queue.get(timeout=1.0)
                            self.process_audio_chunk(audio_chunk)
                        except Empty:
                            continue
                        except Exception as e:
                            print(f"Error processing audio chunk: {e}")
                            
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                self.stop_event.set()
        
        self.audio_processor = Thread(target=audio_processing_thread)
        self.audio_processor.start()

    def stop_recording(self):
        """Stop recording and return processed audio if available"""
        was_recording = self.is_recording
        self.is_recording = False
        self.stop_event.set()
        
        if self.audio_processor:
            self.audio_processor.join()
            self.audio_processor = None
            
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Only process audio if we actually had speech
        if was_recording and self.has_speech and self.speech_buffer:
            try:
                audio_data = np.concatenate(self.speech_buffer)
                # Ensure mono audio
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    audio_data = np.mean(audio_data, axis=1)
                return audio_data
            except Exception as e:
                print(f"Error concatenating audio: {e}")
                return None
        return None

    def reset(self):
        """Reset all audio handler states for next interaction"""
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        if self.audio_processor:
            self.stop_event.set()
            self.audio_processor.join()
            self.audio_processor = None
        
        self.audio_queue = Queue()
        self.speech_buffer = []
        self.audio_data = []
        self.silence_start = None
        self.has_speech = False
        self.auto_stopped = False
        self.stop_event.clear()
        self.is_recording = False

class SpeechToText:
    def __init__(self):
        # Initialize Whisper model
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.torch_dtype = torch.float32

            model_id = "openai/whisper-tiny.en"
            
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            print("Successfully initialized Whisper tiny model")
        except Exception as e:
            print(f"Error initializing Whisper model: {e}")
            raise

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""
        
        try:
            # Ensure audio is float32 and in the correct range [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Convert to mono explicitly if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Create a dict with the audio data and sampling rate
            audio_dict = {
                "array": audio,
                "sampling_rate": config.SAMPLE_RATE
            }
            
            # Process audio with Whisper
            result = self.pipe(audio_dict)
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""

class TextToSpeech:
    def __init__(self):
        self._engine = None
        
    def _create_engine(self):
        """Create a new TTS engine instance"""
        if self._engine is not None:
            try:
                self._engine.stop()
                self._engine = None
            except:
                pass
        engine = pyttsx3.init()
        engine.setProperty('rate', config.TTS_RATE)
        engine.setProperty('volume', config.TTS_VOLUME)
        return engine
    
    def speak(self, text):
        """Speak text with proper engine lifecycle management"""
        try:
            # Create a fresh engine instance for each speech
            engine = self._create_engine()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
            try:
                # One retry with a fresh engine
                engine = self._create_engine()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS retry failed: {e}")
        finally:
            # Always cleanup
            if engine:
                try:
                    engine.stop()
                except:
                    pass
                engine = None

class ChatGPT:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.conversation_history = []
        
    def get_response(self, user_input):
        messages = [
            {"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT},
            *self.conversation_history,
            {"role": "user", "content": user_input}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=messages,
                max_tokens=150
            )
            
            assistant_message = response.choices[0].message.content
            self.conversation_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_message}
            ])
            
            return assistant_message
        except Exception as e:
            print(f"Error in GPT response: {e}")
            return "I apologize, but I encountered an error. Could you please try again?"