# Home Assistant Add-on: sherpa-onnx-tts-stt

# Standard library imports
import logging
import os
import subprocess  # For running shell commands to download models
import asyncio
import argparse
from functools import partial
import numpy as np
import math

# Third-party library imports
from wyoming.asr import (
    Transcribe,
    Transcript,
)

from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
#from wyoming.config import PipelineConfig, load_config
from wyoming.event import Event
from wyoming.info import  AsrModel, AsrProgram, TtsVoice, TtsProgram, Describe, Info, Attribution
from wyoming.server import AsyncEventHandler, AsyncServer, AsyncTcpServer
from wyoming.tts import Synthesize

import sherpa_onnx

_LOGGER = logging.getLogger("sherpa_onnx_addon")

# --- Configuration Defaults ---
DEFAULT_LANGUAGE = "zh-CN"  # Default to Chinese
DEFAULT_SPEED = 1.0

def _download_stt_model(model_url, model_path):
        """Downloads and extracts the STT model."""
        if not os.path.exists(model_path):
            _LOGGER.info(f"Downloading STT model: {model_url}")
            os.makedirs(model_path, exist_ok=True)

            # Use curl (or wget) for download and extraction (more robust than Python libraries for large files)
            try:
             subprocess.check_call(
                   ["curl", "-L", model_url, "-o", "stt_model.tar.bz2"]
               )

             subprocess.check_call(["tar", "-xvf", "stt_model.tar.bz2","-C", model_path])
             os.remove("stt_model.tar.bz2") # Clean up

            except subprocess.CalledProcessError as e:
                 _LOGGER.error(f"Error downloading or extracting STT model: {e}")
                 raise  #  Re-raise to stop add-on startup on failure
        else:
         _LOGGER.info("STT model already exists.")
def _download_tts_model(model_url,model_dir):
        """Downloads and extracts the STT model."""
        if not os.path.exists(model_dir):
            _LOGGER.info(f"Downloading TTS model: {model_url}")
            os.makedirs(model_dir, exist_ok=True)

            # Use curl (or wget) for download and extraction (more robust than Python libraries for large files)
            try:
             subprocess.check_call(
                   ["curl", "-L", model_url, "-o", "tts_model.tar.bz2"]
               )

             subprocess.check_call(["tar", "-xvf", "tts_model.tar.bz2","-C", model_dir])
             os.remove("tts_model.tar.bz2") # Clean up

            except subprocess.CalledProcessError as e:
                 _LOGGER.error(f"Error downloading or extracting TTS model: {e}")
                 raise  #  Re-raise to stop add-on startup on failure
        else:
         _LOGGER.info("TTS model already exists.")



def _initialize_models():
        """Downloads (if necessary) and initializes the STT and TTS models."""

        # --- STT Model ---
        stt_model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2"
        stt_model_dir =  "/stt_model"
        _download_stt_model(stt_model_url, stt_model_dir)


        # --- TTS Model ---
        tts_model_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2"
        tts_vocoder = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx"
        tts_model_dir =  "/tts_model"
        _download_tts_model(tts_model_url,tts_model_dir)
        _download_tts_model(tts_vocoder,tts_model_dir)

class SherpaOnnxEventHandler(AsyncEventHandler):
    """Event handler for sherpa-onnx TTS and STT."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args,  # For language/speed settings
        tts_model,
        stt_model,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_ = wyoming_info

#        self.config = load_config()
#        self.pipeline_config: PipelineConfig = self.config.pipelines[
#            self.cli_args.pipeline
#        ]

        _LOGGER.info(f"CLI Args: {self.cli_args}")
        self.language = self.cli_args.language or DEFAULT_LANGUAGE
        self.speed = self.cli_args.speed or DEFAULT_SPEED

        self.tts_model = tts_model

        self.stt_model = stt_model

#        self._initialize_models()  # Download and initialize models
        self.audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.audio = b""


    async def initialize(self) -> None:
           """ Async initialization (if needed, after models are loaded) """
           pass


    async def handle_event(self, event: Event) -> bool:
        """Handles a single event."""
        _LOGGER.debug(f"Received event: {event}") # important for debugging
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_.event())
            _LOGGER.debug(f"Sent info:{self.wyoming_info_.event()}")
            return True  # Stop other handlers


        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            audio = self.tts_model.generate(
                 text=synthesize.text,
                 sid=0, # Speaker ID, adjust if using a multi-speaker model
                 speed=self.speed

            )
            _LOGGER.info(f"Synthesizing: {synthesize.text}")
            if isinstance(audio.samples, list):
                audio_samples = np.array(audio.samples, dtype=np.float32)
            elif isinstance(audio.samples, np.ndarray):
                audio_samples = audio.samples.astype(np.float32) # Ensure float32
            else:
                raise TypeError("Unexpected type for audio.samples: {}".format(type(audio.samples)))

            # Scale to int16 and convert to bytes
            audio_samples = (audio_samples * 32767).astype(np.int16)
            audio_bytes = audio_samples.tobytes()
            # Send AudioStart
            await self.write_event(
                AudioStart(
                    rate=audio.sample_rate,
                    width=2,
                    channels=1
                ).event()
            )
            _LOGGER.debug("Sent Audio Start")
            # Send TTS Chunk (raw audio)
            bytes_per_chunk = 1024;
            num_chunks = int(math.ceil(len(audio_bytes) / bytes_per_chunk))

            # Split into chunks
            for i in range(num_chunks):
                offset = i * bytes_per_chunk
                chunk = audio_bytes[offset : offset + bytes_per_chunk]
                await self.write_event(
                    AudioChunk(
                        audio=chunk,
                        rate=audio.sample_rate,
                        width=2,
                        channels=1,
                    ).event(),
                )
            _LOGGER.debug("Sent TTS Chunk")
            # Send Audio Stop
            await self.write_event(AudioStop().event())
            _LOGGER.debug("Sent Audio Stop")
            

            return True # We handled the event

        elif Transcribe.is_type(event.type):
                # STT (Speech-to-Text)
                transcribe: Transcribe = Transcribe.from_event(event)
                _LOGGER.debug(f"Received Transcribe request: {transcribe}")
                return True




        elif AudioStart.is_type(event.type):
                # Handle AudioStart (if needed, e.g., for resetting state)
                _LOGGER.debug(f"Received audio start event")
                audio_start = AudioStart.from_event(event)
                self.audio_recv_rate=audio_start.rate
                self.stream=self.stt_model.create_stream()
                self.audio = b""
                return True



        elif AudioChunk.is_type(event.type):
                 # Handle AudioChunk
                audio_chunk = AudioChunk.from_event(event)
                #_LOGGER.debug(f"Received audio chunk: {len(audio_chunk.audio)} bytes")

                _LOGGER.debug("Processing audio chunk...")

                # Convert to expected format.
                chunk = self.audio_converter.convert(audio_chunk)
                self.audio += chunk.audio                
                return True
        elif AudioStop.is_type(event.type):
                audio_stop = AudioStop.from_event(event)
                _LOGGER.debug(f"Recevie audio stop:{audio_stop}")
                if self.audio:
                    audio_array = np.frombuffer(self.audio, dtype=np.int16).astype(np.float32) / 32768.0
                    self.stream.accept_waveform(self.audio_recv_rate, audio_array)
                    #_LOGGER.debug(f'{audio_array}')
                    #_LOGGER.debug(f'{self.audio}')
                    self.stt_model.decode_stream(self.stream)
                    result=self.stream.result
                    _LOGGER.debug(f'{result}')


                if result and result.text:
                    await self.write_event(Transcript(text=result.text).event())
                    _LOGGER.info(f"Final transcript on stop: {result.text}")

                return True
        return False # Unhandled events



async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        default="default",
        help="Name of Wyoming pipeline to use",
    )
    # Add custom arguments for language and speed
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Language for TTS (default: zh)")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Speech speed (default: 1.0)")

    # Wyoming Server arguments
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10400)
    parser.add_argument("--uri", help="URI of server (e.g., tcp://0.0.0.0:10500)")

     # Parse arguments
    cli_args = parser.parse_args()



    # Create the Wyoming info object.  This describes what
    #  the add-on supports.
    wyoming_info = Info(
        asr=[ 
            AsrProgram(
            name="sherpa-onnx-streaming-paraformer",
            description="k2-fsa Chinese/English ASR system using Paraformer models from sherpa-onnx.",
            attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
            installed=True,
            version="0.0.1",
            models=[
            AsrModel(
                name="sherpa-onnx-paraformer-zh-2023-03-28",
                description="Paraformer Chinese ASR model",
                languages=["zh-CN"],
                attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed= True,  #  model is now bundled
                version="0.1.0",
        )
        ]
    )
    ],

        tts=[
            TtsProgram(
                name="sherpa-onnx-offline-tts",
                description="Chinese TTS based on sherpa-onnx and the matcha-icefall-zh-baker model.",
                attribution=Attribution(
                    name="k2-fsa",
                    url="https://github.com/k2-fsa/sherpa-onnx",
                ),
                installed= True,
                version="0.1.0",
                voices=[
                    TtsVoice(
                        name="match-icefall-zh-baker",
                        description="matcha-icefall Chinese TTS model",
                        languages=["zh-CN"],
                        attribution=Attribution(
                            name="k2-fsa",
                            url="https://github.com/k2-fsa/sherpa-onnx",
                            ),
                        installed= True,  #  model is now bundled
                        version="0.1.0",
                        )
                    ],
                )

            ]
    )

     # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.info("Starting sherpa-onnx add-on...")

    stt_model_dir="/stt_model"
    tts_model_dir="/tts_model"
    # STT Initialization (adjust paths as needed for extracted model)
    try:
                stt_model = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=os.path.join(stt_model_dir, "model.onnx"),
                tokens=os.path.join(stt_model_dir, "tokens.txt"),
                decoding_method='greedy_search',
                num_threads=4,   # Adjust based on your hardware
                sample_rate=16000,
                feature_dim=80,
                debug=False,
            )
    except Exception as e:  # More specific exception handling is better
            _LOGGER.exception("Failed to initialize STT model:")
            _LOGGER.error(e)
            raise

    # TTS Initialization
    try:
                tts_model = sherpa_onnx.OfflineTts(
                sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=os.path.join(tts_model_dir,"model-steps-3.onnx"),
                vocoder=os.path.join(tts_model_dir,"hifigan_v2.onnx"),
                lexicon=os.path.join(tts_model_dir,"lexicon.txt"),
                tokens=os.path.join(tts_model_dir,"tokens.txt"),
                data_dir=os.path.join(tts_model_dir,""), # Add your espeak-ng-data path if necessary
                dict_dir=os.path.join(tts_model_dir,"dict")
                ),
                provider="cpu",    # or "cuda" if you have a GPU
                num_threads=6,     # Adjust as needed
                debug=False,       # Set to True for debugging output
                ),

                rule_fsts=f"{tts_model_dir}/phone.fst,{tts_model_dir}/date.fst,{tts_model_dir}/number.fst",  # Example rule FSTs, adjust path if needed
                max_num_sentences=1,
                )
                )

    except Exception as e:
            _LOGGER.exception("Failed to initialize TTS model:")
            raise

    # Create the server and handler, using our custom handler.
    if cli_args.uri is not None:
        # Connect to remote server
        server = AsyncServer.from_url(cli_args.url)
        reader, writer = await AsyncTcpClient.connect(server)
        await AsyncEventHandler.run_handler(
            SherpaOnnxEventHandler(wyoming_info,  cli_args, reader, writer)
        )
    else:
        # Run local server
        server = AsyncTcpServer(cli_args.host, cli_args.port)
        _LOGGER.info(f"Starting server...{cli_args.host}, {cli_args.port}")
        await server.run(
           partial(
               SherpaOnnxEventHandler,
               wyoming_info,
               cli_args,
               tts_model,
               stt_model,
               )
           )
    _LOGGER.info("Stopped")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

