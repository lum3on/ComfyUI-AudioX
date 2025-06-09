import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import os
import folder_paths
import copy

from .audiox_utils import (
    load_audiox_model_from_file,
    prepare_text_conditioning,
    prepare_video_conditioning,
    prepare_audio_conditioning,
    convert_image_to_video,
    postprocess_audio_output,
    validate_inputs,
    get_generation_parameters,
    get_device,
    clear_model_cache,
    create_empty_conditioning_values,
    create_minimal_conditioning,
    create_video_only_conditioning
)

def safe_import_audiox():
    """Safely import AudioX generation functions with proper error handling."""
    try:
        print("AudioX: Importing generation functions...")
        # Add timeout and better error handling
        import signal
        import sys

        def timeout_handler(signum, frame):
            raise TimeoutError("Import timeout - this may be due to network issues downloading models")

        # Set a timeout for the import (30 seconds)
        if sys.platform != 'win32':  # signal.alarm not available on Windows
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)

        try:
            from stable_audio_tools.inference.generation import generate_diffusion_cond
            print("AudioX: ✅ Generation functions imported successfully!")
            return generate_diffusion_cond
        finally:
            if sys.platform != 'win32':
                signal.alarm(0)  # Cancel the alarm

    except (ImportError, TimeoutError) as e:
        print(f"AudioX: ❌ Failed to import generation functions: {e}")
        raise ImportError(
            f"AudioX dependencies not found or import timed out. "
            f"This may be due to network issues downloading transformer models. "
            f"Error: {e}"
        )

class AudioXModelLoader:
    """Node for loading AudioX models from ComfyUI model directories."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "Select AudioX model file from models/diffusion_models/"
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto"
                }),
            },
            "optional": {
                "config_name": (["auto"] + folder_paths.get_filename_list("configs"), {
                    "default": "auto",
                    "tooltip": "Optional: Select config file from models/configs/ (auto-detect if available)"
                }),
            }
        }

    RETURN_TYPES = ("AUDIOX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AudioX/Models"

    # Add this to help with debugging
    DESCRIPTION = "Load AudioX models from ComfyUI model directories. Fixed version 1.0.1"

    def load_model(self, model_name: str, device: str, config_name: str = "auto"):
        """Load the AudioX model from ComfyUI model directories."""
        try:
            if device == "auto":
                device = get_device()

            print(f"AudioX: Loading model '{model_name}' on device '{device}'")

            # Get full path to model file
            model_path = folder_paths.get_full_path("diffusion_models", model_name)
            if not model_path:
                available_models = folder_paths.get_filename_list("diffusion_models")
                raise FileNotFoundError(
                    f"Model file not found: {model_name}\n"
                    f"Available models: {available_models}\n"
                    f"Please place AudioX model files in: models/diffusion_models/"
                )

            print(f"AudioX: Model path: {model_path}")

            # Check if model file exists and is readable
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file does not exist: {model_path}")

            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"AudioX: Model file size: {file_size:.1f} MB")

            # Handle config file
            config_path = None
            if config_name != "auto":
                config_path = folder_paths.get_full_path("configs", config_name)
                if config_path:
                    print(f"AudioX: Using config: {config_path}")
                else:
                    print(f"AudioX: Config file not found: {config_name}")
            else:
                # Try to auto-detect config file
                config_path = self._find_config_file(model_path, model_name)
                if config_path:
                    print(f"AudioX: Auto-detected config: {config_path}")
                else:
                    print("AudioX: No config file found, using default")

            # Load the model
            print("AudioX: Starting model loading...")
            model, model_config = load_audiox_model_from_file(model_path, config_path, device)
            print("AudioX: Model loaded successfully!")

            return ({
                "model": model,
                "config": model_config,
                "device": device,
                "model_path": model_path,
                "config_path": config_path
            },)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"AudioX Model Loading Error: {error_details}")
            raise RuntimeError(f"Failed to load AudioX model: {str(e)}")

    def _find_config_file(self, model_path: str, model_name: str) -> Optional[str]:
        """Try to find a matching config file for the model."""
        import os

        # Try same directory as model first
        model_dir = os.path.dirname(model_path)
        model_base = os.path.splitext(model_name)[0]

        # Common config file patterns
        config_patterns = [
            f"{model_base}.json",
            f"{model_base}_config.json",
            "config.json",
            "model_config.json",
            "audiox_config.json"  # Added AudioX specific pattern
        ]

        # Check model directory first
        for pattern in config_patterns:
            config_path = os.path.join(model_dir, pattern)
            if os.path.exists(config_path):
                print(f"AudioX: Found config in model directory: {config_path}")
                return config_path

        # Try configs directory
        for pattern in config_patterns:
            config_path = folder_paths.get_full_path("configs", pattern)
            if config_path and os.path.exists(config_path):
                print(f"AudioX: Found config in configs directory: {config_path}")
                return config_path

        print("AudioX: No config file found, will use default")
        return None

class AudioXTextToAudio:
    """Node for text-to-audio generation using AudioX."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "Typing on a keyboard",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "AudioX/Generation"
    
    def generate_audio(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float, 
                      seed: int, duration_seconds: float):
        """Generate audio from text prompt."""
        try:
            generate_diffusion_cond = safe_import_audiox()
            
            # Validate inputs
            validate_inputs(text_prompt=text_prompt)
            
            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]
            
            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)
            
            # Try minimal conditioning first (text-only)
            conditioning = [create_minimal_conditioning(text_prompt, int(duration_seconds), device)]
            
            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )
            
            # Generate audio with fallback to full conditioning if needed
            with torch.no_grad():
                try:
                    output = generate_diffusion_cond(
                        copied_model,
                        conditioning=conditioning,
                        device=device,
                        **gen_params
                    )
                except (ValueError, RuntimeError) as e:
                    if "not found in batch metadata" in str(e) or "size of tensor" in str(e):
                        print(f"AudioX: Minimal conditioning failed ({e}), trying full conditioning...")
                        # Fall back to full conditioning with empty values
                        empty_values = create_empty_conditioning_values(device, int(duration_seconds))
                        conditioning = [{
                            "text_prompt": text_prompt,
                            "video_prompt": empty_values["video_prompt"],  # Empty video tensor
                            "audio_prompt": empty_values["audio_prompt"],  # Empty audio tensor
                            "seconds_start": 0,
                            "seconds_total": int(duration_seconds)
                        }]
                        output = generate_diffusion_cond(
                            copied_model, # Use copied model in fallback too
                            conditioning=conditioning,
                            device=device,
                            **gen_params
                        )
                    else:
                        raise
            
            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)
            
        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {str(e)}")

class AudioXTextToMusic:
    """Node for text-to-music generation using AudioX."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "A music with piano and violin",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_music"
    CATEGORY = "AudioX/Generation"
    
    def generate_music(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float, 
                      seed: int, duration_seconds: float):
        """Generate music from text prompt."""
        try:
            generate_diffusion_cond = safe_import_audiox()
            
            # Validate inputs
            validate_inputs(text_prompt=text_prompt)
            
            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]
            
            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)
            
            # Try minimal conditioning first (text-only)
            conditioning = [create_minimal_conditioning(text_prompt, int(duration_seconds), device)]
            
            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )
            
            # Generate music with fallback to full conditioning if needed
            with torch.no_grad():
                try:
                    output = generate_diffusion_cond(
                        copied_model,
                        conditioning=conditioning,
                        device=device,
                        **gen_params
                    )
                except (ValueError, RuntimeError) as e:
                    if "not found in batch metadata" in str(e) or "size of tensor" in str(e):
                        print(f"AudioX: Minimal conditioning failed ({e}), trying full conditioning...")
                        # Fall back to full conditioning with empty values
                        empty_values = create_empty_conditioning_values(device, int(duration_seconds))
                        conditioning = [{
                            "text_prompt": text_prompt,
                            "video_prompt": empty_values["video_prompt"],  # Empty video tensor
                            "audio_prompt": empty_values["audio_prompt"],  # Empty audio tensor
                            "seconds_start": 0,
                            "seconds_total": int(duration_seconds)
                        }]
                        output = generate_diffusion_cond(
                            copied_model, # Use copied model in fallback too
                            conditioning=conditioning,
                            device=device,
                            **gen_params
                        )
                    else:
                        raise
            
            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)
            
        except Exception as e:
            raise RuntimeError(f"Music generation failed: {str(e)}")

class AudioXVideoToAudio:
    """Node for video-to-audio generation using AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "video": ("IMAGE",),  # ComfyUI video format
                "text_prompt": ("STRING", {
                    "default": "Generate general audio for the video",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio_from_video"
    CATEGORY = "AudioX/Generation"

    def generate_audio_from_video(self, model: Dict, video: torch.Tensor, text_prompt: str,
                                 steps: int, cfg_scale: float, seed: int, duration_seconds: float):
        """Generate audio from video input."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Validate inputs
            validate_inputs(text_prompt=text_prompt, video_tensor=video)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # CRITICAL FIX: Use specialized video-only conditioning to avoid sequence length conflicts
            print(f"AudioX: Creating video-only conditioning for {duration_seconds}s generation")
            conditioning = create_video_only_conditioning(video, text_prompt, duration_seconds, device)

            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate audio
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Video-to-audio generation failed: {str(e)}")

class AudioXVideoToMusic:
    """Node for video-to-music generation using AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "video": ("IMAGE",),  # ComfyUI video format
                "text_prompt": ("STRING", {
                    "default": "Generate music for the video",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_music_from_video"
    CATEGORY = "AudioX/Generation"

    def generate_music_from_video(self, model: Dict, video: torch.Tensor, text_prompt: str,
                                 steps: int, cfg_scale: float, seed: int, duration_seconds: float):
        """Generate music from video input."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Validate inputs
            validate_inputs(text_prompt=text_prompt, video_tensor=video)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # CRITICAL FIX: Use specialized video-only conditioning to avoid sequence length conflicts
            print(f"AudioX: Creating video-only conditioning for music generation ({duration_seconds}s)")
            conditioning = create_video_only_conditioning(video, text_prompt, duration_seconds, device)

            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate music
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Video-to-music generation failed: {str(e)}")

class AudioXMultiModalGeneration:
    """Node for multi-modal audio generation using AudioX."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("AUDIOX_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "Generate audio",
                    "multiline": True
                }),
                "steps": ("INT", {
                    "default": 250,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 7.0,
                    "min": 0.1,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "video": ("IMAGE",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_multimodal_audio"
    CATEGORY = "AudioX/Generation"

    def generate_multimodal_audio(self, model: Dict, text_prompt: str, steps: int, cfg_scale: float,
                                 seed: int, duration_seconds: float, video=None, image=None, audio=None):
        """Generate audio using multiple modalities."""
        try:
            generate_diffusion_cond = safe_import_audiox()

            # Validate inputs
            validate_inputs(text_prompt=text_prompt, video_tensor=video,
                          image_tensor=image, audio_tensor=audio)

            # Get model components
            audiox_model = model["model"]
            # Create a deep copy of the model for this generation pass to prevent state mutation
            copied_model = copy.deepcopy(audiox_model)
            model_config = model["config"]
            device = model["device"]

            # Calculate sample size based on duration
            sample_rate = model_config["sample_rate"]
            sample_size = int(duration_seconds * sample_rate)

            # Prepare conditioning based on available inputs
            conditioning = {
                "text_prompt": text_prompt,
                "seconds_start": 0,
                "seconds_total": int(duration_seconds)
            }

            # Always add empty values for video and audio conditioning to avoid missing key errors
            empty_values = create_empty_conditioning_values(device, int(duration_seconds))
            conditioning["video_prompt"] = empty_values["video_prompt"]  # Will be overridden if video/image provided
            conditioning["audio_prompt"] = empty_values["audio_prompt"]  # Will be overridden if audio provided

            # Handle video input
            if video is not None:
                video_conditioning = prepare_video_conditioning(video, text_prompt, model_config)
                conditioning["video_prompt"] = video_conditioning["video_prompt"]

            # Handle image input (convert to video)
            elif image is not None:
                # Convert image to video
                target_fps = model_config.get("video_fps", 8)
                duration_frames = int(duration_seconds * target_fps)
                video_tensor = convert_image_to_video(image, duration_frames)
                conditioning["video_prompt"] = [video_tensor.unsqueeze(0)]

            # Handle audio input
            if audio is not None:
                conditioning["audio_prompt"] = audio.unsqueeze(0) if audio.dim() == 2 else audio

            # Get generation parameters
            gen_params = get_generation_parameters(
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                sample_size=sample_size
            )

            # Generate audio
            with torch.no_grad():
                output = generate_diffusion_cond(
                    copied_model,
                    conditioning=[conditioning],
                    device=device,
                    **gen_params
                )

            # Postprocess output
            audio_tensor, metadata = postprocess_audio_output(output, sample_rate)

            # Return in ComfyUI AUDIO format with proper sample rate
            audio_output = {"waveform": audio_tensor, "sample_rate": metadata["sample_rate"]}
            return (audio_output,)

        except Exception as e:
            raise RuntimeError(f"Multi-modal audio generation failed: {str(e)}")

class AudioXAudioProcessor:
    """Node for audio processing and utilities."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True}),
                "target_channels": (["mono", "stereo", "keep"], {"default": "keep"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "sample_rate", "info")
    FUNCTION = "process_audio"
    CATEGORY = "AudioX/Utils"

    def process_audio(self, audio: dict, normalize: bool = True, target_channels: str = "keep"):
        """Process audio tensor."""
        try:
            # Extract waveform and sample rate from AUDIO format
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]

            processed_audio = waveform.clone()

            # Handle channel conversion
            if target_channels == "mono" and processed_audio.shape[0] > 1:
                processed_audio = torch.mean(processed_audio, dim=0, keepdim=True)
            elif target_channels == "stereo" and processed_audio.shape[0] == 1:
                processed_audio = processed_audio.repeat(2, 1)

            # Normalize if requested
            if normalize:
                max_val = torch.max(torch.abs(processed_audio))
                if max_val > 0:
                    processed_audio = processed_audio / max_val

            # Clamp to valid range
            processed_audio = torch.clamp(processed_audio, -1.0, 1.0)

            # Generate info string
            channels = processed_audio.shape[0]
            duration = processed_audio.shape[1] / sample_rate
            info = f"Channels: {channels}, Duration: {duration:.2f}s, Sample Rate: {sample_rate}Hz"

            # Return in ComfyUI AUDIO format with separate sample rate output
            audio_output = {"waveform": processed_audio, "sample_rate": sample_rate}
            return (audio_output, sample_rate, info)

        except Exception as e:
            raise RuntimeError(f"Audio processing failed: {str(e)}")

class AudioXVideoMuter:
    """Node for removing audio from video and passing through muted video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),  # ComfyUI video format (batch of images)
            },
            "optional": {
                "preserve_metadata": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("muted_video",)
    FUNCTION = "mute_video"
    CATEGORY = "AudioX/Utils"

    def mute_video(self, video: torch.Tensor, preserve_metadata: bool = True):
        """
        Remove audio from video by passing through the video frames without any audio track.
        This is useful for workflows where you want to replace original audio with generated audio.

        Args:
            video: Video tensor in ComfyUI format (batch of images)
            preserve_metadata: Whether to preserve video metadata (frame rate, etc.)

        Returns:
            Tuple containing the muted video (same as input video frames)
        """
        try:
            # Validate input
            if not isinstance(video, torch.Tensor):
                raise ValueError("Video input must be a torch.Tensor")

            if video.dim() != 4:
                raise ValueError(f"Video tensor must be 4-dimensional (batch, height, width, channels), got {video.dim()}D")

            # Simply pass through the video frames - the "muting" happens at the output level
            # where no audio track is associated with these frames
            muted_video = video.clone()

            # Log the operation
            batch_size, height, width, channels = video.shape
            print(f"AudioX Video Muter: Processed {batch_size} frames ({height}x{width}x{channels})")
            print(f"AudioX Video Muter: Video muted - audio track removed")

            return (muted_video,)

        except Exception as e:
            raise RuntimeError(f"Video muting failed: {str(e)}")

class AudioXVideoAudioCombiner:
    """Node for combining muted video with generated audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),  # ComfyUI video format (batch of images)
                "audio": ("AUDIO",),  # ComfyUI audio format
            },
            "optional": {
                "sync_duration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically sync audio duration to video duration"
                }),
                "loop_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop audio if shorter than video"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("video", "synced_audio", "info")
    FUNCTION = "combine_video_audio"
    CATEGORY = "AudioX/Utils"

    def combine_video_audio(self, video: torch.Tensor, audio: dict,
                           sync_duration: bool = True, loop_audio: bool = False):
        """
        Combine video frames with audio, ensuring proper synchronization.

        Args:
            video: Video tensor in ComfyUI format
            audio: Audio dictionary with 'waveform' and 'sample_rate'
            sync_duration: Whether to sync audio duration to video
            loop_audio: Whether to loop audio if it's shorter than video

        Returns:
            Tuple containing (video, synced_audio, info_string)
        """
        try:
            # Validate inputs
            if not isinstance(video, torch.Tensor):
                raise ValueError("Video input must be a torch.Tensor")

            if not isinstance(audio, dict) or 'waveform' not in audio or 'sample_rate' not in audio:
                raise ValueError("Audio input must be a dictionary with 'waveform' and 'sample_rate' keys")

            # Get video and audio properties
            num_frames = video.shape[0]
            audio_waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # Assume standard video frame rate (can be made configurable)
            video_fps = 8.0  # Default FPS for ComfyUI video workflows
            video_duration = num_frames / video_fps
            audio_duration = audio_waveform.shape[-1] / sample_rate

            print(f"AudioX Combiner: Video frames: {num_frames}, duration: {video_duration:.2f}s")
            print(f"AudioX Combiner: Audio duration: {audio_duration:.2f}s")

            synced_audio = audio.copy()

            if sync_duration:
                target_samples = int(video_duration * sample_rate)
                current_samples = audio_waveform.shape[-1]

                if current_samples < target_samples:
                    if loop_audio:
                        # Loop audio to match video duration
                        repeat_count = (target_samples + current_samples - 1) // current_samples
                        looped_waveform = audio_waveform.repeat(1, 1, repeat_count)
                        synced_audio['waveform'] = looped_waveform[:, :, :target_samples]
                        print(f"AudioX Combiner: Audio looped {repeat_count} times to match video duration")
                    else:
                        # Pad audio with silence
                        padding_samples = target_samples - current_samples
                        padding = torch.zeros(audio_waveform.shape[0], audio_waveform.shape[1],
                                            padding_samples, device=audio_waveform.device,
                                            dtype=audio_waveform.dtype)
                        synced_audio['waveform'] = torch.cat([audio_waveform, padding], dim=-1)
                        print(f"AudioX Combiner: Audio padded with {padding_samples} samples of silence")

                elif current_samples > target_samples:
                    # Trim audio to match video duration
                    synced_audio['waveform'] = audio_waveform[:, :, :target_samples]
                    print(f"AudioX Combiner: Audio trimmed to match video duration")

            # Generate info string
            final_audio_duration = synced_audio['waveform'].shape[-1] / sample_rate
            info = (f"Video: {num_frames} frames ({video_duration:.2f}s), "
                   f"Audio: {final_audio_duration:.2f}s, "
                   f"Sample Rate: {sample_rate}Hz")

            return (video, synced_audio, info)

        except Exception as e:
            raise RuntimeError(f"Video-audio combination failed: {str(e)}")
