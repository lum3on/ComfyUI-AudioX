import os
import sys
import folder_paths as comfy_paths

# Version identifier to force reload
__version__ = "1.0.9"

# Add the audiox directory to Python path
AUDIOX_ROOT = os.path.join(os.path.dirname(__file__), "audiox")
sys.path.insert(0, AUDIOX_ROOT)

# Debug environment information
print(f"AudioX: Python executable: {sys.executable}")
print(f"AudioX: Python version: {sys.version.split()[0]}")
print(f"AudioX: AudioX root: {AUDIOX_ROOT}")
print(f"AudioX: Current working directory: {os.getcwd()}")

# Ensure critical paths are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"AudioX: Added to path: {current_dir}")

# Ensure all required dependencies are available
def ensure_dependencies():
    """Ensure all AudioX dependencies are available."""
    # Complete list of required dependencies
    required_deps = [
        "descript-audio-codec",
        "einops-exts",
        "x-transformers",
        "alias-free-torch",
        "vector-quantize-pytorch",
        "local-attention",
        "k-diffusion",
        "aeiou",
        "auraloss",
        "encodec",
        "laion-clap",
        "prefigure",
        "v-diffusion-pytorch"
    ]

    missing_deps = []

    # Check all dependencies
    dependency_checks = [
        ("dac", "descript-audio-codec"),
        ("einops_exts", "einops-exts"),
        ("x_transformers", "x-transformers"),
        ("alias_free_torch", "alias-free-torch"),
        ("vector_quantize_pytorch", "vector-quantize-pytorch"),
        ("local_attention", "local-attention"),
        ("k_diffusion", "k-diffusion"),
        ("aeiou", "aeiou"),
        ("auraloss", "auraloss"),
        ("encodec", "encodec"),
        ("laion_clap", "laion-clap"),
        ("prefigure", "prefigure"),
        ("diffusion", "v-diffusion-pytorch")
    ]

    for import_name, package_name in dependency_checks:
        try:
            __import__(import_name)
            print(f"AudioX: {package_name} available")
        except ImportError as e:
            print(f"AudioX: {package_name} not available: {e}")
            missing_deps.append(package_name)

    # Install missing dependencies immediately
    if missing_deps:
        print(f"AudioX: Installing {len(missing_deps)} missing dependencies...")
        print(f"AudioX: Using Python: {sys.executable}")

        try:
            import subprocess

            # Install all missing dependencies in one command for efficiency
            install_cmd = [sys.executable, "-m", "pip", "install"] + missing_deps
            print(f"AudioX: Running: {' '.join(install_cmd)}")

            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("AudioX: All dependencies installed successfully!")
                print("AudioX: Reloading modules...")

                # Clear import cache to force reload
                import importlib
                modules_to_reload = [
                    'dac', 'einops_exts', 'x_transformers', 'alias_free_torch',
                    'vector_quantize_pytorch', 'local_attention', 'k_diffusion',
                    'aeiou', 'auraloss', 'encodec', 'laion_clap', 'prefigure',
                    'diffusion'
                ]

                for module_name in modules_to_reload:
                    if module_name in sys.modules:
                        importlib.reload(sys.modules[module_name])

                # Re-check critical dependencies
                missing_deps.clear()
                for import_name, package_name in dependency_checks:
                    try:
                        __import__(import_name)
                        print(f"AudioX: ✓ {package_name} now available")
                    except ImportError:
                        missing_deps.append(package_name)
                        print(f"AudioX: ✗ {package_name} still missing")

            else:
                print(f"AudioX: Installation failed: {result.stderr}")
                print("AudioX: Trying individual installations...")

                # Try installing each dependency individually
                for dep in missing_deps[:]:
                    try:
                        result = subprocess.run([
                            sys.executable, "-m", "pip", "install", dep
                        ], capture_output=True, text=True, timeout=120)

                        if result.returncode == 0:
                            print(f"AudioX: ✓ {dep} installed")
                            missing_deps.remove(dep)
                        else:
                            print(f"AudioX: ✗ Failed to install {dep}")
                    except Exception as e:
                        print(f"AudioX: Error installing {dep}: {e}")

        except Exception as install_error:
            print(f"AudioX: Installation error: {install_error}")

    if missing_deps:
        print(f"AudioX: ⚠️  Still missing: {missing_deps}")
        print("AudioX: Some features may not work correctly")
    else:
        print("AudioX: ✅ All dependencies available!")

    return len(missing_deps) == 0

# EMERGENCY: Force install critical dependencies immediately
print("AudioX: EMERGENCY DEPENDENCY CHECK...")
try:
    import vector_quantize_pytorch
    print("AudioX: vector_quantize_pytorch already available")
except ImportError:
    print("AudioX: INSTALLING vector_quantize_pytorch NOW...")
    import subprocess
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "vector-quantize-pytorch"
        ], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("AudioX: ✅ vector_quantize_pytorch installed successfully!")
            # Force reload the module
            try:
                import importlib
                import vector_quantize_pytorch
                importlib.reload(vector_quantize_pytorch)
            except:
                pass
        else:
            print(f"AudioX: ❌ Failed to install vector_quantize_pytorch: {result.stderr}")
    except Exception as e:
        print(f"AudioX: ❌ Installation error: {e}")

# Ensure dependencies before importing nodes
deps_ok = ensure_dependencies()

# Import our nodes with error handling
try:
    print("AudioX: Importing nodes...")
    from .nodes import (
        AudioXModelLoader,
        AudioXTextToAudio,
        AudioXTextToMusic,
        AudioXVideoToAudio,
        AudioXVideoToMusic,
        AudioXMultiModalGeneration,
        AudioXAudioProcessor,
        AudioXVideoMuter,
        AudioXVideoAudioCombiner
    )
    print("AudioX: ✅ Nodes imported successfully!")
except Exception as e:
    print(f"AudioX: ❌ Failed to import nodes: {e}")
    print("AudioX: This might be due to missing dependencies or network issues.")
    print("AudioX: Creating placeholder nodes...")

    # Create placeholder nodes that will show error messages
    class PlaceholderNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"error_info": ("STRING", {"default": f"AudioX import failed: {str(e)}"})}}

        RETURN_TYPES = ("STRING",)
        FUNCTION = "show_error"
        CATEGORY = "AudioX/Error"

        def show_error(self, error_info):
            raise RuntimeError(f"AudioX nodes are not available: {error_info}")

    # Use placeholder for all nodes
    AudioXModelLoader = PlaceholderNode
    AudioXTextToAudio = PlaceholderNode
    AudioXTextToMusic = PlaceholderNode
    AudioXVideoToAudio = PlaceholderNode
    AudioXVideoToMusic = PlaceholderNode
    AudioXMultiModalGeneration = PlaceholderNode
    AudioXAudioProcessor = PlaceholderNode
    AudioXVideoMuter = PlaceholderNode
    AudioXVideoAudioCombiner = PlaceholderNode

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AudioXModelLoader": AudioXModelLoader,
    "AudioXTextToAudio": AudioXTextToAudio,
    "AudioXTextToMusic": AudioXTextToMusic,
    "AudioXVideoToAudio": AudioXVideoToAudio,
    "AudioXVideoToMusic": AudioXVideoToMusic,
    "AudioXMultiModalGeneration": AudioXMultiModalGeneration,
    "AudioXAudioProcessor": AudioXAudioProcessor,
    "AudioXVideoMuter": AudioXVideoMuter,
    "AudioXVideoAudioCombiner": AudioXVideoAudioCombiner,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioXModelLoader": "AudioX Model Loader",
    "AudioXTextToAudio": "AudioX Text to Audio",
    "AudioXTextToMusic": "AudioX Text to Music",
    "AudioXVideoToAudio": "AudioX Video to Audio",
    "AudioXVideoToMusic": "AudioX Video to Music",
    "AudioXMultiModalGeneration": "AudioX Multi-Modal Generation",
    "AudioXAudioProcessor": "AudioX Audio Processor",
    "AudioXVideoMuter": "AudioX Video Muter",
    "AudioXVideoAudioCombiner": "AudioX Video Audio Combiner",
}

# Web directory for any web components
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
