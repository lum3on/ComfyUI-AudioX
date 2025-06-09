#!/usr/bin/env python3
"""
Installation script for AudioX dependencies.
Run this script to ensure all required dependencies are installed.
"""

import sys
import subprocess
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True
    except ImportError:
        pass
    return False

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_name
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ {package_name} installed successfully")
            return True
        else:
            print(f"✗ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error installing {package_name}: {e}")
        return False

def main():
    """Main installation function."""
    print("AudioX Dependency Installer")
    print("=" * 40)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print()
    
    # List of required packages
    required_packages = [
        ("torch", "torch"),
        ("torchaudio", "torchaudio"),
        ("transformers", "transformers"),
        ("huggingface_hub", "huggingface_hub"),
        ("einops", "einops"),
        ("einops-exts", "einops_exts"),
        ("numpy", "numpy"),
        ("safetensors", "safetensors"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("descript-audio-codec", "dac"),
        ("Pillow", "PIL"),
        ("ema_pytorch", "ema_pytorch"),
        ("x-transformers", "x_transformers"),
        ("alias-free-torch", "alias_free_torch"),
        ("vector-quantize-pytorch", "vector_quantize_pytorch"),
        ("local-attention", "local_attention"),
        ("k-diffusion", "k_diffusion"),
        ("aeiou", "aeiou"),
        ("auraloss", "auraloss"),
        ("encodec", "encodec"),
        ("laion-clap", "laion_clap"),
        ("prefigure", "prefigure"),
        ("v-diffusion-pytorch", "diffusion"),
    ]
    
    missing_packages = []
    
    # Check which packages are missing
    print("Checking installed packages...")
    for package_name, import_name in required_packages:
        if check_package(package_name, import_name):
            print(f"✓ {package_name}")
        else:
            print(f"✗ {package_name} (missing)")
            missing_packages.append(package_name)
    
    print()
    
    if not missing_packages:
        print("✓ All required packages are already installed!")
        return True
    
    # Install missing packages
    print(f"Installing {len(missing_packages)} missing packages...")
    print()
    
    success_count = 0
    for package_name in missing_packages:
        if install_package(package_name):
            success_count += 1
    
    print()
    print("=" * 40)
    print(f"Installation complete: {success_count}/{len(missing_packages)} packages installed successfully")
    
    if success_count == len(missing_packages):
        print("✓ All dependencies installed successfully!")
        
        # Test critical imports
        print("\nTesting critical imports...")
        try:
            from x_transformers import ContinuousTransformerWrapper, Encoder
            print("✓ x_transformers import successful")
        except ImportError as e:
            print(f"✗ x_transformers import failed: {e}")
            return False
            
        try:
            import einops_exts
            print("✓ einops_exts import successful")
        except ImportError as e:
            print(f"✗ einops_exts import failed: {e}")
            return False
            
        try:
            import dac
            print("✓ dac import successful")
        except ImportError as e:
            print(f"✗ dac import failed: {e}")
            return False

        try:
            import alias_free_torch
            print("✓ alias_free_torch import successful")
        except ImportError as e:
            print(f"✗ alias_free_torch import failed: {e}")
            return False

        try:
            import vector_quantize_pytorch
            print("✓ vector_quantize_pytorch import successful")
        except ImportError as e:
            print(f"✗ vector_quantize_pytorch import failed: {e}")
            return False

        print("\n✓ All critical imports working!")
        return True
    else:
        print("✗ Some packages failed to install. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
