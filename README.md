# ComfyUI-AudioX

#THIS REPO IS WIP
A powerful audio generation extension for ComfyUI that integrates AudioX models for high-quality audio synthesis from text and video inputs.

## 🎵 Features

- **Text to Audio**: Generate high-quality audio from text descriptions
- **Text to Music**: Create musical compositions from text prompts
- **Video to Audio**: Extract and generate audio from video content
- **Video Processing**: Mute videos and combine with generated audio

## 🚀 Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-AudioX.git
cd ComfyUI-AudioX
pip install -r requirements.txt
```

## 📋 Available Nodes

### Core Nodes
- **AudioX Model Loader**: Load AudioX models with device configuration
- **AudioX Text to Audio**: Generate audio from text descriptions
- **AudioX Text to Music**: Create music from text prompts
- **AudioX Video to Audio**: Generate audio from video content
- **AudioX Video to Music**: Create musical soundtracks for videos

### Processing Nodes
- **AudioX Audio Processor**: Process and enhance audio
- **AudioX Video Muter**: Remove audio from video files
- **AudioX Video Audio Combiner**: Combine video with generated audio

## 🎯 Quick Start

### Basic Text to Audio
1. Add **AudioX Model Loader** node
2. Add **AudioX Text to Audio** node
3. Connect model output to audio generation node
4. Enter your text prompt
5. Execute workflow

### Video to Audio Processing
1. Load video using ComfyUI video nodes
2. Add **AudioX Video Muter** to remove original audio
3. Add **AudioX Video to Audio** for new audio generation
4. Use **AudioX Video Audio Combiner** to merge results

## 📁 Example Workflows

The repository includes example workflows:
- `example_workflow.json` - Basic text to audio
- `audiox_video_to_audio_workflow.json` - Video processing
- `simple_video_to_audio_workflow.json` - Simplified video to audio

## ⚙️ Requirements

- ComfyUI (latest version recommended)
- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- Sufficient disk space for model downloads

## 🔧 Configuration

### Model Storage
Models are automatically downloaded to ComfyUI's standard directories:
- `models/diffusion_models/`
- `models/checkpoints/`
- `models/unet/`

### Device Selection
- Automatic device detection (CUDA/MPS/CPU)
- Manual device specification available
- Memory-efficient processing options

## 🐛 Troubleshooting

### Common Issues

**Frontend Errors**: If you encounter "beforeQueued" errors:
- Refresh browser (Ctrl+R)
- Clear browser cache
- Restart ComfyUI

**Model Loading**: For model download issues:
- Check internet connection
- Ensure sufficient disk space
- Verify model file integrity

**Memory Issues**: For VRAM/RAM problems:
- Reduce batch sizes
- Use CPU mode for large models
- Close other applications

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- AudioX team for original models and research
- ComfyUI community for the excellent framework
- All contributors and testers

## 📈 Version

**Current Version**: v1.0.9
- ✅ Fixed beforeQueued frontend errors
- ✅ Improved workflow execution stability
- ✅ Enhanced video processing capabilities
- ✅ Better error handling and user experience

---

For support and updates, visit the [GitHub repository](https://github.com/your-username/ComfyUI-AudioX).
