#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .utils import load_ckpt_state_dict

from torch import nn
# Lazy import transformers to prevent hanging during startup
# from transformers import AutoProcessor, CLIPVisionModelWithProjection
import einops
from .temptransformer import SA_Transformer
from torchvision import transforms
import torch
import einops
import torchvision.transforms as transforms


class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)
            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]


class CLIPConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(
            self,
            output_dim: int,
            clip_model_name: str = "clip-vit-base-patch32",
            video_fps: int = 5,
            out_features: str = 128,
            enable_grad: bool = False,
            in_features: int = 5000,
            project_out: bool = False,
    ):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim = 768, output_dim=output_dim, project_out=project_out)
        
        sa_depth=4
        num_heads=16
        dim_head=64
        hidden_scale=4
        duration = 10
        
        self.clip_model_name=clip_model_name
                
        if self.clip_model_name=='clip-vit-base-patch32':
            out_features = 128
            temporal_dim=768

            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)

            in_features = 50*video_fps*duration

            # Lazy import to prevent hanging during startup
            self.visual_encoder_model = None  # Will be loaded when needed
            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            
            self.in_features = in_features
            self.out_features = out_features

            self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim*hidden_scale, 0.)
            self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration*video_fps, temporal_dim))

            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            self.preprocess_CLIP = transforms.Compose([
                transforms.Normalize(mean=clip_mean, std=clip_std)
            ])

    def _load_visual_encoder(self):
        """Lazy loading of CLIP visual encoder to prevent startup hangs."""
        if self.visual_encoder_model is None:
            try:
                print("AudioX: Loading CLIP visual encoder...")
                from transformers import CLIPVisionModelWithProjection
                self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-base-patch32')
                print("AudioX: ✅ CLIP visual encoder loaded successfully!")
            except Exception as e:
                print(f"AudioX: ❌ Failed to load CLIP visual encoder: {e}")
                # Create a dummy model that returns zeros
                class DummyVisualEncoder:
                    def __call__(self, pixel_values):
                        batch_size = pixel_values.shape[0]
                        return type('obj', (object,), {'last_hidden_state': torch.zeros(batch_size, 50, 768)})()
                    def to(self, device):
                        return self
                    def eval(self):
                        return self
                self.visual_encoder_model = DummyVisualEncoder()
        return self.visual_encoder_model

    def process_video_with_custom_preprocessing(self, video_tensor: torch.Tensor):
        print(f"AudioX CLIP: Input to preprocessing shape: {video_tensor.shape}")

        # Ensure tensor is in the right range [0, 1]
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0

        # Ensure we have exactly 3 channels (RGB)
        if video_tensor.shape[1] != 3:
            print(f"AudioX CLIP: Warning - video has {video_tensor.shape[1]} channels, expected 3")
            if video_tensor.shape[1] > 3:
                video_tensor = video_tensor[:, :3, :, :]  # Take first 3 channels
            else:
                # Repeat channels to get 3
                video_tensor = video_tensor.repeat(1, 3, 1, 1)[:, :3, :, :]

        print(f"AudioX CLIP: After channel fix shape: {video_tensor.shape}")

        # Resize to 224x224 for CLIP model
        current_height, current_width = video_tensor.shape[2], video_tensor.shape[3]
        if current_height != 224 or current_width != 224:
            print(f"AudioX CLIP: Resizing from {current_height}x{current_width} to 224x224 using torchvision.transforms.Resize")
            resize_op = transforms.Resize([224, 224], antialias=True)
            video_tensor = resize_op(video_tensor)
            print(f"AudioX CLIP: After resize shape: {video_tensor.shape}")

        # Apply CLIP normalization
        try:
            video_tensor = self.preprocess_CLIP(video_tensor)
            print(f"AudioX CLIP: After normalization shape: {video_tensor.shape}")
        except Exception as e:
            print(f"AudioX CLIP: Normalization failed: {e}")
            print(f"AudioX CLIP: Tensor stats - min: {video_tensor.min()}, max: {video_tensor.max()}, shape: {video_tensor.shape}")
            raise

        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, Video_tensors: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        visual_encoder_model = self._load_visual_encoder().eval().to(device)
        proj = self.proj.to(device)

        # Handle None values in Video_tensors
        if Video_tensors is None or all(v is None for v in Video_tensors):
            # Return empty video features when no video is provided
            batch_size = 1
            empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
            return empty_visual_feat, torch.ones(batch_size, 1).to(device)

        # Filter out None values
        valid_videos = [v for v in Video_tensors if v is not None]
        if not valid_videos:
            # Return empty video features when no valid videos
            batch_size = 1
            empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
            return empty_visual_feat, torch.ones(batch_size, 1).to(device)

        original_videos = torch.cat(valid_videos, dim=0).to(device)
        print(f"AudioX CLIP: Original videos shape: {original_videos.shape}")

        # Handle different video tensor formats
        if original_videos.dim() == 5:
            # Format: [B, T, C, H, W]
            batch_size, time_length, _, _, _ = original_videos.size()
            is_zero = torch.all(original_videos == 0, dim=(1,2,3,4))
            Video_tensors = original_videos
            Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')
        elif original_videos.dim() == 4:
            # Format: [T, C, H, W] - add batch dimension
            time_length, _, _, _ = original_videos.size()
            batch_size = 1
            is_zero = torch.all(original_videos == 0, dim=(1,2,3))
            Video_tensors = original_videos  # Already in correct format for processing
        else:
            raise ValueError(f"Unexpected video tensor dimensions: {original_videos.shape}")

        print(f"AudioX CLIP: Video tensors shape for processing: {Video_tensors.shape}")

        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)
        if self.clip_model_name=='clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state # Shape: (actual_time_length, num_patches, clip_hidden_dim) if batch_size is 1 for pixel_values

            # If batch_size for pixel_values was > 1, video_hidden is (B_frames * actual_time_length, num_patches, clip_hidden_dim)
            # The 'time_length' variable here is the original number of frames from the input video.
            # Let's call it input_time_length to avoid confusion.
            input_time_length = time_length

            # Rearrange to (batch_size * num_patches, input_time_length, temporal_dim)
            # Example if original video_hidden was (input_time_length, num_patches, hidden_dim) and batch_size = 1:
            #   (1 * input_time_length, num_patches, hidden_dim) -> (1 * num_patches, input_time_length, hidden_dim)
            #   So, (input_time_length, 50, 768) -> (50, input_time_length, 768)
            video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h', b=batch_size, t=input_time_length)
            
            # Ensure video_hidden's time dimension matches Temp_pos_embedding's expected time dimension
            actual_time_dim_video = video_hidden.shape[1]
            expected_time_dim_pos = self.Temp_pos_embedding.shape[1]

            if actual_time_dim_video < expected_time_dim_pos:
                padding_size = expected_time_dim_pos - actual_time_dim_video
                padding = torch.zeros(video_hidden.shape[0], padding_size, video_hidden.shape[2],
                                      device=video_hidden.device, dtype=video_hidden.dtype)
                video_hidden = torch.cat((video_hidden, padding), dim=1)
            elif actual_time_dim_video > expected_time_dim_pos:
                video_hidden = video_hidden[:, :expected_time_dim_pos, :]
            
            # After padding/truncation, this dimension is now expected_time_dim_pos
            current_time_length_for_ops = expected_time_dim_pos

            video_hidden += self.Temp_pos_embedding # Now dimensions should match
            video_hidden = self.Temp_transformer(video_hidden)
            # Use current_time_length_for_ops (which is expected_time_dim_pos) for the rearrange
            video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h', b=batch_size, t=current_time_length_for_ops)

        video_hidden = proj(video_hidden.view(-1, self.in_features)) # self.in_features expects current_time_length_for_ops
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)
                
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


    
class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False,
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        self.t5_model_name = t5_model_name
        self.max_length = max_length
        self.enable_grad = enable_grad

        # Lazy loading - don't load the model during init
        self.tokenizer = None
        self.model = None

    def _load_t5_model(self):
        """Lazy loading of T5 model to prevent startup hangs."""
        if self.model is None or self.tokenizer is None:
            try:
                print(f"AudioX: Loading T5 model {self.t5_model_name}...")
                from transformers import T5EncoderModel, AutoTokenizer

                # Suppress logging from transformers
                previous_level = logging.root.manager.disable
                logging.disable(logging.ERROR)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.t5_model_name)
                        model = T5EncoderModel.from_pretrained(self.t5_model_name).train(self.enable_grad).requires_grad_(self.enable_grad).to(torch.float16)
                    finally:
                        logging.disable(previous_level)

                if self.enable_grad:
                    self.model = model
                else:
                    self.__dict__["model"] = model

                print(f"AudioX: ✅ T5 model {self.t5_model_name} loaded successfully!")
            except Exception as e:
                print(f"AudioX: ❌ Failed to load T5 model: {e}")
                # Create dummy tokenizer and model
                class DummyTokenizer:
                    def __call__(self, texts, **kwargs):
                        return {"input_ids": torch.zeros(len(texts), 128, dtype=torch.long),
                               "attention_mask": torch.ones(len(texts), 128, dtype=torch.bool)}

                class DummyT5Model:
                    def __call__(self, **kwargs):
                        batch_size = kwargs["input_ids"].shape[0]
                        seq_len = kwargs["input_ids"].shape[1]
                        return {"last_hidden_state": torch.zeros(batch_size, seq_len, self.T5_MODEL_DIMS[self.t5_model_name])}
                    def to(self, device):
                        return self
                    def eval(self):
                        return self

                self.tokenizer = DummyTokenizer()
                self.model = DummyT5Model()

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        # Load model lazily
        self._load_t5_model()

        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]    
            
        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask    
    
class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p
        self.max_length = max_length
        self.g2p = G2p()
        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        phoneme_ignore = [" ", *string.punctuation]
        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)



class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.cat(audio, dim=0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        
        latents = self.pretransform.encode(audio)
        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]


class AudioAutoencoderConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform        
        self.empty_audio_feat = nn.Parameter(torch.zeros(1, 215, self.proj_out.out_features), requires_grad=True)
        nn.init.constant_(self.empty_audio_feat, 0)

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            original_audios = torch.cat(audio, dim=0).to(device)
            is_zero = torch.all(original_audios == 0, dim=(1,2))  
            audio = original_audios

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        latents = self.pretransform.encode(audio)
        latents = latents.permute(0, 2, 1)
        latents = self.proj_out(latents)

        # Get the target sequence length from latents
        # latents shape is (batch_size, seq_len, num_channels) after permute and proj_out
        # So, target_seq_len is latents.shape[1]
        target_seq_len = latents.shape[1]

        # Get the base empty_audio_feat and its fixed sequence length
        # self.empty_audio_feat shape is (1, fixed_seq_len, output_dim)
        base_empty_audio_feat = self.empty_audio_feat.to(device) # Ensure it's on the correct device
        fixed_seq_len = base_empty_audio_feat.shape[1] # Original fixed sequence length

        # Expand base_empty_audio_feat to current batch size
        expanded_empty_feat = base_empty_audio_feat.expand(latents.shape[0], -1, -1)

        # Adjust sequence length
        if target_seq_len == fixed_seq_len:
            adjusted_empty_audio_feat = expanded_empty_feat
        elif target_seq_len > fixed_seq_len:
            if fixed_seq_len == 0: # Avoid division by zero if fixed_seq_len is 0
                # Fallback: create zeros of the target shape. This case should ideally not happen with a proper empty_audio_feat.
                adjusted_empty_audio_feat = torch.zeros_like(latents)
            else:
                num_repeats = (target_seq_len + fixed_seq_len - 1) // fixed_seq_len
                adjusted_empty_audio_feat = expanded_empty_feat.repeat(1, num_repeats, 1)
                adjusted_empty_audio_feat = adjusted_empty_audio_feat[:, :target_seq_len, :]
        else: # target_seq_len < fixed_seq_len
            adjusted_empty_audio_feat = expanded_empty_feat[:, :target_seq_len, :]

        is_zero_expanded = is_zero.view(latents.shape[0], 1, 1)
        latents = torch.where(is_zero_expanded, adjusted_empty_audio_feat, latents)
        return [latents, torch.ones(latents.shape[0], latents.shape[1]).to(latents.device)] # latents.shape[2] changed to latents.shape[1] for mask
    

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)
            
            output[key] = conditioner(conditioner_inputs, device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clip":
            conditioners[id] = CLIPConditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
            
        elif conditioner_type == "audio_autoencoder":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = AudioAutoencoderConditioner(pretransform, **conditioner_config)            
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)