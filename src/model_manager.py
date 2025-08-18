import whisper
import torch
import threading
from typing import Dict, Optional
import time

class ModelManager:
    """Singleton class to manage Whisper model loading and caching."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._models: Dict[str, object] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._device = self._get_optimal_device()
        self._initialized = True
        
        print(f"ModelManager initialized with device: {self._device}")
        self._setup_device_optimizations()
    
    def _get_optimal_device(self) -> str:
        """Determine the best device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        else:
            return "cpu"
    
    def _setup_device_optimizations(self):
        """Configure device-specific optimizations."""
        if self._device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
        elif self._device == "mps":
            # MPS (Metal Performance Shaders) optimizations
            torch.backends.mps.empty_cache() if hasattr(torch.backends.mps, 'empty_cache') else None
    
    def get_model(self, model_size: str = "medium", force_reload: bool = False):
        """
        Get a Whisper model, loading it if not already cached.
        
        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            force_reload: Force reloading the model even if cached
            
        Returns:
            Loaded Whisper model
        """
        if model_size not in self._model_locks:
            self._model_locks[model_size] = threading.Lock()
        
        with self._model_locks[model_size]:
            if force_reload or model_size not in self._models:
                print(f"Loading Whisper {model_size} model on {self._device}...")
                start_time = time.time()
                
                # Load model
                model = whisper.load_model(model_size)
                model = model.to(self._device)
                
                # Apply device-specific optimizations
                if self._device == "cuda":
                    model = model.half()  # Use FP16 for CUDA
                elif self._device == "mps":
                    # MPS supports FP16 but may be unstable, use with caution
                    try:
                        model = model.half()
                    except Exception as e:
                        print(f"Warning: Could not use FP16 on MPS, using FP32: {e}")
                
                self._models[model_size] = model
                
                load_time = time.time() - start_time
                print(f"Model {model_size} loaded in {load_time:.2f}s")
            
            return self._models[model_size]
    
    def get_device(self) -> str:
        """Get the current device being used."""
        return self._device
    
    def clear_cache(self):
        """Clear all cached models and free memory."""
        with self._lock:
            self._models.clear()
            self._model_locks.clear()
            
            if self._device == "cuda":
                torch.cuda.empty_cache()
            elif self._device == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        if self._device == "cuda":
            stats["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            stats["gpu_reserved"] = torch.cuda.memory_reserved() / 1024**3   # GB
            stats["gpu_free"] = (torch.cuda.get_device_properties(0).total_memory - 
                                torch.cuda.memory_reserved()) / 1024**3  # GB
        elif self._device == "mps":
            # MPS memory tracking is limited
            stats["mps_allocated"] = torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0
        
        stats["models_loaded"] = list(self._models.keys())
        return stats
    
    def recommend_model_size(self, duration_minutes: float, target_speed: str = "balanced") -> str:
        """
        Recommend optimal model size based on audio duration and performance target.
        
        Args:
            duration_minutes: Audio duration in minutes
            target_speed: "fast", "balanced", or "accurate"
            
        Returns:
            Recommended model size
        """
        if target_speed == "fast":
            if duration_minutes > 60:
                return "base"
            else:
                return "small"
        elif target_speed == "accurate":
            if self._device in ["cuda", "mps"]:
                return "large"
            else:
                return "medium"
        else:  # balanced
            if duration_minutes > 120:
                return "small"
            elif duration_minutes > 30:
                return "medium" if self._device in ["cuda", "mps"] else "small"
            else:
                return "medium"