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
            # MPS has compatibility issues with Whisper models, use CPU for now
            print("⚠️  MPS (Apple Silicon GPU) detected but using CPU for better compatibility")
            return "cpu"
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
    
    def recommend_model_size(self, duration_minutes: float, target_quality: str = "balanced", available_memory_gb: float = None) -> dict:
        """
        Recommend optimal model size based on audio duration, quality target, and available resources.
        
        Args:
            duration_minutes: Audio duration in minutes
            target_quality: "fastest", "balanced", or "highest"
            available_memory_gb: Available GPU/system memory in GB (auto-detected if None)
            
        Returns:
            Dictionary with recommended model, estimated time, and resource requirements
        """
        if available_memory_gb is None:
            available_memory_gb = self._estimate_available_memory()
        
        # Model memory requirements (approximate, in GB)
        model_memory = {
            "tiny": 0.5,
            "base": 0.7,
            "small": 1.5,
            "medium": 3.0,
            "large": 6.0
        }
        
        # Processing speed multipliers relative to medium model
        speed_multipliers = {
            "tiny": 3.0,
            "base": 2.2,
            "small": 1.5,
            "medium": 1.0,
            "large": 0.6
        }
        
        # Quality scores (relative accuracy)
        quality_scores = {
            "tiny": 0.6,
            "base": 0.7,
            "small": 0.8,
            "medium": 0.9,
            "large": 1.0
        }
        
        # Filter models by available memory
        available_models = [model for model, mem in model_memory.items() 
                          if mem <= available_memory_gb]
        
        if not available_models:
            available_models = ["tiny"]  # Fallback to smallest model
        
        if target_quality == "fastest":
            # Prioritize speed, choose fastest available model
            recommended = min(available_models, 
                            key=lambda x: duration_minutes / speed_multipliers[x])
        elif target_quality == "highest":
            # Prioritize quality, choose best available model
            recommended = max(available_models, key=lambda x: quality_scores[x])
        else:  # balanced
            # Balance quality and speed based on duration
            if duration_minutes > 120:
                # Long recordings: favor speed
                candidates = ["tiny", "base", "small"]
            elif duration_minutes > 30:
                # Medium recordings: balanced approach
                candidates = ["small", "medium"]
            else:
                # Short recordings: can afford better quality
                candidates = ["medium", "large"]
            
            # Choose best available from candidates
            available_candidates = [m for m in candidates if m in available_models]
            recommended = available_candidates[-1] if available_candidates else available_models[0]
        
        # Estimate processing time (baseline: 1 minute of audio = 5 seconds processing on medium/CUDA)
        base_processing_time = duration_minutes * 5.0  # seconds
        if self._device == "cpu":
            base_processing_time *= 3.0  # CPU is ~3x slower
        
        estimated_time = base_processing_time / speed_multipliers[recommended]
        
        return {
            "model_size": recommended,
            "estimated_time_minutes": estimated_time / 60.0,
            "memory_required_gb": model_memory[recommended],
            "quality_score": quality_scores[recommended],
            "speed_multiplier": speed_multipliers[recommended],
            "available_models": available_models,
            "target_quality": target_quality
        }
    
    def _estimate_available_memory(self) -> float:
        """Estimate available memory for model loading."""
        if self._device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    total_memory = props.total_memory / (1024**3)  # GB
                    # Reserve 20% for system and other processes
                    return total_memory * 0.8
            except Exception:
                pass
            return 8.0  # Conservative fallback for CUDA
        elif self._device == "mps":
            # Apple Silicon typically has 8-64GB unified memory
            return 16.0  # Conservative estimate
        else:
            # CPU - use system RAM (but models are much slower)
            try:
                import psutil
                return psutil.virtual_memory().available / (1024**3) * 0.3  # 30% of available RAM
            except Exception:
                return 4.0  # Fallback
    
    def get_model_info(self, model_size: str) -> dict:
        """Get detailed information about a specific model size."""
        model_info = {
            "tiny": {
                "parameters": "39M",
                "vram_required": "~0.5GB",
                "relative_speed": "3x faster",
                "relative_accuracy": "Good for clear speech",
                "best_for": "Quick transcription of clear audio"
            },
            "base": {
                "parameters": "74M", 
                "vram_required": "~0.7GB",
                "relative_speed": "2x faster",
                "relative_accuracy": "Good general purpose",
                "best_for": "Balanced speed and quality"
            },
            "small": {
                "parameters": "244M",
                "vram_required": "~1.5GB", 
                "relative_speed": "1.5x faster",
                "relative_accuracy": "Good for most content",
                "best_for": "Standard transcription needs"
            },
            "medium": {
                "parameters": "769M",
                "vram_required": "~3GB",
                "relative_speed": "Baseline",
                "relative_accuracy": "Very good (recommended)",
                "best_for": "High-quality transcription"
            },
            "large": {
                "parameters": "1550M",
                "vram_required": "~6GB", 
                "relative_speed": "1.7x slower",
                "relative_accuracy": "Best available",
                "best_for": "Critical accuracy requirements"
            }
        }
        
        return model_info.get(model_size, {
            "parameters": "Unknown",
            "vram_required": "Unknown", 
            "relative_speed": "Unknown",
            "relative_accuracy": "Unknown",
            "best_for": "Unknown model size"
        })