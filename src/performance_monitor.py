import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    processing_speed: Optional[float]  # x real-time
    segments_processed: int
    total_segments: int
    model_size: str
    device: str

class PerformanceMonitor:
    """Monitor and track performance metrics during transcription."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Current job tracking
        self.current_job = {
            'start_time': None,
            'audio_duration': 0,
            'segments_processed': 0,
            'total_segments': 0,
            'model_size': 'medium',
            'device': 'cpu'
        }
    
    def start_monitoring(self, job_info: Dict):
        """Start performance monitoring for a job."""
        with self.lock:
            if self.monitoring:
                return
            
            self.monitoring = True
            self.current_job.update(job_info)
            self.current_job['start_time'] = time.time()
            self.metrics_history.clear()
            
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            print(f"Performance monitoring started for {job_info.get('model_size', 'unknown')} model on {job_info.get('device', 'unknown')}")
    
    def stop_monitoring(self) -> Dict:
        """Stop monitoring and return performance summary."""
        with self.lock:
            if not self.monitoring:
                return {}
            
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)
            
            return self._generate_summary()
    
    def update_progress(self, segments_processed: int, total_segments: int):
        """Update job progress."""
        with self.lock:
            self.current_job['segments_processed'] = segments_processed
            self.current_job['total_segments'] = total_segments
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 100 measurements
                    if len(self.metrics_history) > 100:
                        self.metrics_history = self.metrics_history[-100:]
                
                time.sleep(2.0)  # Sample every 2 seconds
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        
        # GPU metrics
        gpu_memory_used_gb = 0
        gpu_memory_total_gb = 0
        
        try:
            import torch
            device = self.current_job.get('device', 'cpu')
            
            if device == 'cuda' and torch.cuda.is_available():
                gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif device == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
                gpu_memory_used_gb = torch.mps.current_allocated_memory() / (1024**3)
                # MPS doesn't provide total memory easily
                gpu_memory_total_gb = 16.0  # Estimate for Apple Silicon
        except ImportError:
            pass
        
        # Processing speed calculation
        processing_speed = None
        if (self.current_job['start_time'] and 
            self.current_job['segments_processed'] > 0 and 
            self.current_job['audio_duration'] > 0):
            
            elapsed = time.time() - self.current_job['start_time']
            processed_duration = (self.current_job['segments_processed'] / 
                                self.current_job['total_segments']) * self.current_job['audio_duration']
            if elapsed > 0:
                processing_speed = processed_duration / elapsed
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            processing_speed=processing_speed,
            segments_processed=self.current_job['segments_processed'],
            total_segments=self.current_job['total_segments'],
            model_size=self.current_job['model_size'],
            device=self.current_job['device']
        )
    
    def _generate_summary(self) -> Dict:
        """Generate performance summary from collected metrics."""
        if not self.metrics_history:
            return {}
        
        # Calculate averages and peaks
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        gpu_memory_values = [m.gpu_memory_used_gb for m in self.metrics_history]
        speed_values = [m.processing_speed for m in self.metrics_history if m.processing_speed]
        
        total_time = time.time() - (self.current_job['start_time'] or time.time())
        
        summary = {
            'total_processing_time': total_time,
            'audio_duration': self.current_job['audio_duration'],
            'real_time_factor': speed_values[-1] if speed_values else None,
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            'peak_cpu_percent': max(cpu_values) if cpu_values else 0,
            'avg_memory_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
            'peak_memory_percent': max(memory_values) if memory_values else 0,
            'peak_gpu_memory_gb': max(gpu_memory_values) if gpu_memory_values else 0,
            'model_size': self.current_job['model_size'],
            'device': self.current_job['device'],
            'segments_processed': self.current_job['segments_processed'],
            'total_segments': self.current_job['total_segments'],
            'avg_processing_speed': sum(speed_values) / len(speed_values) if speed_values else None
        }
        
        return summary
    
    def get_current_metrics(self) -> Optional[Dict]:
        """Get the latest metrics."""
        with self.lock:
            if not self.metrics_history:
                return None
            return asdict(self.metrics_history[-1])
    
    def get_metrics_history(self) -> List[Dict]:
        """Get all collected metrics."""
        with self.lock:
            return [asdict(m) for m in self.metrics_history]