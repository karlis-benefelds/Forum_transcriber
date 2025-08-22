#!/usr/bin/env python3
"""
Demo script showing the new Advanced Model Selection features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_manager import ModelManager

def demo_intelligent_recommendations():
    """Demonstrate the intelligent model recommendation system."""
    print("🎯 Advanced Model Selection Demo")
    print("=" * 50)
    
    manager = ModelManager()
    print(f"Device detected: {manager.get_device()}")
    
    # Demo scenarios
    scenarios = [
        {"duration": 5, "quality": "fastest", "desc": "Short 5-min meeting"},
        {"duration": 30, "quality": "balanced", "desc": "Standard 30-min lecture"}, 
        {"duration": 90, "quality": "balanced", "desc": "Long 90-min seminar"},
        {"duration": 15, "quality": "highest", "desc": "Critical 15-min presentation"},
        {"duration": 180, "quality": "fastest", "desc": "3-hour workshop (speed needed)"},
    ]
    
    print("\n💡 Intelligent Model Recommendations:")
    print("-" * 70)
    print(f"{'Scenario':<25} {'Model':<8} {'Time Est':<10} {'Quality':<8} {'Memory'}")
    print("-" * 70)
    
    for scenario in scenarios:
        rec = manager.recommend_model_size(
            duration_minutes=scenario["duration"],
            target_quality=scenario["quality"]
        )
        
        print(f"{scenario['desc']:<25} "
              f"{rec['model_size'].upper():<8} "
              f"{rec['estimated_time_minutes']:.1f}min{'':<4} "
              f"{int(rec['quality_score']*100)}%{'':<4} "
              f"{rec['memory_required_gb']:.1f}GB")

def demo_model_specifications():
    """Show detailed model specifications."""
    print("\n📊 Model Specifications:")
    print("-" * 80)
    print(f"{'Model':<8} {'Parameters':<12} {'Memory':<10} {'Speed':<12} {'Best For'}")
    print("-" * 80)
    
    manager = ModelManager()
    models = ["tiny", "base", "small", "medium", "large"]
    
    for model in models:
        info = manager.get_model_info(model)
        print(f"{model.upper():<8} "
              f"{info['parameters']:<12} "
              f"{info['vram_required']:<10} "
              f"{info['relative_speed']:<12} "
              f"{info['best_for']}")

def demo_memory_awareness():
    """Demonstrate memory-aware recommendations."""
    print("\n🧠 Memory-Aware Recommendations:")
    print("-" * 60)
    
    manager = ModelManager()
    memory_scenarios = [2, 4, 8, 16]  # GB
    
    for mem_gb in memory_scenarios:
        rec = manager.recommend_model_size(
            duration_minutes=60,  # 1 hour lecture
            target_quality="highest",
            available_memory_gb=mem_gb
        )
        print(f"With {mem_gb}GB memory: {rec['model_size'].upper()} model recommended")
        print(f"  Available models: {', '.join(rec['available_models']).upper()}")

if __name__ == "__main__":
    demo_intelligent_recommendations()
    demo_model_specifications()
    demo_memory_awareness()
    
    print("\n" + "=" * 50)
    print("🚀 Ready to Use!")
    print("=" * 50)
    print("Command Line Examples:")
    print("  python src/main.py --audio_path lecture.mp4 --curl '...' --target_quality fastest")
    print("  python src/main.py --audio_path lecture.mp4 --curl '...' --model_size large")
    print("\nWeb Interface:")
    print("  python app.py")
    print("  Then visit: http://localhost:8000")