"""
Environment Setup and Verification Script for DeepReach
========================================================

This script verifies and sets up the PyTorch/CUDA environment for training
DeepReach models. It checks dependencies and provides installation guidance.

Usage:
    python setup_environment.py
"""

import sys
import platform
import subprocess
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_python():
    """Check Python version."""
    print_section("Python Version")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python {version}")
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        print("✓ Python version is compatible (>=3.8)")
        return True
    else:
        print("✗ Python version is too old. Please upgrade to Python 3.8+")
        return False

def check_pytorch():
    """Check PyTorch installation."""
    print_section("PyTorch Installation")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} is installed")
        return True
    except ImportError:
        print("✗ PyTorch is NOT installed")
        print("\nTo install PyTorch with CUDA support, run:")
        print("  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\nFor other CUDA versions, visit: https://pytorch.org/get-started/locally/")
        return False

def check_cuda():
    """Check CUDA availability."""
    print_section("CUDA/GPU Support")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            
            # Test CUDA functionality
            try:
                test_tensor = torch.randn(10, 10).cuda()
                print("✓ CUDA tensor operations working")
                return True
            except Exception as e:
                print(f"✗ CUDA tensor operations failed: {e}")
                return False
        else:
            print("⚠ CUDA is NOT available. CPU-only training will be slow.")
            print("  To enable GPU support, ensure you have:")
            print("  - NVIDIA CUDA Toolkit 12.1+ installed")
            print("  - NVIDIA cuDNN 8.9+ installed")
            print("  - Compatible NVIDIA GPU (Compute Capability 3.5+)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def check_dependencies():
    """Check key dependencies."""
    print_section("Key Dependencies")
    
    required = ['numpy', 'matplotlib', 'scipy', 'torch', 'tensorboard', 'plotly']
    failed = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            failed.append(package)
    
    if failed:
        print(f"\nTo install missing packages, run:")
        print(f"  pip install {' '.join(failed)}")
        return False
    return True

def check_requirements_file():
    """Check if requirements.txt exists and is accessible."""
    print_section("Requirements File")
    req_file = Path(__file__).parent / 'requirements.txt'
    
    if req_file.exists():
        print(f"✓ Found: {req_file}")
        print(f"  To install all requirements, run:")
        print(f"  pip install -r requirements.txt")
        return True
    else:
        print(f"✗ requirements.txt not found at {req_file}")
        return False

def check_project_structure():
    """Check if project structure is correct."""
    print_section("Project Structure")
    
    project_root = Path(__file__).parent
    required_dirs = ['dynamics', 'experiments', 'utils', 'runs']
    required_files = ['run_experiment.py', 'requirements.txt']
    
    all_good = True
    
    print("Checking directories:")
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.is_dir():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - MISSING")
            all_good = False
    
    print("\nChecking files:")
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.is_file():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} - MISSING")
            all_good = False
    
    return all_good

def check_wandb():
    """Check WandB (optional)."""
    print_section("Weights & Biases (Optional)")
    
    try:
        import wandb
        print(f"✓ wandb is installed")
        print("\n  To use wandb for logging, run:")
        print("  wandb login")
        return True
    except ImportError:
        print("⚠ wandb is NOT installed (optional)")
        print("  For wandb integration, install with: pip install wandb")
        return False

def main():
    """Run all checks."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  DeepReach Environment Verification Script".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    print(f"\nPlatform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    results = {
        'Python Version': check_python(),
        'PyTorch': check_pytorch(),
        'CUDA/GPU': check_cuda(),
        'Dependencies': check_dependencies(),
        'Requirements File': check_requirements_file(),
        'Project Structure': check_project_structure(),
        'WandB (Optional)': check_wandb(),
    }
    
    # Summary
    print_section("Summary")
    
    critical = ['Python Version', 'PyTorch', 'Dependencies', 'Project Structure']
    critical_pass = all(results[key] for key in critical)
    
    for category, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{category:<30} {status}")
    
    print("\n" + "="*70)
    
    if critical_pass:
        print("✓ ENVIRONMENT READY FOR TRAINING")
        print("\nYou can now run the training script:")
        print("  python train_dubins3d.py")
        
        if not results['CUDA/GPU']:
            print("\nNote: Training will use CPU. For faster training, enable CUDA.")
    else:
        print("✗ ENVIRONMENT NOT READY")
        print("\nPlease fix the CRITICAL issues above before training.")
        sys.exit(1)
    
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
