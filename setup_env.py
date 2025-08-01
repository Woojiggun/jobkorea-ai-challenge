"""
Environment setup script for JobKorea AI Challenge
Handles dependency installation and compatibility checks
"""
import subprocess
import sys
import os
from pathlib import Path

def create_venv():
    """Create virtual environment"""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    print("[OK] Virtual environment created")

def get_pip_path():
    """Get pip executable path"""
    if os.name == 'nt':  # Windows
        return Path("venv/Scripts/pip.exe")
    else:  # Unix-like
        return Path("venv/bin/pip")

def upgrade_pip():
    """Upgrade pip to latest version"""
    pip_path = get_pip_path()
    print("\nUpgrading pip...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    print("[OK] Pip upgraded")

def install_dependencies():
    """Install dependencies with compatibility fixes"""
    pip_path = get_pip_path()
    
    # Install core dependencies first
    print("\nInstalling core dependencies...")
    core_deps = [
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.24.0",
        "faiss-cpu>=1.7.0",
        "networkx>=2.5",
        "pydantic>=1.8.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "aiofiles>=0.8.0",
        "psutil>=5.8.0"
    ]
    
    for dep in core_deps:
        print(f"Installing {dep}...")
        subprocess.run([str(pip_path), "install", dep], check=True)
    
    # Install sentence-transformers with specific versions for compatibility
    print("\nInstalling sentence-transformers with compatible versions...")
    subprocess.run([
        str(pip_path), "install", 
        "transformers==4.36.0",
        "huggingface-hub==0.19.4",
        "sentence-transformers==2.2.2"
    ], check=True)
    
    # Install optional dependencies
    print("\nInstalling optional dependencies...")
    optional_deps = [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "pytest-asyncio>=0.15.0",
        "python-dotenv>=0.19.0"
    ]
    
    for dep in optional_deps:
        try:
            subprocess.run([str(pip_path), "install", dep], check=True)
        except subprocess.CalledProcessError:
            print(f"[WARNING] Failed to install optional dependency: {dep}")

def verify_installation():
    """Verify installation by importing key modules"""
    print("\nVerifying installation...")
    
    # Activate virtual environment for imports
    if os.name == 'nt':
        activate_cmd = "venv\\Scripts\\activate.bat && python -c"
    else:
        activate_cmd = "source venv/bin/activate && python -c"
    
    test_imports = [
        "import numpy",
        "import faiss",
        "import networkx",
        "import sentence_transformers",
        "from sentence_transformers import SentenceTransformer"
    ]
    
    for test_import in test_imports:
        try:
            if os.name == 'nt':
                result = subprocess.run(
                    ["venv\\Scripts\\python.exe", "-c", test_import],
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    ["venv/bin/python", "-c", test_import],
                    capture_output=True,
                    text=True
                )
            
            if result.returncode == 0:
                print(f"[OK] {test_import}")
            else:
                print(f"[FAIL] {test_import}")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"[ERROR] {test_import}: {e}")

def create_activation_scripts():
    """Create convenient activation scripts"""
    # Windows batch file
    with open("activate.bat", "w") as f:
        f.write("@echo off\n")
        f.write("call venv\\Scripts\\activate.bat\n")
        f.write("echo Virtual environment activated!\n")
    
    # Unix shell script
    with open("activate.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("source venv/bin/activate\n")
        f.write("echo 'Virtual environment activated!'\n")
    
    # Make shell script executable on Unix
    if os.name != 'nt':
        os.chmod("activate.sh", 0o755)
    
    print("\n[OK] Activation scripts created:")
    print("  - Windows: activate.bat")
    print("  - Unix: source activate.sh")

def main():
    """Main setup function"""
    print("JobKorea AI Challenge - Environment Setup")
    print("=" * 50)
    
    try:
        # Create virtual environment
        create_venv()
        
        # Upgrade pip
        upgrade_pip()
        
        # Install dependencies
        install_dependencies()
        
        # Verify installation
        verify_installation()
        
        # Create activation scripts
        create_activation_scripts()
        
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("\nTo activate the environment:")
        if os.name == 'nt':
            print("  Run: activate.bat")
        else:
            print("  Run: source activate.sh")
        print("\nThen run tests:")
        print("  python test_simple.py")
        print("  python scripts/demo.py")
        
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()