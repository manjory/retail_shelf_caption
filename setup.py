#!/usr/bin/env python3
"""
Setup script for Retail Shelf Captioning Project

pre req
# Fix SSL certificates on macOS

run on commnadlined
/Applications/Python\ 3.*/Install\ Certificates.command

# Or if you're using Homebrew Python:
pip install --upgrade certifi

"""



import subprocess
import sys
import os


def install_requirements():
    """Install all required packages"""

    print("🚀 Setting up Retail Shelf Captioning Project...")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False

    # Install requirements
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

        # Download NLTK data
        print("📚 Downloading NLTK data...")
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')

        print("✅ Setup completed successfully!")
        print("\n🎯 Next steps:")
        print("1. Generate dataset: python clone_data.py")
        print("2. Train model: python src/train_model.py")
        print("3. Generate captions: python caption.py image.jpg")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False


if __name__ == "__main__":
    install_requirements()
