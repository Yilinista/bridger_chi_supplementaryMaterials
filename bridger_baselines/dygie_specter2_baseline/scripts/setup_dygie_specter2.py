#!/usr/bin/env python3
"""
Setup script for DyGIE++ + SPECTER2 Bridger baseline

This script helps set up the required dependencies and models for running
the improved Bridger baseline with DyGIE++ term extraction and SPECTER2 embeddings.
"""

import os
import sys
import subprocess
import urllib.request
import tarfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        raise RuntimeError("Python 3.7 or higher is required")
    logger.info(f"✅ Python version: {sys.version}")


def install_python_packages():
    """Install required Python packages"""
    logger.info("Installing Python packages...")
    
    packages = [
        "torch>=1.6.0",
        "sentence-transformers>=2.0.0", 
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "spacy>=3.0.0",
        "scispacy>=0.4.0"
    ]
    
    for package in packages:
        logger.info(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            raise


def install_spacy_models():
    """Install required spaCy models"""
    logger.info("Installing spaCy models...")
    
    models = [
        "en_core_web_sm",  # Basic English model
        "en_core_sci_sm"   # Scientific English model (from scispacy)
    ]
    
    for model in models:
        logger.info(f"Installing spaCy model: {model}")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model])
        except subprocess.CalledProcessError as e:
            if model == "en_core_sci_sm":
                # Try installing from scispacy
                logger.info("Trying to install scientific model from scispacy...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", 
                        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz"
                    ])
                except subprocess.CalledProcessError:
                    logger.warning(f"Could not install {model}, will use en_core_web_sm as fallback")
            else:
                logger.error(f"Failed to install {model}: {e}")
                raise


def setup_dygiepp():
    """Set up DyGIE++ repository and model"""
    logger.info("Setting up DyGIE++...")
    
    dygiepp_dir = Path("dygiepp")
    
    # Clone DyGIE++ repository if not exists
    if not dygiepp_dir.exists():
        logger.info("Cloning DyGIE++ repository...")
        try:
            subprocess.check_call([
                "git", "clone", "https://github.com/dwadden/dygiepp.git"
            ])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone DyGIE++ repository: {e}")
            raise
    
    # Install DyGIE++ requirements
    requirements_file = dygiepp_dir / "requirements.txt"
    if requirements_file.exists():
        logger.info("Installing DyGIE++ requirements...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some DyGIE++ requirements may have failed to install: {e}")
    
    # Download SciERC model
    model_dir = dygiepp_dir / "pretrained_models"
    model_dir.mkdir(exist_ok=True)
    
    scierc_model_path = model_dir / "scierc"
    if not scierc_model_path.exists():
        logger.info("Downloading SciERC model (this may take a while)...")
        
        model_url = "https://ai2-s2-dygiepp.s3.amazonaws.com/models/scierc.tar.gz"
        model_file = model_dir / "scierc.tar.gz"
        
        try:
            # Download model
            urllib.request.urlretrieve(model_url, model_file)
            
            # Extract model
            with tarfile.open(model_file, 'r:gz') as tar:
                tar.extractall(model_dir)
            
            # Clean up tar file
            model_file.unlink()
            
            logger.info("✅ SciERC model downloaded and extracted")
            
        except Exception as e:
            logger.error(f"Failed to download SciERC model: {e}")
            logger.info("You can manually download it from:")
            logger.info("https://ai2-s2-dygiepp.s3.amazonaws.com/models/scierc.tar.gz")
            raise


def test_installation():
    """Test if all components are working"""
    logger.info("Testing installation...")
    
    # Test imports
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✅ Sentence Transformers imported successfully")
    except ImportError:
        logger.error("❌ Sentence Transformers not installed")
        return False
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        logger.info("✅ spaCy basic model loaded")
        
        try:
            nlp_sci = spacy.load("en_core_sci_sm") 
            logger.info("✅ spaCy scientific model loaded")
        except OSError:
            logger.warning("⚠️ Scientific spaCy model not available, will use basic model")
            
    except (ImportError, OSError):
        logger.error("❌ spaCy not properly installed")
        return False
    
    # Test SPECTER2 model loading
    try:
        logger.info("Testing SPECTER2 model loading...")
        model = SentenceTransformer('allenai/specter2_base')
        test_text = ["This is a test sentence about machine learning."]
        embeddings = model.encode(test_text)
        logger.info(f"✅ SPECTER2 model working, embedding shape: {embeddings.shape}")
    except Exception as e:
        logger.error(f"❌ SPECTER2 model test failed: {e}")
        return False
    
    # Test DyGIE++ setup
    dygiepp_dir = Path("dygiepp")
    if dygiepp_dir.exists():
        model_path = dygiepp_dir / "pretrained_models" / "scierc"
        if model_path.exists():
            logger.info("✅ DyGIE++ SciERC model found")
        else:
            logger.warning("⚠️ DyGIE++ SciERC model not found")
            return False
    else:
        logger.warning("⚠️ DyGIE++ repository not found")
        return False
    
    logger.info("🎉 All components tested successfully!")
    return True


def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*70)
    print("SETUP COMPLETED - USAGE INSTRUCTIONS")
    print("="*70)
    
    print("\n1. Generate embeddings (first time, may take hours):")
    print("   python embedding_generator.py \\")
    print("     --evaluation-data /path/to/your/evaluation.csv \\")
    print("     --force-regenerate")
    
    print("\n2. Run improved baseline evaluation (fast after embeddings are generated):")
    print("   python bridger_baselines_improved.py \\")
    print("     --evaluation-data /path/to/your/evaluation.csv")
    
    print("\n3. Compare with original baseline:")
    print("   python bridger_baselines_improved.py \\")
    print("     --evaluation-data /path/to/your/evaluation.csv \\")
    print("     --compare-original")
    
    print("\n4. Check embedding statistics:")
    print("   python bridger_baselines_improved.py --stats-only")
    
    print("\n5. For help with any script:")
    print("   python [script_name].py --help")
    
    print("\nFiles created:")
    print("  - embedding_generator.py: Generate and store embeddings")
    print("  - bridger_baselines_improved.py: Run improved baseline evaluation")
    print("  - setup_dygie_specter2.py: This setup script")
    
    print(f"\nEmbeddings will be stored in: ./bridger_embeddings/")
    print(f"DyGIE++ installed in: ./dygiepp/")


def main():
    """Main setup function"""
    logger.info("Starting DyGIE++ + SPECTER2 setup...")
    
    try:
        # Check Python version
        check_python_version()
        
        # Install Python packages
        install_python_packages()
        
        # Install spaCy models
        install_spacy_models()
        
        # Setup DyGIE++
        setup_dygiepp()
        
        # Test installation
        if test_installation():
            logger.info("🎉 Setup completed successfully!")
            print_usage_instructions()
        else:
            logger.error("❌ Setup completed but some tests failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()