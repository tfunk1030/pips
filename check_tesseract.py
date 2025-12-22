#!/usr/bin/env python3
"""Verify Tesseract OCR installation and version."""
import shutil
import subprocess
import sys


def check_tesseract():
    """Check Tesseract binary installation and version."""
    # Check if tesseract is in PATH
    tesseract_path = shutil.which("tesseract")

    if not tesseract_path:
        print("ERROR: Tesseract not found in PATH")
        print("Please install Tesseract OCR:")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("  macOS: brew install tesseract")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        return False

    print(f"Tesseract found at: {tesseract_path}")

    # Get version
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        version_output = result.stdout + result.stderr
        print("\nTesseract version info:")
        print(version_output)

        # Parse version number
        lines = version_output.strip().split('\n')
        if lines:
            version_line = lines[0]
            if "tesseract" in version_line.lower():
                # Check for version 4.x or 5.x (LSTM support)
                if " 4." in version_line or " 5." in version_line or "v4." in version_line or "v5." in version_line:
                    print("\n✓ Tesseract version 4.x or 5.x detected (LSTM support available)")
                    return True
                elif " 3." in version_line or "v3." in version_line:
                    print("\n⚠ Tesseract version 3.x detected - LSTM not available")
                    print("Please upgrade to Tesseract 4.x or 5.x for OEM mode 3 (LSTM) support")
                    return False

        print("\n✓ Tesseract is installed")
        return True

    except subprocess.TimeoutExpired:
        print("ERROR: Tesseract command timed out")
        return False
    except Exception as e:
        print(f"ERROR: Could not run tesseract: {e}")
        return False


def check_pytesseract():
    """Check pytesseract Python package."""
    try:
        import pytesseract
        print(f"\n✓ pytesseract package installed")
        try:
            version = pytesseract.get_tesseract_version()
            print(f"  Tesseract version via pytesseract: {version}")
        except Exception as e:
            print(f"  Warning: pytesseract could not get version: {e}")
        return True
    except ImportError:
        print("\n✗ pytesseract package not installed")
        print("  Install with: pip install pytesseract")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Tesseract OCR Verification")
    print("=" * 60)

    tesseract_ok = check_tesseract()
    pytesseract_ok = check_pytesseract()

    print("\n" + "=" * 60)
    if tesseract_ok and pytesseract_ok:
        print("✓ All checks passed - Tesseract is ready for OCR")
        sys.exit(0)
    else:
        print("✗ Some checks failed - please address the issues above")
        sys.exit(1)
