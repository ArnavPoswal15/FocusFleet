import subprocess
import sys
import pathlib

def run_pyinstaller():
    """Run PyInstaller to bundle the application into a single executable.
    Adjust hidden imports and data files as needed.
    """
    # Resolve paths relative to this script
    base_path = pathlib.Path(__file__).parent
    spec_file = base_path / "app.spec"
    # If a spec file does not exist, we generate a basic one on the fly
    if not spec_file.exists():
        # Basic command without a spec file
        cmd = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--windowed",
            "--add-data",
            f"{base_path / 'model.h5'}{pathlib.Path(':') if sys.platform.startswith('win') else ':'}{'.'}",
            "app.py",
        ]
    else:
        cmd = [sys.executable, "-m", "PyInstaller", str(spec_file)]
    try:
        subprocess.run(cmd, check=True)
        print("PyInstaller bundling completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller failed: {e}")

if __name__ == "__main__":
    run_pyinstaller()
