import subprocess
import sys

script = "phoneme-cmd.py"
output_dir = "dist"

cmd = [
    sys.executable,
    "-m",
    "nuitka",
    "--mode=standalone",
    "--output-dir=" + output_dir,
    "--remove-output",
    script,
]

subprocess.run(cmd, check=True)
print(f"Build complete")
