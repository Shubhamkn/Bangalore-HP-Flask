import subprocess
import sys

def install_dependencies():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")

if __name__ == '__main__':
    install_dependencies()
