# clone_data.py
import subprocess

def clone_submodule():
    subprocess.run("git clone git@gitlab.com:xyntopia/pydoxtools_test_data.git tests/data", shell=True, check=True)
