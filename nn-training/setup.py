from setuptools import setup, Command
from setuptools.command.build_py import build_py

import subprocess
import os
import sys

class CustomBuild(build_py):
    def run(self):
        subprocess.run([
            "unzip",
            "pgn-extract.zip"
        ], stdout=sys.stdout)

        os.chdir("pgn-extract")
        subprocess.run("make", stdout=sys.stdout)
        os.chdir("..")
        super().run()

setup(
    name='nn-training', 
    version='0.1',
    author='William Galvin', 
    install_requires=[ 
        "atlas-chess",
        "numpy",
        "h5py",
        "hdf5plugin",
        "Pillow",
        "pytest",
        "sqlitedict",
        "torch",
    ], 
    cmdclass={"build_py": CustomBuild},
)