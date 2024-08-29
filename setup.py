import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

class CustomInstallCommand(install):
    def run(self):
        # Run the standard install
        install.run(self)
        # Custom post-install script
        self.execute_post_install_script()

    def execute_post_install_script(self):
        script = """
        wget https://github.com/xg-chu/lightning_track/releases/download/resources/resources.tar -O ./resources.tar
        tar -xvf resources.tar
        mv resources/emoca/* ./flame_feature_extractor/feature_extractor/emoca/assets/
        mv resources/FLAME/* ./flame_feature_extractor/renderer/assets/
        mv resources/mica/* ./flame_feature_extractor/feature_extractor/mica/assets/
        rm -r resources/
        """
        subprocess.check_call(script, shell=True)

setup(
    name="flame_feature_extractor",
    version="0.0.1",
    description="Flame Feature Extraction",
    author="Nabarun Goswami",
    author_email="nabarungoswami@mi.t.u-tokyo.ac.jp",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "mediapipe",
        "scikit-image",
        "onnx",
        "onnxruntime",
    ],
    cmdclass={
        'install': CustomInstallCommand,
    },
)
