from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "Fire_smoke_detection"
AUTHOR_USER_NAME = "Dr.Anjit"
SRC_REPO = "fire_smoke_detection"
AUTHOR_EMAIL = "taanjit@yahoo.in"

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Fire and Smoke Detection using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "python-box>=7.0.0",
        "ensure>=1.0.2",
        "Flask>=2.3.0",
    ],
    python_requires=">=3.8",
)
