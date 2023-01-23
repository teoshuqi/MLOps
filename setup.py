# setup.py
from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# setup.py
setup(
    name="SuperstoreMarketingCampaign",
    version=0.1,
    description="Classify customers that will likely to suscribe to new campaign",
    author="ShuQi",
    author_email="19shuqi@gmail.com",
    url="https://madewithml.com/",
    python_requires=">=3.7",
    install_requires=[required_packages],
)
