from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# with open("requirements.txt", "r") as f:
#     requirements = f.read().splitlines()
#     requirements = [l for l in requirements if not l.startswith('#')]

setup(
    name="DN_population_analysis",
    version="0.1",
    packages=["dn_experiments","dn_connectomics"],
    author="Jonas Braun, Femke Hurtak, Sibo Wang-Chen",
    author_email="jonas.braun@epfl.ch, femke.hurtak@epfl.ch, sibo.wang@epfl.ch",
    description="Code for preprocessing and analysis of the data published in Braun et al. 2023/2024",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeLy-EPFL/dn_interactions.git",
    python_requires='>=3.7, <3.10',
    # install_requires=requirements,
)
