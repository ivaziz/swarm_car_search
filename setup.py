from setuptools import setup, find_packages

setup(
    name="swarm_car_search",
    version="1.0.0",
    description="Autonomous Multi-Robot Swarm for Missing Vehicle Search & Identification in Outdoor Urban Environments",
    author="Mohammed & Ibrahim (Swarm Coordination Subsystem)",
    author_email="swarm-robotics@graduation-project.local",
    license="MIT",
    packages=find_packages(exclude=["tests*", "output*"]),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "viz": [
            "matplotlib>=3.7.0",
            "numpy>=1.24.0",
            "pillow>=10.0.0",
        ],
        "yaml": [
            "pyyaml>=6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "swarm-sim=scripts.run_simulation:main",
            "swarm-eval=scripts.run_evaluation:main",
            "swarm=scripts.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
