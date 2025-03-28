from setuptools import setup, find_namespace_packages

setup(
    name="fanuc-ml",
    version="0.1.0",
    description="Machine Learning for FANUC Robot Arm Positioning",
    author="FANUC ML Team",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/fanuc-ml",
    package_dir={"": "src", "fanuc_tools": "tools"},
    packages=find_namespace_packages(where="src") + ["fanuc_tools"],
    py_modules=["main"],
    entry_points={
        "console_scripts": [
            "fanuc-train=main:main",
            "fanuc-eval=tools.run_eval:main",
            "fanuc-test=tools.test_install:main",
        ],
    },
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "pybullet>=3.2.1",
        "gymnasium>=0.28.1",
        "stable-baselines3>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "imageio>=2.22.0",
        "pillow>=9.0.0",
    ],
    extras_require={
        "directml": ["torch-directml>=0.2.0.dev230426"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
) 