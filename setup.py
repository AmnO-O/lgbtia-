from setuptools import setup, find_packages

setup(
    name="stereoqueer_pipeline",
    version="1.0.0",
    description="Python Training Pipeline for StereoQueerEval SemEval 2027 and Multi-task Hate Speech / Toxic Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "sq-train=train:main",
            "sq-eval=evaluate:main",
            "sq-predict=predict:main",
        ],
    },
)
