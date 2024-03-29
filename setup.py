from setuptools import setup, find_packages

setup(
    name='BraggPy',
    version='0.1.1',
    description='caluculate Bragg spot intensity from single crystals',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Surpris/BraggPy',
    author='Surpris',
    author_email='take90-it09-easy27@outlook.jp',
    license='Apache-2.0 License',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)