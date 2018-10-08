from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

requirements = ['numpy', 'pandas']

setup(
    name='id3tree',
    description='This is used for creating multi-split entropy based ID3Trees.',
    long_description=long_description,
    url='https://github.com/ragabala/Id3trees',
    author='Ragavendran balakrishnan',
    version='0.0.4',
    packages=find_packages(),
    author_email='ragavendranb92@gmail.com',
    license='TEST',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Operating System :: POSIX :: Linux',
        'License :: Freeware',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.7'],
    extras_require={
        'dev': [
            'pylint',
        ],

    },

)
