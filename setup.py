import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='fewerbytes',
    version='0.0.1',
    author='Brian Kopp',
    author_email='briankopp.usa@gmail.com',
    description='Compression techniques for numpy arrays',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/briankopp/fewerbytes',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy'
    ],
    classifier=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Ubuntu'
    ),
)
