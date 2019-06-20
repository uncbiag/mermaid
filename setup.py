from setuptools import setup, find_packages

setup(
    name='mermaid',
    version='0.2.0',
    description='Image registration toolbox',
    long_description='Image registration toolbox based on pyTorch to allow for rapid prototyping of image registration algorithms',
    author='Marc Niethammer',
    author_email='mn@cs.unc.edu',
    packages=['mermaid'],  #same as name
    url='',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[] #["numpy>=1.10"],
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python']    
)
