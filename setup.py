from setuptools import setup

setup(name='geomle',
      version='1.0',
      description='Intrinsic dimension',
      url='https://github.com/premolab/GeoMLE',
      author='Mokrov, Gomtsyan, Panov, Yanovich',
      author_email='panov.maxim@gmail.com ',
      license='MIT',
      packages=['geomle'],
      install_requires=[
          'numpy>=1.13.1',
          'scikit-learn>=0.18',
          'pandas>=0.19',
      ],
      zip_safe=False)