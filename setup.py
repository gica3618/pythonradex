from setuptools import setup,find_packages

def readme():
    with open('README.txt') as f:
        return f.read()

setup(name='pythonradex',
      version='0.1',
      description='python implementation of RADEX',
      long_description=readme(),
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Astronomy',
                   'Intended Audience :: Science/Research'],
      keywords=['RADEX','radiative transfer'],
      url='https://github.com/gica3618/pythonradex',
      author='Gianni Cataldi',
      author_email='cataldi.gia@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['scipy','numpy','numba'],
      include_package_data=True,
      zip_safe=False,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      python_requires='>=3')
