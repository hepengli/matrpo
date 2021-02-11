from setuptools import setup, find_packages

setup(name='matrpo',
      version='0.0.1',
      description='Multi-Agent Trust Region Policy Optimization',
      url='https://github.com/hepengli/matrpo',
      author='Hepeng Li and Haibo He',
      author_email='hepengli@uri.edu',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'tensorflow', 'numpy-stl']
)
