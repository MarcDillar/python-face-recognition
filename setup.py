from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='simple-face-recognizer',
    version='0.1.0',
    description='Simple face recognition for teaching purposes',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Marc Dillar',
    author_email='marc.dillar@gmail.com',
    url='https://github.com/MarcDillar/python-face-recognition',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)
