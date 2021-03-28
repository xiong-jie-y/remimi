import setuptools

__version__ = '0.0.1dev1'

def _parse_requirements(path):
    with open(path) as f:
        return [
            line.rstrip()
            for line in f
            if not (line.isspace() or line.startswith('#'))
        ]

requirements = _parse_requirements('requirements.txt')

setuptools.setup(
    name='remimi',
    version=__version__,
    packages=setuptools.find_packages(),
    install_requires=requirements
)
