from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['gcsfs==0.7.1']

if __name__ == '__main__':
    setup(
        name = 'trainer',
        install_requires = REQUIRED_PACKAGES,
        packages = find_packages(include=['trainer'])
    )