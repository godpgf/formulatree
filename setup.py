#coding=utf-8
#author=godpgf

from setuptools import setup, find_packages

from pip.req import parse_requirements


setup(
    name='formulatree',
    version='0.0.1',
    description='formula for python',
    packages=find_packages(exclude=[]),
    author='godpgf',
    author_email='godpgf@qq.com',
    package_data={'': ['*.*']},
    url='https://github.com/godpgf',
    install_requires=[str(ir.req) for ir in parse_requirements("requirements.txt", session=False)],
    zip_safe=False,
    #entry_points={
    #    "console_scripts": [
    #        "rqalpha = rqalpha.__main__:entry_point",
    #    ]
    #},
)