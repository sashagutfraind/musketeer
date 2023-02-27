#future: this package is not currently available from pypi
import setuptools

with open("README.md", "r") as fh:
	description = fh.read()

setuptools.setup(
	name="musketeer",
	version="1.3",
	author="Alexander Gutfraind and Ilya Safro",
	author_email="",
	packages=["musketeer"],
	description="Multiscale Entropic Network Generator",
	long_description=description,
	long_description_content_type="text/markdown",
	url="https://github.com/sashagutfraind/musketeer/",
	license='GNU',
	python_requires='>=3.8',
	install_requires=[]
)
