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
	install_requires=["Pillow>=9.4.0,<10.0.0", "PyYAML>=6.0,<7.0", 
						"matplotlib>=3.7.0,<3.8.0", "networkx>=3.0,<3.1", 
						"numpy>=1.24.2,<1.25.0", "scipy>=1.10.0,<1.12.0"
	],
	extra_require={
		'dev': ["pytest>=7.2.1", "pytest-cov"]
	},
	classifiers=[
          'Intended Audience :: Scientists',
          'License :: OSI Approved :: GNU',
          'Programming Language :: Python',
          'Topic :: Network Analysis'
          ]
)
