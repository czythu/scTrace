# Copyright (C) 2024 Zeyu Chen, G-Lab, Tsinghua University

from setuptools import setup, find_packages
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

with open("README.md", "r") as fh: 
	description = fh.read() 


setup( 
	name="scTrace", 
	version="0.1.6",
	author="Zeyu Chen", 
	author_email="chenzy22@mails.tsinghua.edu.cn", 
    packages=find_packages(),
	description="scTrace+: enhance the cell fate inference by integrating the lineage-tracing and multi-faceted transcriptomic similarity information", 
	long_description=description, 
	long_description_content_type="text/markdown", 
	url="https://github.com/czythu/scTrace", 
	license='MIT', 
	python_requires='>=3.7', 
	install_requires=[
		"numpy",
    	"pandas",
    	"scipy",
    	"scikit-learn",
    	"seaborn",
    	"matplotlib",
    	"scanpy",
    	"leidenalg",
    	"pyro-ppl",
    	"POT",
		"node2vec",
		"scStateDynamics"
	]
) 
