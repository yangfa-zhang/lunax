#!/usr/bin/env python
#-*- coding:utf-8 -*-

from setuptools import setup, find_packages            #这个包没有的可以pip一下

setup(
    name = "lunax",      #这里是pip项目发布的名称
    version = "0.0.8",  #版本号，数值大的会优先被pip
    keywords = ["pip", "tabular data"],			# 关键字
    description = "A machine learning framework.",	# 描述
    long_description= "Lunax is a machine learning framework specifically designed for the processing and analysis of tabular data. The name Lunax is derived from the name of a beloved feline mascot Luna at South China University of Technology ",
    license = "MIT Licence",		# 许可证

    url = "https://github.com/yangfa-zhang/lunax",     #项目相关文件地址，一般是github项目地址即可
    author = "yangfa-zhang",			# 作者
    author_email = "yangfa1027@gmail.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy", "pandas","xgboost","tabulate","python-abc","typing","scikit-learn","optuna","lightgbm","catboost","matplotlib","seaborn","tabpfn"]          #这个项目依赖的第三方库
)
