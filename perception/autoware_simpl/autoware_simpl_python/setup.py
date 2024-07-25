from glob import glob
import os.path as osp

from setuptools import find_packages
from setuptools import setup

package_name = "autoware_simpl_python"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (osp.join("share", package_name), ["package.xml"]),
        (osp.join("share", package_name, "config"), glob("config/*")),
        (osp.join("share", package_name, "launch"), glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ktro2828",
    maintainer_email="kotaro.uetake@tier4.jp",
    description="A ROS package of SIMPL written in Python.",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"{package_name}_node = scripts.simpl_node:main",
        ],
    },
)
