from setuptools import setup

package_name = "vps_ros2"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="UAV Engineer",
    maintainer_email="aadarsh@uav.local",
    description="VPS ROS 2 Node Wrapper",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "vps_node = vps_ros2.vps_node:main",
        ],
    },
)
