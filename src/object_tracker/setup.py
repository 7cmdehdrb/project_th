from setuptools import find_packages, setup
import os
import sys
from glob import glob

package_name = "object_tracker"

resource_path = os.path.join(os.path.dirname(__file__), "resource")
resource_file_names = os.listdir(resource_path)

resource_valid_extensions = (".json", ".yaml", ".txt", ".pt", ".pth")
filtered_resource_files = [
    f for f in resource_file_names if f.endswith(resource_valid_extensions)
]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name]
            + [f"resource/{file}" for file in filtered_resource_files],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="min",
    maintainer_email="7cmdehdrb@naver.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "real_time_segmentation_node = object_tracker.real_time_segmentation_node:main",
            "object_pose_estimation_server = object_tracker.object_pose_estimation_server:main",
            "pointcloud_pose_estimation_server = object_tracker.pointcloud_pose_estimation_server:main",
            "closest_object_classifier = object_tracker.closest_object_classifier:main",
        ],
    },
)
