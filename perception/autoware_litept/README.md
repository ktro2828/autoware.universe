# autoware_litept

## Purpose

The `autoware_litept` package is used for 3D LiDAR semantic segmentation.

## Inner-workings / Algorithms

This package implements a TensorRT powered inference node for LitePT [1].

The sparse convolution backend corresponds to [spconv](https://github.com/traveller59/spconv).
Autoware installs it automatically in its setup script. If needed, the user can also build it and install it following the [following instructions](https://github.com/autowarefoundation/spconv_cpp).

## Inputs / Outputs

### Input

| Name                 | Type                            | Description       |
| -------------------- | ------------------------------- | ----------------- |
| `~/input/pointcloud` | `sensor_msgs::msg::PointCloud2` | Input pointcloud. |

### Output

| Name                  | Type                            | Description                  |
| --------------------- | ------------------------------- | ---------------------------- |
| `~/output/pointcloud` | `sensor_msgs::msg::PointCloud2` | Output segmented pointcloud. |

## Parameters

### The `build_only` option

The `autoware_litept` node has a `build_only` option to build the TensorRT engine file from the specified ONNX file, and terminates the application in the end.

```shell
ros2 launch autoware_litept litept.launch.xml build_only:=true
```

## Assumptions / Known Limits

This node assumes that the input pointcloud follows the `PointXYZIRC` layout defined in `autoware_point_types`.

## References / External Links

[1] Yuanwen Yue and Damien Robert and Jianyuan Wang and Sunghwan Hong and Jan Dirk Wegner and Christian Rupprecht and Konrad Schindler. "LitePT: Lighter Yet Stronger Point Transformer" 2025 arXiv preprint arXiv:2512.13689. <!-- cspell:disable-line -->
