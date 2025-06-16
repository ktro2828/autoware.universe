# autoware_mtr

## Purpose

The `autoware_mtr` is used for 3D object motion prediction based on ML-based model called MTR.

## Inner-workings / Algorithms

The implementation bases on MTR [1] [2] work. It uses TensorRT library for data process and network interface.

### Inputs Representation

- $X_A\in R^{B\times N\times T_{past}\times D_{agent}}$: Agent histories input.
- $M_A\in R^{B\times N\times T_{past}}$: Agent histories mask.
- $X_M\in R^{B\times K\times P\times D_{map}}$: Map points input.
- $M_M\in R^{B\times K\times P}$: Map points mask.
- $C_A\in R^{B\times N\times 3}$: Agent current positions.
- $C_M\in R^{B\times K\times 3}$: Map center positions.
- $I_A\in R^{B}$: Target agent indices.
- $C_A\in R^{B}$: Target agent label ids.

### Outputs Representation

- $P_{score}\in R^{N\times M}$: Predicted scores for each agent and mode.
- $P_{trajectory}\in R^{N\times M\times T_{future}\times D_{trajectory}}$: Predicted trajectories for each agent and mode.

## Inputs / Outputs

### Inputs

| Name                            | Type                                            | Description           |
| ------------------------------- | ----------------------------------------------- | --------------------- |
| `~/input/objects`               | `autoware_perception_msgs::msg::TrackedObjects` | Input tracked agents. |
| `~/input/vector_map`            | `autoware_map_msgs::msg::LeneletMapBin`         | Input vector map.     |
| `/localization/kinematic_state` | `nav_msgs::msg::Odometry`                       | Ego vehicle odometry. |

### Outputs

| Name               | Type                                              | Description               |
| ------------------ | ------------------------------------------------- | ------------------------- |
| `~/output/objects` | `autoware_perception_msgs::msg::PredictedObjects` | Predicted agents' motion. |

## Parameters

{{ json_to_markdown("perception/autoware_mtr/schema/mtr.scheme.json") }}

## [WIP] Model Training / Deployment

Now we are preparing a library to train and deploy MTR and other ML models featuring motion prediction tasks.

## Testing

Unit tests are provided and can be run with:

```shell
colcon test --packages-select autoware_mtr
colcon test-result --all
```

To print the test's details with while the tests are being run, use the `--event-handlers console_cohesion+` option to print the details directly to the console:

```shell
colcon test --event-handlers console_cohesion+ --packages-select autoware_mtr
```

## Assumptions / Known limits

### Number of predicted agents

We have not supported the dynamic shape inference yet. Therefore, the number of predicted agents must be fixed as `preprocess.max_num_target` ($B$) and `preprocess.max_num_agent` ($N$).
This value is determined when exporting ONNX.

Note that the following parameters are also determined when exporting ONNX:

- `preprocess.num_past`: $T_{past}$
- `preprocess.max_num_polyline`: $K$
- `preprocess.max_num_point`: $P$
- `postprocess.num_mode`: $M$
- `postprocess.num_future`: $T_{future}$

### Agent History Lifetime

Under the hood, `MTRNode` stores and accumulates agent history in every callback, but removes the history that is not observed in callbacks.

## References / External Links

[1] S. Shi, L. Jiang, D. Dai, and B. Schiele, “Motion Transformer with Global Intention Localization and Local Movement Refinement,” arXiv preprint arXiv:2209.13508, 2023. <!-- cspell:disable-line -->

[2] <https://github.com/sshaoshuai/MTR>
