"""Generate the tiny deterministic model used by InferenceEngineTests."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [1, 3, 2, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "cppvision_identity",
        [helper.make_value_info("input", tensor_type)],
        [helper.make_value_info("output", tensor_type)],
    )
    model = helper.make_model(
        graph,
        producer_name="CppVisionInferenceEngine",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    output = Path(__file__).resolve().parents[1] / "tests" / "models" / "identity_nchw.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    print(output)


if __name__ == "__main__":
    main()
