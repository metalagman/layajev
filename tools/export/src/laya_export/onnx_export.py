"""Deterministic CPU/FP32 ONNX export and structural validation."""

from __future__ import annotations

from pathlib import Path


INPUT_NAMES = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
OUTPUT_NAMES = ("logits", "act_logits")


def export_onnx(model, output: Path) -> None:
    """Export the complete model as opset-18 with constrained dynamic shapes."""

    import torch

    input_ids = torch.tensor(
        [[2, 10, 3, 4, 20, 4, 21, 3, 30, 31, 3]],
        dtype=torch.int64,
    )
    input_ids = input_ids.repeat(2, 1)
    attention_mask = torch.ones_like(input_ids)
    marker_pos = torch.tensor([[3, 5], [3, 5]], dtype=torch.int64)
    marker_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
    qtype = torch.tensor([0, 1], dtype=torch.int64)
    batch = torch.export.Dim("batch", min=1)
    sequence = torch.export.Dim("sequence", min=2)
    markers = torch.export.Dim("markers", min=1)
    dynamic_shapes = {
        "input_ids": {0: batch, 1: sequence},
        "attention_mask": {0: batch, 1: sequence},
        "marker_pos": {0: batch, 1: markers},
        "marker_mask": {0: batch, 1: markers},
        "qtype": {0: batch},
    }
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, marker_pos, marker_mask, qtype),
            str(output),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
            external_data=True,
            do_constant_folding=True,
        )
    canonicalize_onnx(output)


def canonicalize_onnx(path: Path) -> None:
    """Remove non-semantic exporter debug metadata and serialize deterministically."""

    import onnx

    model = onnx.load(str(path), load_external_data=False)

    def strip_metadata(message) -> None:
        if "metadata_props" in message.DESCRIPTOR.fields_by_name:
            message.ClearField("metadata_props")
        for field, value in message.ListFields():
            if field.message_type is None:
                continue
            if field.is_repeated:
                for child in value:
                    strip_metadata(child)
            else:
                strip_metadata(value)

    strip_metadata(model)
    path.write_bytes(model.SerializeToString(deterministic=True))


def validate_onnx(path: Path) -> dict[str, object]:
    """Run ONNX checker and enforce the frozen graph interface."""

    import onnx

    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    inputs = [value.name for value in model.graph.input]
    outputs = [value.name for value in model.graph.output]
    if inputs != list(INPUT_NAMES) or outputs != list(OUTPUT_NAMES):
        raise ValueError(f"ONNX interface mismatch: inputs={inputs}, outputs={outputs}")
    if not model.opset_import or model.opset_import[0].version != 18:
        raise ValueError("ONNX graph is not opset 18")
    expected_types = {
        "input_ids": onnx.TensorProto.INT64,
        "attention_mask": onnx.TensorProto.INT64,
        "marker_pos": onnx.TensorProto.INT64,
        "marker_mask": onnx.TensorProto.BOOL,
        "qtype": onnx.TensorProto.INT64,
        "logits": onnx.TensorProto.FLOAT,
        "act_logits": onnx.TensorProto.FLOAT,
    }
    values = {value.name: value for value in (*model.graph.input, *model.graph.output)}
    for name, expected_type in expected_types.items():
        if values[name].type.tensor_type.elem_type != expected_type:
            raise ValueError(f"ONNX dtype mismatch: {name}")
    shapes = {
        name: [
            dimension.dim_param if dimension.dim_param else dimension.dim_value
            for dimension in value.type.tensor_type.shape.dim
        ]
        for name, value in values.items()
    }
    expected_shapes = {
        "input_ids": ["batch", "sequence"],
        "attention_mask": ["batch", "sequence"],
        "marker_pos": ["batch", "markers"],
        "marker_mask": ["batch", "markers"],
        "qtype": ["batch"],
        "logits": ["batch", "markers"],
        "act_logits": ["batch", 2],
    }
    if shapes != expected_shapes:
        raise ValueError(f"ONNX shape contract mismatch: {shapes}")
    return {
        "inputs": inputs,
        "outputs": outputs,
        "opset": model.opset_import[0].version,
        "nodes": len(model.graph.node),
        "shapes": shapes,
    }
