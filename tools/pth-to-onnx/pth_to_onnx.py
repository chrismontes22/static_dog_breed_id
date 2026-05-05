# export_onnx.py
import torch
import torchvision.models as models

# 1. Recreate your model architecture
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 73)  # 73 breeds
model.load_state_dict(torch.load("Dogrun2.pth", map_location="cpu"))
model.eval()  # Set to inference mode

# 2. Create dummy input matching training shape
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX (single file, server-ready)
torch.onnx.export(
    model, dummy_input, "dogmodel.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=14, do_constant_folding=True
)

print("✅ Successfully exported dogmodel.onnx (~44 MB)")
print("📁 Verify file exists: ls -lh dogmodel.onnx")