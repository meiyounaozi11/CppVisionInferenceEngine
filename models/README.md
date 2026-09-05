# Models

Model binaries are intentionally kept out of Git. The representative workload
uses `mobilenetv2-7.onnx` from the ONNX Model Zoo; its source, SHA-256,
preprocessing contract, and attribution are recorded in
`docs/REPRESENTATIVE_WORKLOAD.md` and `THIRD_PARTY_NOTICES.md`.

From a repository checkout, acquire and verify the local model with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_stage5_assets.ps1
```

The script writes to `models/representative/`, which is ignored by Git, and
fails on a SHA-256 mismatch.
