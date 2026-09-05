# Stage 6 Representative Inputs

Stage 6 retains the MobileNetV2 workload documented in `REPRESENTATIVE_WORKLOAD.md` and adds a second JPEG to reduce dependence on one decode pattern.

| File | Source | SHA-256 | Use |
|---|---|---|---|
| `assets/representative/cat_image.jpg` | Stage 5 Wikimedia Commons source | `D91F623700391ABCDC5B73544CF0C6DBEFFED4B925F8D9438AAD93183D3FA1E` | Primary disk/memory workload |
| `assets/representative/cat_image_2.jpg` | Wikimedia Commons `Picture_of_cat.jpg`; author `Theeyes 07`; CC0 (`https://commons.wikimedia.org/wiki/File:Picture_of_cat.jpg`) | `A4DFDC0C19852A77FBB1313A7B23B7571210FC496BECD0A93325BF3106FB5E78` | Cross-image validation |

The second image is a workload variation, not a second model or accuracy claim.
