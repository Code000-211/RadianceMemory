# Vocabulary-weighted lm_head classes

The width classes of Qwen3.8-27B's vocabulary-weighted lm_head, used by the IQR-V4 recipes
(`data/recipes/q38-27b-*-v4.recipe`) from IQ2R-V4 to IQ6R-V4. `tools/vhead.py` turns the bf16
checkpoint and one of these files into the checkpoint those recipes convert:

| file | mean width | widths (rows) |
|---|---|---|
| `q38-27b-vhead-4.0.cls` | 4.0 b | 3 (130560), 4 (31744), 5 (41472), 6 (44544) |
| `q38-27b-vhead-4.5.cls` | 4.5 b | 4 (171008), 5 (43008), 6 (22016), 7 (12288) |
| `q38-27b-vhead-5.0.cls` | 5.0 b | 4 (130560), 5 (33280), 6 (38400), 7 (46080) |

Each 256-row run of the vocabulary was given the width that minimises its rows' output error
(dW H dW^T against the head's input Hessian) weighted by their tokens' mean probability on
calibration text, at the file's mean width. The format is in `tools/vhead.py`.
