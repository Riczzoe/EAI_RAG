1. 处理原始 ImageNet 数据集：每类采样一张，目录使用纯 `wnid`

    - Add `src/utils/io.py`.
    - Add `configs/dataset.yaml` to define dataset paths, sampling settings, and runtime flags.
    - Add `src/data/split_kb.py` to implement per-class sampling for KB candidates.
    - Write outputs to `data/interim/sampled_kb_images` using `wnid` directory naming.
    - Add `scripts/prepare_imagenet.py` as a entrypoints.
