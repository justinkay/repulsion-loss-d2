# repulsion-loss-detectron2

Implementation of ["Repulsion Loss: Detecting Pedestrians in a Crowd"](https://arxiv.org/abs/1711.07752) for Faster RCNN, on top of [Detectron2](https://github.com/facebookresearch/detectron2).

Note: this was built on version [0.1.1](https://github.com/facebookresearch/detectron2/releases/tag/v0.1.1) of Detectron2 and has not been tested on later versions.

## Usage

You should not need any additional libraries other than what is needed for Detectron2.

Installation with pip:

```
pip install -e .
```

Then, simply:

```
from detectron2.config import get_cfg

from repulsion_loss.config import add_reploss_config


cfg = get_cfg()
add_reploss_config(cfg)

cfg.merge_from_file("[path/to/repulsion_loss]/configs/Base-RCNN-FPN-RepLoss.yaml")
...
```

## References

[Xinlong Wang, et al. "Repulsion Loss: Detecting Pedestrians in a Crowd." CVPR2018.](https://arxiv.org/abs/1711.07752)

[Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick. Detectron2. 2019.](https://github.com/facebookresearch/detectron2)
