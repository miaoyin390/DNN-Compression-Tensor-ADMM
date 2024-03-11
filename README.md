This is the official implementation for [Towards Efficient Tensor Decomposition-Based DNN Model Compression with Optimization Framework (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Yin_Towards_Efficient_Tensor_Decomposition-Based_DNN_Model_Compression_With_Optimization_Framework_CVPR_2021_paper.pdf)


### > Example commands:

Train baseline: `python main.py --model resnet32 --dataset cifar10 --lr 0.1 --sched step --decay-epochs 55 --epochs 200 --gpus 2 --mixup 0 --cutmix 0 --smoothing 0.1 --batch-size 128 --decay-rate 0.2`

ADMM from local model: `python main.py --model resnet32 --dataset cifar10 --lr 0.1 --sched cosine --admm --format tk --model-path [pretrained_model_path] --epochs 200 --gpus 2 --smoothing 0.1 --batch-size 128`

ADMM from online model: `python main.py --model resnet32 --dataset cifar10 --lr 0.1 --sched cosine --admm --format tk --pretrained --epochs 200 --gpus 2 --smoothing 0.1 --batch-size 128`

Fine-tune from local ADMM model: `python main.py --model tkc_resnet32 --dataset cifar10 --lr 0.005 --sched cosine --decompose --model-path [admm_model_path] --epochs 200 --gpus 2 --smoothing 0.1 --batch-size 128`

### > Experimental results:
(P and F in Ratio column means parameters and FLOPs, respectively.)

| Model                  | Top-1 (%) | Top-5 (%) | Ratio |                                                     Configuration                                                     |
|------------------------|:---------:|--|:-----:|:---------------------------------------------------------------------------------------------------------------------:|
| ResNet32-Baseline      |   92.49   | N/A |  N/A  |                   weight-decay=5e-4                                                         |
| ResNet32-TK-ADMM       |   93.44   | N/A |  3x   |        lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1         |
| ResNet32-TK-FT         |   93.29   | N/A |  3x   |       lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1        |
| ResNet32-TT-ADMM       |   93.07   | N/A |  3x   |        lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1         |
| ResNet32-TT-FT         |   93.05   | N/A |  3x   |       lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1        |
| ResNet18-Baseline      |   69.76   | 89.08 |  N/A  |                                                      Torchvision                                                      |
| ResNet18-TK-ADMM       |   69.46   | 89.13 | 2.65x |               lr=0.01, epochs=140, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=30                |
| ResNet18-TK-ADMM       |   69.88   | 89.43 | 2.65x |                               lr=0.01, epochs=140, optimizer=momentum, scheduler=cosine                               |
| ResNet18-TK-FT         |   69.81   | 89.34 | P3x, F2.65x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=30               |
| ResNet18-TT-general-ADMM |   69.26   | 88.94 | 2.68x |               lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                |
| ResNet18-TT-general-FT |   69.65   | 89.19 | 2.68x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30               |
| ResNet18-TT-special-ADMM |   69.60   | 89.16 | 2.65x |               lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                |
| ResNet18-TT-special-FT |   69.82   | 89.18 | 2.65x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30               |
| ResNet50-Baseline      |   76.13   | 92.86 |  N/A  |                                                      Torchvision                                                      |
| ResNet50-TT-general-ADMM |   76.98   | 93.34 | 3.05x |               lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                |
| ResNet50-TT-general-FT |   76.44   | 93.18 | 3.05x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30               |
| ResNet50-TK |   76.44   | 92.94 | P3x, F2.7x |               --lr 0.008 --epochs=180 --opt momentum --scheduler cosine --pretrained --decompose --distillation 'hard' --teacher-model resnet50                |
| DeiT-tiny-baseline    |   72.20   |  |       |                                                                                                                       |
| DeiT-tiny-TT-ADMM      |   71.68   | 90.28 | 30\%  |                    lr=0.01, epochs=200, optimizer=momentum, scheduler=cosine, distillation='hard'                     |
| DeiT-tiny-TT-FT        |   72.61   | 90.86 | 30\%  |                    lr=0.001, epochs=100, optimizer=momentum, scheduler=cosine, distillation='hard'                    |
| DeiT-small-baseline   |   79.90   |  |       |                                                                                                                       |
| DeiT-small-TT-ADMM     |   78.81   | 94.30 | 35\%  | lr=0.01, epochs=400, optimizer=momentum, scheduler=cosine, distillation='hard', mixup=0.3, cutmix=1.0, smoothing=0.1  |
| DeiT-small-TT-FT       |   79.07   | 94.53 | 35\%  | lr=0.002, epochs=200, optimizer=momentum, scheduler=cosine, distillation='hard', mixup=0.3, cutmix=1.0, smoothing=0.1 |
| Mobilenetv2-baseline   |  71.80  |        |       |                                                                                                                         |
| Mobilenetv2-SVD-ADMM   |   70.03   | 89.17 |       |             lr=0.05, epochs=180, optimizer=momentum, scheduler=cosine, distillation='hard', smoothing=0.1             |
| Mobilenetv2-SVD-FT     | 69.94 | 89.16 | 30\%  |            lr=0.005, epochs=180, optimizer=momentum, scheduler=cosine, distillation='hard', smoothing=0.1                 |

#### > Cite
```
@inproceedings{yin2021towards,
  title={Towards efficient tensor decomposition-based dnn model compression with optimization framework},
  author={Yin, Miao and Sui, Yang and Liao, Siyu and Yuan, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10674--10683},
  year={2021}
}
```
