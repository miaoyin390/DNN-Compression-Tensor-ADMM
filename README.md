### 🚫 Please keep confidential!
### 🚫 DO NOT share without Professor's permission!

#### > Example command:

`python main.py --lr 0.1 --sched step --decay-epochs 55 --epochs 200 --gpus 2 --model resnet32 --mixup 0 --cutmix 0 --smoothing 0.1 --batch-size 128 --decay-rate 0.2`

#### > Experimental results:

| Model                     | Top-1 (%) | Top-5 (%) | Ratio |                                                  Configuration                                           |
|---------------------------|:---------:|-----------|:-----:|:--------------------------------------------------------------------------------------------------------:|
| ResNet32-Baseline         | 92.49     | N/A       | N/A   | N/A                                                                                                      |
| ResNet32-ADMM-TK          |   93.44   | N/A       |   3x  | lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1   |
| ResNet32-FT-TKR           |   93.29   | N/A       |   3x  | lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1 |
| ResNet32-ADMM-TT          |   93.07   | N/A       |   3x  | lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1   |
| ResNet32-FT-TTR           |   93.05   | N/A       |   3x  | lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1 |
| ResNet18-Baseline         | 69.76     | 89.08     | N/A   | Torchvision                                                                                              |
| ResNet18-ADMM-TK         | 69.46     | 89.13     | 2.65x | lr=0.01, epochs=140, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=30                 |
| ResNet18-ADMM-TK         | 69.88     | 89.43     | 2.65x | lr=0.01, epochs=140, optimizer=momentum, scheduler=cosine                                                |
| ResNet18-FT-TKR           | 69.81     | 89.34     | 2.65x | lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=30                |
| ResNet18-ADMM-TT-general | 69.26     | 88.94     | 2.68x | lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                 |
| ResNet18-FT-TTR-general   | 69.65     | 89.19     | 2.68x | lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30                |
| ResNet18-ADMM-TT-special | 69.60     | 89.16     | 2.65x | lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                 |
| ResNet18-FT-TTR-special   | 69.82     | 89.18     | 2.65x | lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30                |
| ResNet50-Baseline         | 76.13     | 92.86     | N/A   | Torchvision                                                                                              |
| ResNet50-ADMM-TT-general | 76.98  |   93.34    | 3.05x | lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                 |
| ResNet50-FT-TTR-general   | 76.44 |  93.18 | 3.05x | lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30                         |
| ResNet50-ADMM-TTR-special |     |       | 3.05x | lr=0.01, epochs=150, optimizer=momentum, scheduler=cosine                 |
| ResNet50-FT-TTR-special   |      |       | 3.05x | lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30                    |
| DeiT-tiny-ADMM-TT |     |       | 35\% | lr=0.01, epochs=200, optimizer=momentum, scheduler=cosine                 |
| DeiT-tiny-FT-TT |     |       | 35\% | lr=0.001, epochs=100, optimizer=momentum, scheduler=cosine                 |
