### 🚫 Please keep confidential!
### 🚫 DO NOT share without Professor's permission!

#### > Example command:

`python main.py --lr 0.1 --sched step --decay-epochs 55 --epochs 200 --gpus 2 --model resnet32 --mixup 0 --cutmix 0 --smoothing 0.1 --batch-size 128 --decay-rate 0.2`

#### > Experimental results:

| Model                  | Top-1 (%) | Top-5 (%) | Ratio |                                                     Configuration                                                     |
|------------------------|:---------:|--|:-----:|:---------------------------------------------------------------------------------------------------------------------:|
| ResNet32-Baseline      |   92.49   | N/A |  N/A  |                                                          N/A                                                          |
| ResNet32-TK-ADMM       |   93.44   | N/A |  3x   |        lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1         |
| ResNet32-TK-FT         |   93.29   | N/A |  3x   |       lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1        |
| ResNet32-TT-ADMM       |   93.07   | N/A |  3x   |        lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1         |
| ResNet32-TT-FT         |   93.05   | N/A |  3x   |       lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1        |
| ResNet18-Baseline      |   69.76   | 89.08 |  N/A  |                                                      Torchvision                                                      |
| ResNet18-TK-ADMM       |   69.46   | 89.13 | 2.65x |               lr=0.01, epochs=140, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=30                |
| ResNet18-TK-ADMM       |   69.88   | 89.43 | 2.65x |                               lr=0.01, epochs=140, optimizer=momentum, scheduler=cosine                               |
| ResNet18-TK-FT         |   69.81   | 89.34 | 2.65x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=30               |
| ResNet18-TT-general-ADMM |   69.26   | 88.94 | 2.68x |               lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                |
| ResNet18-TT-general-FT |   69.65   | 89.19 | 2.68x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30               |
| ResNet18-TT-special-ADMM |   69.60   | 89.16 | 2.65x |               lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                |
| ResNet18-TT-special-FT |   69.82   | 89.18 | 2.65x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30               |
| ResNet50-Baseline      |   76.13   | 92.86 |  N/A  |                                                      Torchvision                                                      |
| ResNet50-TT-general-ADMM |   76.98   | 93.34 | 3.05x |               lr=0.01, epochs=150, optimizer=momentum, scheduler=step, decay_rate=0.1, decay_epochs=40                |
| ResNet50-TT-general-FT |   76.44   | 93.18 | 3.05x |               lr=0.001, epochs=105, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=30               |
| DeitT-tiny-baseline    |   72.20   |  |       |                                                                                                                       |
| DeiT-tiny-TT-ADMM      |   71.68   | 90.28 | 30\%  |                    lr=0.01, epochs=200, optimizer=momentum, scheduler=cosine, distillation='hard'                     |
| DeiT-tiny-TT-FT        |   72.61   | 90.86 | 30\%  |                    lr=0.001, epochs=100, optimizer=momentum, scheduler=cosine, distillation='hard'                    |
| DeitT-small-baseline   |   79.90   |  |       |                                                                                                                       |
| Deit-small-TT-ADMM     |   78.81   | 94.30 |       | lr=0.01, epochs=400, optimizer=momentum, scheduler=cosine, distillation='hard', mixup=0.3, cutmix=1.0, smoothing=0.1  |
| Deit-small-TT-FT       |       |      | 40\%  | lr=0.002, epochs=200, optimizer=momentum, scheduler=cosine, distillation='hard', mixup=0.3, cutmix=1.0, smoothing=0.1 |
| Mobilenetv2-baseline   |  71.80  |        |       |                                                                                                                         |
| Mobilenetv2-SVD-ADMM   |   70.03   | 89.17 |       |             lr=0.05, epochs=180, optimizer=momentum, scheduler=cosine, distillation='hard', smoothing=0.1             |
| Mobilenetv2-SVD-FT     | 69.94 | 89.16 | 30\%  |            lr=0.005, epochs=180, optimizer=momentum, scheduler=cosine, distillation='hard', smoothing=0.1                 |
