### 🚫 Please keep confidential!
### 🚫 DO NOT share without Professor's permission!

#### > Example command:

`python main.py --lr 0.1 --sched step --decay-epochs 55 --epochs 200 --gpus 2 --model resnet32 --mixup 0 --cutmix 0 --smoothing 0.1 --batch-size 128 --decay-rate 0.2`

#### > Experimental results:

| Model            | Top-1 (%) | Ratio |                                                  Config                                                  |
|------------------|:---------:|:-----:|:--------------------------------------------------------------------------------------------------------:|
| ResNet32-ADMM-TK |   93.44   |   3x  | lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1   |
| ResNet32-FT-TKR  |   93.29   |   3x  | lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1 |
| ResNet32-ADMM-TT |   93.07   |   3x  | lr=0.1, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.2, decay_epochs=55, smoothing=0.1   |
| ResNet32-FT-TTR  |   93.05   |   3x  | lr=0.005, epochs=200, optimizer=momentum, scheduler=step, decay_rate=0.4, decay_epochs=55, smoothing=0.1 |
