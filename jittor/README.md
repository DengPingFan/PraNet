# PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI2020-Oral)

## Introduction

The repo provides inference code of **PraNet (MICCAI-2020)** with [Jittor deep-learning framework](https://github.com/Jittor/jittor).

> **Jittor** is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model. Jittor also contains a wealth of high-performance model libraries, including: image recognition, detection, segmentation, generation, differentiable rendering, geometric learning, reinforcement learning, etc. The front-end language is Python. Module Design and Dynamic Graph Execution is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA, C++.

## Usage

PraNet is also implemented in the Jittor toolbox which can be found in `./jittor`.
+ Create environment by `python3.7 -m pip install jittor` on Linux. 
As for MacOS or Windows users, using Docker `docker run --name jittor -v $PATH_TO_PROJECT:/home/PraNet -it jittor/jittor /bin/bash` 
is easier and necessary. 
A simple way to debug and run the script is running a new command in the container through `docker exec -it jittor /bin/bash` and start the experiments. (More details refer to this [installation tutorial](https://github.com/Jittor/jittor#install))

+ First, run `sudo sysctl vm.overcommit_memory=1` to set the memory allocation policy.

+ Second, switch to the project root by `cd /home/PraNet`

+ For testing, run `python3.7 jittor/MyTest.py`. 

> Note that the Jittor model is just converted from the original PyTorch model via toolbox, and thus, the trained weights of PyTorch model can be used to the inference of Jittor model.

## Performance Comparison

The performance has slight difference due to the different operator implemented between two frameworks.  The download link ([Pytorch](https://drive.google.com/file/d/1tW0OOxPSuhfSbMijaMPwRDPElW1qQywz/view?usp=sharing) / [Jittor](https://drive.google.com/file/d/1qpzNTWLAhepCT0OGNdjUIk-SVMCGUEdf/view?usp=sharing)) of prediction results on four testing dataset, including Kvasir, CVC-612, CVC-ColonDB, ETIS, and CVC-T.

| Kvasir dataset      | mean Dice | mean IoU | $F_\beta^w$ | $S_\alpha$ | $E_\phi^max$ | M     |
|---------------------|-----------|----------|-------------|------------|--------------|-------|
| PyTorch             | 0.898     | 0.840    | 0.885       | 0.915      | 0.948        | 0.030 |
| Jittor              | 0.895     | 0.836    | 0.880       | 0.913      | 0.945        | 0.030 |

| CVC-612 dataset     | mean Dice | mean IoU | $F_\beta^w$ | $S_\alpha$ | $E_\phi^max$ | M     |
|---------------------|-----------|----------|-------------|------------|--------------|-------|
| PyTorch             | 0.899     | 0.849    | 0.896       | 0.936      | 0.979        | 0.009 |
| Jittor              | 0.900     | 0.850    | 0.897       | 0.937      | 0.978        | 0.009 |

| CVC-ColonDB dataset | mean Dice | mean IoU | $F_\beta^w$ | $S_\alpha$ | $E_\phi^max$ | M     |
|---------------------|-----------|----------|-------------|------------|--------------|-------|
| PyTorch             | 0.709     | 0.640    | 0.696       | 0.819      | 0.869        | 0.045 |
| Jittor              | 0.708     | 0.637    | 0.695       | 0.817      | 0.869        | 0.044 |

| ETIS dataset        | mean Dice | mean IoU | $F_\beta^w$ | $S_\alpha$ | $E_\phi^max$ | M     |
|---------------------|-----------|----------|-------------|------------|--------------|-------|
| PyTorch             | 0.628     | 0.567    | 0.600       | 0.794      | 0.841        | 0.031 |
| Jittor              | 0.627     | 0.565    | 0.600       | 0.793      | 0.845        | 0.032 |

| CVC-T dataset       | mean Dice | mean IoU | $F_\beta^w$ | $S_\alpha$ | $E_\phi^max$ | M     |
|---------------------|-----------|----------|-------------|------------|--------------|-------|
| PyTorch             | 0.871     | 0.797    | 0.843       | 0.925      | 0.972        | 0.010 |
| Jittor              | 0.870     | 0.796    | 0.842       | 0.925      | 0.973        | 0.010 |

## Speedup

The jittor-based code can speed up the inference efficiency.

| Batch Size  	|     PyTorch    	|     Jittor     	|     Speedup    	|
|-----------	|----------------	|----------------	|----------------	|
|     1     	|     52 FPS     	|     67 FPS     	|     1.29x       	|
|     4     	|     194 FPS    	|     255 FPS    	|     1.31x       	|
|     8     	|     391 FPS    	|     508 FPS    	|     1.30x      	|
|     16    	|     476 FPS    	|     593 FPS    	|     1.25x       	|

## Citation

If you find our work useful in your research, please consider citing:
    
    
    @article{fan2020pra,
        title={PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
        author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
        journal={MICCAI},
        year={2020}
    }

and the jittor framework:

    @article{hu2020jittor,
      title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
      author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
      journal={Science China Information Sciences},
      volume={63},
      number={222103},
      pages={1--21},
      year={2020}
    }


# Acknowledgements

Thanks to Liang Dun from Tsinghua University ([The Graphics and Geometric Computing Group](https://cg.cs.tsinghua.edu.cn/#people.htm)) for his help in the framework conversion process.
