

# CGRNet

Pytorch Code of CGRNet method for "Efficient Cross-modality Graph Reasoning for RGB-Infrared Person Re-identification" in 
[PDF](https://ieeexplore.ieee.org/abstract/document/9468909)

### Results on the SYSU-MM01 Dataset an the RegDB Dataset 
| Method | Datasets                   | Rank@1    | Rank@10  | mAP       |
| ------ | -------------------------- | --------- | -------- | --------- |
| CGRNet | #SYSU-MM01 (All-Search)    | ~ 52.64 % | ~ 85.25% | ~ 51.37 % |
| CGRNet | #SYSU-MM01 (Indoor-Search) | ~ 60.33 % | ~ 91.26% | ~ 66.75%  |
| CGRNet | #RegDB                     | ~ 75.58 % | ~ 88.4%  | ~67.86%   |



*The code has been tested in Python 3.7, PyTorch=1.0. Both of these two datasets may have some fluctuation due to random spliting

### 1. Prepare the datasets.

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py`  in to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Training.
  Train a model by
  ```bash
python train.version3.py
  ```

  - `--dataset`: which dataset "sysu" or "regdb".

  - `--lr`: initial learning rate.
  
  - `--gpu`:  which gpu to run.

You may need manually define the data path first.

### 3. Citation

Please kindly cite the references in your publications if it helps your research:
```
@article{jian2021efficient,
  title={Efficient Cross-modality Graph Reasoning for RGB-Infrared Person Re-identification},
  author={Jian, Feng Yu and Chen, Feng and Ji, Yi-mu and Wu, Fei and Sun, Jing},
  journal={IEEE Signal Processing Letters},
  year={2021},
  publisher={IEEE}
}
```

### 4. References



```
[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.
```

```
[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
```

```
[3] M. Ye, Z. Wang, X. Lan, and P. C. Yuen. Visible thermal person reidentification via dual-constrained top-ranking. In International Joint Conference on Artificial Intelligence (IJCAI), pages 1092–1099, 2018.
```

