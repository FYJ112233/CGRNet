Paper:Efficient Cross-modality Graph Reasoning for RGB-Infrared Person Re-identification

Pytorch Code for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset and SYSU-MM01 dataset.

We adopt the two-stream network structure introduced in . ResNet50 is adopted as the backbone. The softmax loss is adopted as the baseline.

Datasets	Pretrained	Rank@1	mAP		Model
#RegDB	ImageNet	~ 75.58%~ 67.86%		-----
#SYSU-MM01	ImageNet	~ 52.64%	~ 51.37%		GoogleDrive
*Both of these two datasets may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters.

1. Prepare the datasets.
(1) RegDB Dataset : The RegDB dataset can be downloaded from this website by submitting a copyright form.

(Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website).

A private download link can be requested via sending me an email (mangye16@gmail.com).

(2) SYSU-MM01 Dataset : The SYSU-MM01 dataset can be downloaded from this website.

run python pre_process_sysu.py to pepare the dataset, the training data will be stored in ".npy" format.
2. Training.
Train a model by

python train.py --dataset sysu --lr 0.1 --method agw --gpu 1
--dataset: which dataset "sysu" or "regdb".

--lr: initial learning rate.

--method: method to run or baseline.

--gpu: which gpu to run.

You may need mannully define the data path first.

Parameters: More parameters can be found in the script.

Training Log: The training log will be saved in log/" dataset_name"+ log. Model will be saved in save_model/.

3. Testing.
Test a model on SYSU-MM01 or RegDB dataset by

python test.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
--dataset: which dataset "sysu" or "regdb".

--mode: "all" or "indoor" all search or indoor search (only for sysu dataset).

--trial: testing trial (only for RegDB dataset).

--resume: the saved model path.

--gpu: which gpu to run.



This code largely benefits from following repositories: https://github.com/mangye16/Cross-Modal-Re-ID-baseline and https://github.com/zhanghang1989/PyTorch-Encoding .I am very grateful to the author (@mangye16) for his contribution and help.

References

[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[3] M. Ye, Z. Wang, X. Lan, and P. C. Yuen. Visible thermal person reidentification via dual-constrained top-ranking. In International Joint Conference on Artificial Intelligence (IJCAI), pages 1092–1099, 2018
