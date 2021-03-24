Paper:Efficient Cross-modality Graph Reasoning for RGB-Infrared Person Re-identification

Pytorch Code for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset [1] and SYSU-MM01 dataset [2].

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

Sampling Strategy: N (= bacth size) person identities are randomly sampled at each step, then randomly select four visible and four thermal image. Details can be found in Line 302-307 in train.py.

Training Log: The training log will be saved in log/" dataset_name"+ log. Model will be saved in save_model/.

3. Testing.
Test a model on SYSU-MM01 or RegDB dataset by

python test.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
--dataset: which dataset "sysu" or "regdb".

--mode: "all" or "indoor" all search or indoor search (only for sysu dataset).

--trial: testing trial (only for RegDB dataset).

--resume: the saved model path.

--gpu: which gpu to run.
