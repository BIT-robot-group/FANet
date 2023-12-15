# FANet
FANet: Fast and Accurate Robotic Grasp Detection Based on Keypoints


## Installation
Please refer to [GKNet](https://github.com/ivalab/GraspKpNet) for more installation instructions.

## Dataset
The two training datasets are provided here:
- Cornell: [Google drive, rgb](https://drive.google.com/file/d/11Hl4eNXOhEgDSUXsaP8CMVXa6GAKt0q5/view), [Google drive, rgd](https://drive.google.com/file/d/140vD1K_uq1IlCJwJq2SrTcgNUq9XQfbG/view). 
You can also use the matlab scripts provided in the `ROOT/scripts/data_aug` to generate your own dataset based on the original Cornell dataset. You will need to modify the corresponding path for loading the input images and output files.
- Abridged Jacquard Dataset (AJD):[Google drive](https://drive.google.com/file/d/1mzxMbovcaWE0Fw9ASy3atWXO7mslVYJo/view).

## Usage
After downloading datasets, place each dataset in the corresponding folder under `ROOT/Dataset/`. 
Download models [ctdet_coco_dla_2x](https://www.dropbox.com/sh/eicrmhhay2wi8fy/AAAGrToUcdp0tO-F732Xhsxwa?dl=0) and put it under `ROOT/models/`.

### Training
For training the Cornell Dataset:
~~~
python3 main.py dbmctdet_cornell --exp_id dla34 --batch_size 4 --lr 1.25e-4 --arch dla_34 --dataset cornell --num_epochs 15 --val_intervals 1 --save_all
~~~

For training AJD:
~~~
python3 main.py dbmctdet --exp_id dla34 --batch_size 4 --lr 1.25e-4 --arch dla_34 --dataset jac_coco_36 --num_epochs 30 --val_intervals 1 --save_all --load_model ../models/ctdet_coco_dla_2x.pth
~~~

### Evaluation
You can evaluate your own trained models and put them under `ROOT/models/`.

For evaluating the Cornell Dataset:
~~~
python test.py dbmctdet_cornell --exp_id dla34_test --arch dla_34 --dataset cornell --fix_res --flag_test --load_model ../models/model_dla34_cornell.pth --ae_threshold 1.0 --ori_threshold 0.24 --center_threshold 0.05
~~~

For evaluating AJD:
~~~
python test.py dbmctdet --exp_id dla34_test --arch dla_34 --dataset jac_coco_36 --fix_res --flag_test --load_model ../models/model_dla34_ajd.pth --ae_threshold 0.65 --ori_threshold 0.1745 --center_threshold 0.15
~~~

## Notice
Because I have been quite busy recently, I haven't had much time to validate the code. The actual code has undergone multiple modifications, and I am not sure if this version of the code can fully reproduce the results in the paper. It is provided for reference only. If there are any issues with the code, please feel free to raise an issue.


## Acknowledgement
- Our code is developed upon [GKNet](https://github.com/ivalab/GraspKpNet), thanks for opening source.


## Citation
If you find this project useful in your research, please consider citing:
```shell
@article{zhai2023fanet,
  title={FANet: Fast and Accurate Robotic Grasp Detection Based on Keypoints},
  author={Zhai, Di-Hua and Yu, Sheng and Xia, Yuanqing},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2023},
  publisher={IEEE}
}
```