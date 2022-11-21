# TSAF: Two-Stage Alignment Framework for Unsupervised Domain Adaptation
![](http://i.iamlj.com/win/20221121161620.png)

This repository contains code for our paper: TSAF: Two-Stage Alignment Framework for Unsupervised Domain Adaptation.

## Requirements
- Python 3.6+
- PyTorch 1.5+
- math, sklearn, tensorboardX

## Datasets
You need to create a folder *data* and download the pre-processed versions of the datasets. For MIMIC-IV, you need to get permission even though it is publicly available.
- [HAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MIMIC-IV](https://physionet.org/content/mimiciv/0.4/)
- [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)

## Baselines
We compare our method with the following baselines: [CoDATS](https://dl.acm.org/doi/pdf/10.1145/3394486.3403228), [AdvSKM](https://www.ijcai.org/proceedings/2021/0378.pdf), [CAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf), [CDAN](https://proceedings.neurips.cc/paper/2018/file/ab88b15733f543179858600245108dd8-Paper.pdf), [DDC](https://arxiv.org/pdf/1412.3474.pdf), [DeepCORAL](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35), [DSAN](https://ieeexplore.ieee.org/document/9085896), [HoMM](https://ojs.aaai.org/index.php/AAAI/article/view/5745), and [MMDA](https://arxiv.org/pdf/1901.00282.pdf).  We reimplement these methods based on the original paper and adapt to new dataset

## Training & Testing
Change the config depending on what you want.
```
cd ..
# run on the HAR
python main_har --dataset HAR
# run on the MIMIC-IV
python main_mimic --dataset MIMIC
# run on the WISDM
python main_wisdm --dataset WISDM
```

## Results
![](http://i.iamlj.com/win/20221121171750.png)
