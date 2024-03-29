# NeAR
## Data ([Google Drive](https://drive.google.com/drive/folders/1DpUjb1NFiZkyeAII_Dvvn0iO_QIqkU-E?usp=share_link)) | Paper ([arXiv](https://arxiv.org/abs/2206.15328), [MICCAI'22](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_48))
[Jiancheng Yang](https://jiancheng-yang.com/), Rui Shi, Udaranga Wickramasinghe, Qikui Zhu, [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ), [Pascal Fua](https://people.epfl.ch/pascal.fua)

![NeAR_overview](https://raw.githubusercontent.com/M3DV/NeAR/main/assets/overview.jpg)

The human annotations are imperfect, especially when produced by junior practitioners. Multi-expert consensus is usually regarded as golden standard, while this annotation protocol is too expensive to implement in many real-world projects. In this study, we propose a method to refine human annotation, named Neural Annotation Refinement (NeAR). It is based on a learnable implicit function, which decodes a latent vector into represented shape. By integrating the appearance as an input of implicit functions, the appearance-aware NeAR fixes the annotation artefacts. Our method is demonstrated on the application of adrenal gland analysis. We first show that the NeAR can repair distorted golden standards on a public adrenal gland segmentation dataset. Besides, we develop a new Adrenal gLand ANalysis (ALAN) dataset with the proposed NeAR, where each case consists of a 3D shape of adrenal gland and its diagnosis label (normal vs. abnormal) assigned by experts. We show that models trained on the shapes repaired by the NeAR can diagnose adrenal glands better than the original ones. The ALAN dataset will be open-source, with 1,584 shapes for adrenal gland diagnosis, which serves as a new benchmark for medical shape analysis. 

For more details, please refer to our paper:

**Neural Annotation Refinement: Development of a New 3D Dataset for Adrenal Gland Analysis** ([arXiv](https://arxiv.org/abs/2206.15328))

## Data 
Download ALAN dataset and Synthetic data [here](https://drive.google.com/drive/folders/1DpUjb1NFiZkyeAII_Dvvn0iO_QIqkU-E?usp=share_link). There are two directories in this link:
* `synthetic_data`: the adrenal gland subset of AbdomenCT-1K
    * `appearance`: image appearance
    * `mask_gt`: ground truth annotation
    * `mask_distorted`: distorted annotation
    * `mask_near_refined`: annotation refined with NeAR method
* `ALAN`: a new 3D dataset for Adrenal gLand ANalysis
    * `ALAN`: 1,584 cases of NeAR-repaired, high-quality 3D models of adrenal glands
    * `info.csv`: diagnosis labels


## Code Structure

* [`near/`](./near/):
    * [`datasets/`](./near/datasets/): PyTorch datasets and dataloaders of ALAN.
    * [`models/`](./near/models/): model scripts of NeAR.
    * [`utils/`](./near/utils/)
* [`repairing/`](./repairing/):
    * [`near_repairing/`](./repairing/near_repairing/): NeAR repairing scripts.
    * [`seg_repairing/`](./repairing/seg_repairing/): counterparts (Seg-UNet and Seg-FCN) repairing scripts. 


## How to run

To repairing your own dataset: 

Write your own `Dataset` class as `AlanDataset` in [`refine_dataset.py`](./near/datasets/refine_dataset.py)

Modify `AlanDataset` in [`near_repair.py`](./repairing/near_repairing/near_repair.py) to your own dataset

Assign parameters in [`config_near.py`](./repairing/near_repairing/config_near.py) or use the default parameters

Run the script in terminal: 

    >>> python ./repairing/near_repairing/near_repair.py


## Reference
If you find this project useful in your research, please cite the following paper:

      Yang, J., Shi, R., Wickramasinghe, U., Zhu, Q., Ni, B., & Fua, P. (2022). Neural Annotation Refinement: Development of a New 3D Dataset for Adrenal Gland Analysis. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 503-513). Springer, Cham.

or using the bibtex:

      @inproceedings{yang2022neural,
        title={Neural Annotation Refinement: Development of a New 3D Dataset for Adrenal Gland Analysis},
        author={Yang, Jiancheng and Shi, Rui and Wickramasinghe, Udaranga and Zhu, Qikui and Ni, Bingbing and Fua, Pascal},
        booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages={503--513},
        year={2022},
        organization={Springer}
      }
