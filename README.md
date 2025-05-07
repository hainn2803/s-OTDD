# Lightspeed Geometric Dataset Distances via Projections (ICML 2025)

We are thrilled to announce that our paper has been accepted for presentation at **ICML 2025**.  
The work introduces a projection‑based optimal‑transport metric that computes dataset distances in near‑linear time, offering a practical signal for transfer learning and domain adaptation.

- **Full paper:** [Lightspeed Geometric Dataset Distances via Projections](https://arxiv.org/abs/2501.18901)
- **Reference code:** Can be found in this repo: `otdd/pytorch/sotdd.py`

## Environment Installation

### Via Conda (recommended)

If you use [ana|mini]conda , you can simply do:

```
conda env create -f environment.yaml python=3.8
conda activate otdd
conda install .
```

(you might need to install pytorch separately if you need a custom install)

### Via pip

First install dependencies. Start by install pytorch with desired configuration using the instructions provided in the [pytorch website](https://pytorch.org/get-started/locally/). Then do:
```

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

pip install -r requirements.txt
```
Finally, install this package:
```
pip install .
```


### Datasets:
We follow the experiments in [Geometric Dataset Distances via Optimal Transport](https://github.com/microsoft/otdd). Please follow the instruction in the repo for downloading the dataset. Additionally, we also use Tiny-Imagenet for large scale experiment. Please place the dataset folder as illustration:
```
data/
├── ag_news_csv/
├── amazon_review_full_csv/
├── amazon_review_polarity_csv/
├── CIFAR10/
├── CIFAR100/
├── dbpedia_csv/
├── EMNIST/
├── FashionMNIST/
├── KMNIST/
├── MNIST/
├── sogou_news_csv/
├── tiny-imagenet-200/
├── USPS/
├── yahoo_answers_csv/
├── yelp_review_full_csv/
└── yelp_review_polarity_csv/ 
```

## Experiment Scripts

### Correlation Experiment

For MNIST dataset:
```
python3 correlation_mnist_experiment.py
```

For CIFAR10 dataset:
```
python3 correlation_cifar10_experiment.py
```

### Runtime Experiment

For MNIST dataset:
```
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method hswfs
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method wte
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method sotdd
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method otdd_exact
python3 split_mnist_runtime.py --parent_dir saved_runtime_mnist --method otdd_ga
```

For CIFAR10 dataset:
```
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method hswfs
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method wte
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method sotdd
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method otdd_exact
python3 split_cifar10_runtime.py --parent_dir saved_runtime_cifar10 --method otdd_ga
```

### *NIST Experiment
```
python3 adaptation_nist.py
```

### Augmentation Experiment
```
bash augmentation.sh
```

### Text Classification Experiment

Train baseline:
```
python3 text_cls_baseline.py
```

Pretrain:
```
bash text_pretrain.sh
```

Transfer learning:
```
bash text_transfer.sh
```

Compute distance for each method
```
python3 text_dist.py --method sotdd --max_size 50000
python3 text_dist.py --method otdd --max_size 5000
```


### Tiny-Imagenet Split (224x224) Experiment

Train baseline:
```
python3 resnet18_baseline.py
```

Pretrain:
```
python3 resnet18_pretrain.py
```

Transfer learning:
```
python3 resnet18_finetune.py
```

Compute distance for s-OTDD:
```
python3 tiny_image_dist.py --parent_dir saved_split_task --num_samples 5000 --num_projections 500000
```

## 📑 Citation

If you find this work useful, please cite us:

```bibtex
@inproceedings{nguyen2025lightspeed,
  title     = {Lightspeed Geometric Dataset Distances via Projections},
  author    = {Khai Nguyen, Hai Nguyen *, Tuan Pham, and Nhat Ho},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)},
  year      = {2025},
  publisher = {PMLR},
  note      = {arXiv:2501.18901},
  url       = {https://arxiv.org/abs/2501.18901}
}