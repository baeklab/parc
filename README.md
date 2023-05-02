# PARC: Physics-Aware Recurrent Convolutions
<a href="https://arxiv.org/abs/2204.07234"><img src="https://img.shields.io/badge/cond.mat-arXiv%3A2204.07234-B31B1B.svg"></a>
<a href="https://arxiv.org/abs/2211.04561"><img src="https://img.shields.io/badge/cond.mat-arXiv%3A2211.04561-B31B1B.svg"></a>


Official implementation of [Nguyen, P.C.H., Nguyen, Y.T., Choi, J.B., Seshadri, P., Udaykumar, H.S., & Baek, S.S. (2023). PARC: Physics-Aware Recurrent Convolutional Neural Networks to Assimilate Meso-scale Reactive Mechanics of Energetic Materials. *Science Advances, 9*(17):eadd6868.](https://www.science.org/doi/10.1126/sciadv.add6868)

## Getting Started

### Virtual Environment
It is recommended to create a virtual environment using [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and install PARC within the virtual environment. To create a virtual environment, type the following command in your terminal or command prompt.
```bash
conda create -n parc python=3.8 ipykernel
```
Once the virtual environment has been created, run the following to activate the environment.
```bash
conda activate parc
```

### Dependencies

#### TensorFlow
Tested and developed on TensorFlow 2.8.0. It should be compatible with other TensorFlow2 versions, but we haven't tested. Make sure you follow the installation instructions in https://www.tensorflow.org/install to install TensorFlow2 according to your system configuration.
```bash
pip install tensorflow
```

#### Other Dependencies
```bash
pip install -r requirements.txt
```

### Clone This Repository
```bash
git clone https://github.com/stephenbaek/parc.git
```

```bash
cd parc
```

### Run Examples

The details for using the PARC model is best described in the `demos/PARC_demo.ipynb`. 

## Citation
To cite this work, please use the following:
```
@article{
  nguyen2023parc,
  author = {Phong C. H. Nguyen  and Yen-Thi Nguyen  and Joseph B. Choi
   and Pradeep K. Seshadri  and H. S. Udaykumar  and Stephen S. Baek },
  title = {{PARC}: Physics-aware recurrent convolutional neural networks to
   assimilate meso scale reactive mechanics of energetic materials},
  journal = {Science Advances},
  volume = {9},
  number = {17},
  pages = {eadd6868},
  year = {2023},
  doi = {10.1126/sciadv.add6868},
  URL = {https://www.science.org/doi/abs/10.1126/sciadv.add6868}
}

@article{
  nguyen2023parcel,
  author = {Phong C. H. Nguyen and Yen-Thi Nguyen and Pradeep K. Seshadri
   and Joseph B. Choi and H. S. Udaykumar and Stephen Baek},
  title = {A Physics-Aware Deep Learning Model for Energy Localization in
   Multiscale Shock-To-Detonation Simulations of Heterogeneous Energetic Materials},
  journal = {Propellants, Explosives, Pyrotechnics},
  volume = {48},
  number = {4},
  pages = {e202200268},
  year = {2023},
  doi = {https://doi.org/10.1002/prep.202200268},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/prep.202200268}
}
```
