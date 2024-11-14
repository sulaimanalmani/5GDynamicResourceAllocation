---
marp: true
title: Slice modeling and dynamic resource scaling
theme: default
paginate: true
author: Muhammad Sulaiman
---
# Slice modeling and dynamic resource scaling
**Purpose**: Introduction on 5G slice modeling and dynamic resource scaling using vNetRunner and [MicroOpt](https://arxiv.org/abs/2407.18342) frameworks.

**Key Tasks**

1. Set up the environment
2. Explore the dataset
3. Train VNF models using vNetRunner
4. Compose end-to-end slice models
5. Perform dynamic resource scaling using MicroOpt

---
## Set up repository and environment

Clone the repository to your local machine:
```bash
git clone https://github.com/sulaimanalmani/5GDynamicResourceAllocation.git
```

Navigate to the repo:
```bash
cd 5GDynamicResourceAllocation/
```

Install the required python packages by running these one by one:
```bash
pip install -r requirements.txt
```

Add the local bin directory to the PATH variable:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
---

## Download and extract the resource allocation dataset:
```bash
git clone https://github.com/sulaimanalmani/net_model_dataset.git
```
Navigate to the dataset directory:
```bash
cd net_model_dataset
```
Extract the dataset:
```bash
sh extract_data.sh
```

Navigate back to the main repo:
```bash
cd ../
```
---

## Launch JupyterLab:
```bash
jupyter lab
```

Once you have launched JupyterLab, it will open a new tab in your default browser. You can also access the interface by clicking the link in the terminal. The link will be in the following format:

```bash
http://127.0.0.1:8000/?token=<token>
```
For more information on JupyterLab, you can refer to the [JupyterLab documentation](https://jupyterlab.readthedocs.io/en/stable/).

---

## Accessing the jupyter notebooks

We have devided the session into three notebooks:

1. P1-WorkshopNotebook.ipynb: In this notebook, we will be exploring and visualizing our resource allocation dataset gathered from the in-lab 5G testbed.
2. P2-WorkshopNotebook.ipynb: In this notebook, we will be using the datasets to train VNF models using the vNetRunner framework. Subsequently we will be using the trained VNF models to compose end-to-end slice models.
3. P3-WorkshopNotebook.ipynb: In this notebook, we will be using the [MicroOpt](https://arxiv.org/abs/2407.18342) framework to perform dynamic resource scaling.

---

## Accessing the jupyter notebooks


Please read the tips section below, and then proceed to open each of the notebook in order in the JupyterLab (as shown below) interface and follow the instructions in the notebook to complete the exercises.

![w:400 right](images/jupyter_interface.png)

---

## Tips and Tricks

> Press the Run button in the toolbar or use `Ctrl + Enter` to execute the code in a cell.

> Double-clicking a markdown cell will reveal its code. Press `Ctrl + Enter` to execute the code in the cell and display the rendered markdown again.

> Use the sidebar on the left to navigate the file structure and explore the dataset.