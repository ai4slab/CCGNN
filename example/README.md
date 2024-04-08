## Example

CCGNN for miRNA-target interaction prediction.



**Step 1**. virtual environment creation:

```
conda env create -f environment.yml
```



**Step 2**. data processing:

```
python data_process.py
```



**Step 3**. model training and prediction:

```
python main.py
```



The prediction performance of CCGNN is as follows:

<div style="display: flex;">
    <img src="Figs/ROC%20curve.png" alt="Image 1" style="width: 50%; margin-right: 5px;">
    <img src="Figs/PR%20curve.png" alt="Image 2" style="width: 50%; margin-left: 5px;">
</div>