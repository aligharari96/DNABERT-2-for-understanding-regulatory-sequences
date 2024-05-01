# DNABERT-2-for-understanding-regulatory-sequences
DNase I hypersensitive sites (DHSs) are indicative markers of regulatory DNA, harboring genetic variations linked to diseases and phenotypic traits. This project endeavors to investigate the application of DNABERT-2 for deciphering DNA regulatory sequences. Within this repository, you'll find a PyTorch implementation designed for two primary tasks. The first task involves training a classifier to identify the human subsystem from which DHSs originate. In the second task, we aim to classify cancer and non-cancer cells, aiming to elucidate potential associations between DHS misregulation and disease phenotypes.

## Requirments

All required packages to run the code are listed in requirements.txt. Use the following command to install them

```bash
pip install -r requirements.txt
```

## Usage

general description of the command line prompt is:
```bash
python main.py --task --epochs --batch_size --lr --results_folder --data_dir
```
to run the experiments for the first task use the following command:
```bash
python main.py --task 1 --epochs 20000 --batch_size 128 --lr 0.0001 --results_folder "task_1_results" --data_dir "data/"
```

to run the experiments for the second task use the following command:
```bash
python main.py --task 2 --epochs 20000 --batch_size 128 --lr 0.0001 --results_folder "task_2_results" --data_dir "data/"
```
Place the data files in the 'data_dir' directory. For the second task, name the files as follows: 'cancer_train.ftr', 'cancer_test.ftr', and 'cancer_val.ftr'. For the first task, name the files as 'cell_specifity_train.ftr', 'cell_specifity_val.ftr', and 'cell_specifity_test.ftr'. All results will be saved in the 'results_folder' directory.
I have also included the downloaded files used for preparing data in the 'master_dataset.ipynb' notebook to avoid download issues and speed up execution time. Ideally, I would have placed them in a folder, but I ran out of time.
