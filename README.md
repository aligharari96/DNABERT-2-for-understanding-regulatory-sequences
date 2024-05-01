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
```python
from my_awesome_project import MyAwesomeClass

# Create an instance of MyAwesomeClass
awesome_instance = MyAwesomeClass()

# Call a method of MyAwesomeClass
awesome_instance.do_something_awesome()
```

## Contributing

We welcome contributions! If you'd like to contribute to My Awesome Project, please fork this repository and submit a pull request.
