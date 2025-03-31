# Internship Task
## Improving Writing Assistance at JetBrains AI

### Report
Check out [report](report.pdf) with the results.

### Steps to reproduce
**1. Install required libraries**\
    ```sh
    pip3 install -r requirements.txt
    ```\
**2. Reduce dataset size**\
    Folder `datasets` contains all required datasets. The file [data.py](src/data.py) creates a balanced dataset containing a total of 50000 samples. Additionally, it includes helper functions for data preprocessing, which are later used in the training phase.\
    ```sh
    python3 src/data.py
    ```\
**3. Train models**\
    Check out file [training.ipynb](training.ipynb) 
