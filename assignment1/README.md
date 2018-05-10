On Debian based systems run the following:

### Requirements
Requirements python 3.5+
```bash
sudo apt install -y python3 pip3
```

To install python dependencies
```bash
sudo python3 -m pip install pipenv
pipenv install
```

### Run
```bash
$ python3 assignment1/assignment1.py --help

usage: assignment1.py [-h] [--visualize] [--smote-experiment]
                      [--classify-blackbox] [--classify-whitebox]

Assignment 1 - Cyber Data Analytics

optional arguments:
  -h, --help           show this help message and exit
  --visualize          Produce visualizations
  --smote-experiment   Run the imbalance task experiment
  --classify-blackbox  Fit and cross validate black-box algorithm
  --classify-whitebox  Fit and cross validate white-box algorithm
```
