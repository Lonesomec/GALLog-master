# GALLog: A Graph-based Log Anomaly Detection Method with Active Learning(GALLog)
An implementation of GALLog. Details are provided in our paper,
# Dependencies

To run this code fully, you'll need [PyTorch](https://pytorch.org/) (we're using version 2.0.1), [scikit-learn](https://scikit-learn.org/stable/), and 
[Pyg](https://pytorch-geometric.readthedocs.io/en/latest/index.html).
We've been running our code in Python 3.10.11.

# Running an experiment

`main_active_learning.py`\
runs an log anomaly detection experiment with GALLog.
This code here mainly corresponds to RQ1 and RQ2 in our paper. \
You can adjust the label budget in settings.py to perform experiments related to RQ2

`main_ablation.py main_RS.py main_US.py`\
runs an ablation experiment in GALLog. \
This code here corresponds to RQ3 in our paper. \
Note that for the "RS + US" experiment mentioned in RQ3, can be performed by not selecting the data enhancement option in "main_active_learning.py"

# Datasets

Due to file size limitation, this code package only provides the BGL dataset as a reference. \
For more dataset, please visit [LogHub](https://github.com/logpai/loghub) or the link provided in our paper.



