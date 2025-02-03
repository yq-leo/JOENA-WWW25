# JOENA-WWW25
Implementation of "Joint Optimal Transport and Embedding for Network Alignment" in WWW25

## Datasets
You can run `main.py` directly using one of the following datasets
- phone-email
- foursquare-twitter
- ACM-DBLP
- cora
- Douban

To run your own datasets, make sure your .npz file is under the `datasets` folder with the following keys
- 'edge_index1': edge index of graph 1, shape: 2 x m1
- 'edge_index2': edge index of graph 2, shape: 2 x m2
- 'pos_pairs': anchor links used for training, shape: k1 x 2
- 'test_pairs': anchor links used for testing, shape: k2 x 2
- 'x1': (optional) node attributes of graph 1, shape: n1 x d
- 'x2': (optional) node attributes of graph 2, shape: n2 x d

## Requirements
- numpy
- scipy
- pytorch
- networkx
- torch_geometric
- tqdm

## How to use
1. Clone the repository to your local machine:

```sh
git clone https://github.com/yq-leo/JOENA-WWW25.git
```

2. Navigate to the project directory:

```sh
cd JOENA-WWW25
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

4. To run JOENA on plain networks, execute the following command in the terminal:
```sh
python main.py --dataset={dataset}
```
To run JOENA on attributed networks, add the `--use_attr` arguments:
```sh
python main.py --dataset={dataset} --use_attr
```

5. After running the code, you can visualize the training runs using TensorBoard. Execute the following command, replacing `{dataset}` with your dataset's name:
```sh
tensorboard --logdir logs/{dataset}_results
```
