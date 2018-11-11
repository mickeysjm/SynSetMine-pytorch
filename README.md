# Mining Entity Synonyms with Efficient Neural Set Generation
PyTorch Implementation of Mining Entity Synonyms with Efficient Neural Set Generation.

## Installation

Simply clone this repository via
```
git clone https://github.com/mickeystroller/SynSetMine-pytorch.git
cd SynSetMine-pytorch
```

Check whether the below dependencies are satisfied. If not, simply install them via
```
pip install -r requirements.txt
```

Run the model via
```
chmod +x run.sh
./run.sh
```

By default, we will run on NYT dataset. You can uncomment the code in **run.sh** to run on the other two datasets

## Dependencies

* Python 3 with NumPy
* PyTorch > 0.4.0
* sklearn
* tensorboardX (to display/log information while model running)
* gensim (to load embedding files)
* tqdm (to display information while model running)
* networkx (to calculate one particular evaluation metric)

## Screenshot

<img src="screenshots/screenshot.gif">

## References

If you find this code useful for your research, please cite the following paper in your publication:

```
@inproceedings{Shen2019SynSetMine,
  title={Mining Entity Synonyms with Efficient Neural Set Generation},
  author={Jiaming Shen and Ruiilang Lv and Xiang Ren and Michelle Vanni and Brian Sadler and Jiawei Han},
  booktitle={AAAI},
  year={2019}
}
```

