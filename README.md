This repo simulate problem when try using tensorflow keras save/load weights.

There are 2 types of dataset in this repo
1. **Abalone** dataset (`constants.POPULATION_ABALONE`) - load weight is fine for this dataset
2. **Cmc** dataset (`constants.POPULATION_CMC`) - this has problem with load weight

`main.py` is the main script to run training/evaluating

I test this repo on both Window and Mac, both have machine can reproduce the problem.

- Window -> Python 3.7.4
- Mac -> Python 3.8.5

## Install
Run `pip install -r requirements.txt`

## Reproduce
Start with `main(population_type=constants.POPULATION_CMC, is_train=True)` wait until finished. 

Then run the again with `main(population_type=constants.POPULATION_CMC, is_train=False)`. 

The output `loss` and `accuracy` of both will be different and repeat evaluate **Cmc** dataset will produce different result each times.

However, if we repeat the process with **Abalone** dataset
the problem will not occur.








