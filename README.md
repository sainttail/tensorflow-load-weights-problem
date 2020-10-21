This repo simulate problem when try using tensorflow keras save/load weights.

There are 2 type of datasets in this repo
1. **Abalone** dataset (`constants.POPULATION_ABALONE`) - load weight for fine for this dataset
2. **Cmc** dataset (`constants.POPULATION_CMC`) - this has problem with load weight

`main.py` is the main script to run training/evaluating

## Install
Run `pip install -r requirements.txt`

## Reproduce
Start with `main(population_type=constants.POPULATION_CMC, is_train=True)` wait until finished. 

Then run the again with `main(population_type=constants.POPULATION_CMC, is_train=False)`. 

The output `loss` and `accuracy` of both will be different, but if we repeat the process with **Abalone** dataset
the problem will not occur.








