import os

# dataset name
# dataset = 'ml-1m'
# assert dataset in ['ml-1m', 'pinterest-20']

dataset = 'ml-100k'

# model name 
model = 'NeuMF-end'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = os.path.dirname(os.path.abspath(__file__)) + '/'

train_rating = main_path + 'data/processed/{}-custom/{}.train.rating'.format(dataset,dataset)
test_rating = main_path + 'data/processed/{}-custom/{}.test.rating'.format(dataset,dataset)
test_negative = main_path + 'data/processed/{}-custom/{}.test.negative'.format(dataset,dataset)

model_path = './models/'
GMF_model_path = model_path + 'GMF.pth'
MLP_model_path = model_path + 'MLP.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
