from collections import Counter
import pandas as pd
import preprocess_helpers
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def prepare_data(data):
	data['clean'] = data['review'].progress_apply(
			preprocess_helpers.clean_pipeline)
	data['processed'] = data['clean'].progress_apply(
			preprocess_helpers.preprocess_pipeline)
	# get all processed reviews
	reviews = data.processed.values
	# merge into single variable, separated by whitespaces
	words = ' '.join(reviews)
	# obtain list of words
	words = words.split()

	vocab, word2int = build_vocabulary(words)
	reviews_enc = [
				[word2int[word] for word in review.split()]
				for review in tqdm(reviews)
				]
	seq_length = 256
	features = preprocess_helpers.pad_features(
												reviews_enc,
												pad_id=word2int['<PAD>'],
												seq_length=seq_length)
	labels = data.label.to_numpy()
	return vocab, word2int, reviews_enc, features, labels


def build_vocabulary(words):
	counter = Counter(words)
	vocab = sorted(counter, key=counter.get, reverse=True)
	int2word = dict(enumerate(vocab, 1))
	int2word[0] = '<PAD>'
	word2int = {word: id for id, word in int2word.items()}
	return vocab, word2int


def split_data(features, labels):
	# train test split
	train_size = .7     # we will use 80% of whole data as train set
	val_size = .5       # and we will use 50% of test set as validation set

	# make train set
	split_id = int(len(features) * train_size)
	train_x, remain_x = features[:split_id], features[split_id:]
	train_y, remain_y = labels[:split_id], labels[split_id:]

	# make val and test set
	split_val_id = int(len(remain_x) * val_size)
	val_x, test_x = remain_x[:split_val_id], remain_x[split_val_id:]
	val_y, test_y = remain_y[:split_val_id], remain_y[split_val_id:]

	return (train_x, train_y), (test_x, test_y), (val_x, val_y)


def create_dataloader(train_x, train_y, test_x, test_y, val_x, val_y):

	# define batch size
	batch_size = 128

	# create tensor datasets
	trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
	validset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
	testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

	# create dataloaders
	trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
	valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
	testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

	return trainloader, testloader, valloader