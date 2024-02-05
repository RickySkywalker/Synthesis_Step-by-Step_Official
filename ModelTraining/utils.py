'''
Author: Ruida Wang
'''
import pickle
import random

from datasets import DatasetDict, Dataset
import pyarrow as pa
import torch


def read_dataset(datasetName,
                 trainDataName,
                 evalDataName="train_gold.pkl",
                 testDataName="test_gold.pkl",
                 datasetPath="./Datasets"
                 ):
    trainFilePath = datasetPath + '/' + datasetName + '/' + trainDataName
    evalFilePath = datasetPath + '/' + datasetName + '/' + evalDataName
    testFilePath = datasetPath + '/' + datasetName + '/' + testDataName

    trainData = pickle.load(open(trainFilePath, 'rb'))
    evalData = pickle.load(open(evalFilePath, 'rb'))
    testData = pickle.load(open(testFilePath, 'rb'))

    return trainData, evalData, testData


# This function convert the data in form of [ls_of_text, label] to Dataset object in dataset
def convertToDatasetIMDB(data):
    schema = pa.schema([
        pa.field('text', pa.string()),
        pa.field('label', pa.int32())
    ])

    data_table = pa.Table.from_pydict({
        'text': [data[0][i] for i in range(len(data[0]))],
        'label': [data[1][i] for i in range(len(data[1]))]
    }, schema=schema)

    return Dataset(data_table)


# This function convert the data in form of [ls_of_premise, ls_of_question, ls_of_label] into Dataset object in dataset
def convertToDatasetQNLI(data):
    schema = pa.schema([
        pa.field('premise', pa.string()),
        pa.field('question', pa.string()),
        pa.field('label', pa.int32())
    ])

    data_table = pa.Table.from_pydict({
        'premise': [data[0][i] for i in range(len(data[0]))],
        'question': [data[1][i] for i in range(len(data[1]))],
        'label': [data[2][i] for i in range(len(data[2]))]
    }, schema=schema)

    return Dataset(data_table)


# Note the start_idx and end_idx are string level position for the dataset, we need to convert it to
# token level position during encoding
def convertToDatasetAdQA(data):
    schema = pa.schema([
        pa.field("context", pa.string()),
        pa.field("question", pa.string()),
        pa.field("answers", pa.string()),
        pa.field("start_idx", pa.int32()),
        pa.field("end_idx", pa.int32()),
        pa.field("idx", pa.string())
    ])

    data_table = pa.Table.from_pydict({
        'context': [data[0][i] for i in range(len(data[0]))],
        'question': [data[1][i] for i in range(len(data[1]))],
        'answers': [data[2][i] for i in range(len(data[2]))],
        'start_idx': [data[3][i] for i in range(len(data[3]))],
        'end_idx': [data[4][i] for i in range(len(data[4]))],
        'idx': [data[5][i] for i in range(len(data[5]))]
    }, schema=schema)

    return Dataset(data_table)


def get_best_device(i=0):
    if torch.cuda.is_available():
        return torch.device("cuda:" + str(i))
    elif torch.backends.mps.is_built():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def shuffle(data_ls):
    idx_ls = list(range(len(data_ls[0])))
    random.shuffle(idx_ls)
    for i in range(len(data_ls)):
        data_ls[i] = [data_ls[i][j] for j in idx_ls]
    return data_ls


# This function will pre-process the sst-2 dataset to take out every incomplete sentence
def preprocess_sst2(dataset):
    to_return = [[], []]

    for i in range(len(dataset[0])):
        if "." in dataset[0][i]:
            to_return[0] += [dataset[0][i]]
            to_return[1] += [dataset[1][i]]

    return to_return


def random_select(dataset, num_records=20000):
    dataset = dataset.shuffle(seed=1453)
    return dataset.select(range(num_records))
