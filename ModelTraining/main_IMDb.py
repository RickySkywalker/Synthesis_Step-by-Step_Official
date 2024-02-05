'''
Author: Ruida Wang
'''
import pickle
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import utils
from datasets import DatasetDict
from torch.optim import AdamW
import trainHelper
from torch.utils.data import DataLoader

# Hyper parameter settings
BATCH_SIZE = 8
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
LEARNING_RATE = 5E-6
L2 = 2E-8
EPOCH = 6
PATIENCE = 10
WARMUP_STEPS = 100

print("Para list: ")
print("\t1. batch size:", BATCH_SIZE)
print("\t2. Model name:", MODEL_NAME)
print("\t3. Max length:", MAX_LENGTH)
print("\t4. Learning rate:", LEARNING_RATE)
print("\t5. Weight decay:", L2)
print("\t6. Epoch:", EPOCH)
print("\t7. Patience:", PATIENCE)
print("\t8. Warmup steps:", WARMUP_STEPS)


def tokenize_function(data):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)


def main():
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                num_labels=2,
                                                                max_position_embeddings=MAX_LENGTH)

    trainData, evalData, testData = utils.read_dataset('imdb',
                                                       'imdb_seedData.pkl')

    additional_train_data_1 = pickle.load(open("Datasets/imdb/imdb_additional-1.pkl", 'rb'))
    additional_train_data_2 = pickle.load(open("Datasets/imdb/imdb_additional-2.pkl", 'rb'))

    additional_train_data = [additional_train_data_1[0] + additional_train_data_2[0],
                             additional_train_data_1[1] + additional_train_data_2[1]]

    trainData[0] += [additional_train_data[0][i] for i in range(len(additional_train_data[0]))]
    trainData[1] += [additional_train_data[1][i] for i in range(len(additional_train_data[0]))]

    train_dataset = utils.convertToDatasetIMDB(trainData)
    eval_dataset = utils.convertToDatasetIMDB(evalData)
    test_dataset = utils.convertToDatasetIMDB(testData)

    datasetDict = DatasetDict({'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})

    tokenizedDataset = datasetDict.map(tokenize_function, batched=True, batch_size=25000)
    tokenizedDataset = tokenizedDataset.rename_column('label', 'labels')
    tokenizedDataset.set_format("torch")

    trainDataset = tokenizedDataset['train']
    evalDataset = tokenizedDataset['eval']
    testDataset = tokenizedDataset['test']

    trainDataLoader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
    evalDataLoader = DataLoader(evalDataset, batch_size=BATCH_SIZE, shuffle=True)
    testDataLoader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2)
    device = utils.get_best_device()

    bestModel, bestAcc, bestEpochNum = trainHelper.train(model,
                                                         trainDataLoader,
                                                         evalDataLoader,
                                                         testDataLoader,
                                                         warmupSteps=WARMUP_STEPS,
                                                         numEpochs=EPOCH,
                                                         patience=PATIENCE,
                                                         optimizer=optimizer,
                                                         device=device)

    misClassData = trainHelper.getMisclassficationString(model, evalDataLoader, device=device)
    pickle.dump(misClassData, open("./misclass_data/imdb/distilBert_misclass.pkl", 'wb'))
    print("end")


if __name__ == '__main__':
    main()
