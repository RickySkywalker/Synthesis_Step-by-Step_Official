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
MAX_LENGTH = 512
BATCH_SIZE = 32
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 5E-6
L2 = 2E-8
EPOCH = 10
PATIENCE = 20
WARMUP_STEPS = 300

print("Para list: ")
print("\t1. batch size:", BATCH_SIZE)
print("\t2. Model name:", MODEL_NAME)
print("\t3. Max length:", MAX_LENGTH)
print("\t4. Learning rate:", LEARNING_RATE)
print("\t5. Weight decay:", L2)
print("\t6. Epoch:", EPOCH)
print("\t7. Patience:", PATIENCE)
print("\t8. Warmup steps:", WARMUP_STEPS)


def tokenize_function_qnli(data, model_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(data["question"], data["premise"], padding="max_length", truncation=True, max_length=MAX_LENGTH)


def main(batch_size,
         model_name,
         lr,
         l2,
         epoch,
         patience,
         warmup_steps):
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    trainData, evalData, testData = utils.read_dataset('qnli', 'qnli_seedData.pkl')

    additionalTrainData_1 = pickle.load(open("./Datasets/qnli/qnli_additional-1.pkl", 'rb'))
    additionalTrainData_2 = pickle.load(open("./Datasets/qnli/qnli_additional-2.pkl", 'rb'))

    for i in range(3):
        trainData[i] += additionalTrainData_1[i]
        trainData[i] += additionalTrainData_2[i]

    train_dataset = utils.convertToDatasetQNLI(trainData)
    eval_dataset = utils.convertToDatasetQNLI(evalData)
    test_dataset = utils.convertToDatasetQNLI(testData)

    datasetDict = DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_dataset
    })

    tokenizedDataset = datasetDict.map(tokenize_function_qnli, batched=True, batch_size=50000)
    tokenizedDataset = tokenizedDataset.rename_column('label', 'labels')
    tokenizedDataset.set_format('torch')

    train_dataset = tokenizedDataset['train']
    eval_dataset = tokenizedDataset['eval']
    test_dataset = tokenizedDataset['test']

    trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    evalDataLoader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    testDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=l2)

    device = utils.get_best_device(0)

    bestModel, bestAcc, bestEpochNum = trainHelper.train(model,
                                                         trainDataLoader,
                                                         evalDataLoader,
                                                         testDataLoader,
                                                         warmupSteps=warmup_steps,
                                                         numEpochs=epoch,
                                                         patience=patience,
                                                         optimizer=optimizer,
                                                         device=device)

    model = bestModel

    misClassData = trainHelper.getMisclassficationString_qnli(model, evalDataLoader, device=device)
    pickle.dump(misClassData, open("./misclass_data/qnli/distilBert_misclass.pkl", 'wb'))

    print("end")


if __name__ == '__main__':
    main(BATCH_SIZE,
         MODEL_NAME,
         LEARNING_RATE,
         L2,
         EPOCH,
         PATIENCE,
         WARMUP_STEPS)
