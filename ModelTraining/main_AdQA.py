'''
Author: Ruida Wang
'''
import pickle
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
import utils
from datasets import DatasetDict
from torch.optim import AdamW
import trainHelperQA
from torch.utils.data import DataLoader


MAX_LENGTH = 512
BATCH_SIZE = 8
MODEL_NAME = "distilbert-base-uncased"
LEARNING_RATE = 1E-5
L2 = 1E-2
EPOCH = 10
PATIENCE = 20
WARMUP_STEPS = 0

print("Para list: ")
print("\t1. batch size:", BATCH_SIZE)
print("\t2. Model name:", MODEL_NAME)
print("\t3. Max length:", MAX_LENGTH)
print("\t4. Learning rate:", LEARNING_RATE)
print("\t5. Weight decay:", L2)
print("\t6. Epoch:", EPOCH)
print("\t7. Patience:", PATIENCE)
print("\t8. Warmup steps:", WARMUP_STEPS)

def tokenize_function_adqa(data):
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    tokenized_examples = tokenizer(
        data["question"],
        data["context"],
        truncation='only_second',
        max_length=MAX_LENGTH,
        stride=int(MAX_LENGTH/3),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # If the answer is not in the context, just remove it
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # compute the token positions of the start and end of the answer, we name the token level start and end position as
    # start_pos and end_pos
    tokenized_examples["start_pos"] = []
    tokenized_examples["end_pos"] = []
    tokenized_examples["idx"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]

        tokenized_examples["idx"].append(data["idx"][sample_index])

        start_idx = data["start_idx"][sample_index]
        end_idx = data["end_idx"][sample_index]

        offset_mapping = tokenized_examples["offset_mapping"][i]

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offset_mapping[token_start_index][0] <= start_idx and offset_mapping[token_end_index][1] >= end_idx):
            tokenized_examples["start_pos"].append(cls_index)
            tokenized_examples["end_pos"].append(cls_index)
        else:
            while token_start_index < len(offset_mapping) and offset_mapping[token_start_index][0] <= start_idx:
                token_start_index += 1
            tokenized_examples["start_pos"].append(token_start_index - 1)

            while offset_mapping[token_end_index][1] >= end_idx:
                token_end_index -= 1
            tokenized_examples["end_pos"].append(token_end_index + 1)

    return tokenized_examples


def main():
    model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)


    trainData, evalData, testData = utils.read_dataset('AdQA', 'adqa_seedData.pkl')

    additional_dataset_1 = pickle.load(open("Datasets/AdQA/adqa_additional-1.pkl", 'rb'))
    additional_dataset_2 = pickle.load(open("Datasets/AdQA/adqa_additional-2.pkl", 'rb'))

    for i in range(len(trainData)):
        trainData[i] += additional_dataset_1[i]
        trainData[i] += additional_dataset_2[i]

    train_dataset = utils.convertToDatasetAdQA(trainData)
    eval_dataset = utils.convertToDatasetAdQA(evalData)
    test_dataset = utils.convertToDatasetAdQA(testData)

    datasetDict = DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset,
        'test': test_dataset
    })

    print(datasetDict.column_names)

    tokenizedDataset = datasetDict.map(
        tokenize_function_adqa,
        batched=True,
        batch_size=50000,
        remove_columns=datasetDict["train"].column_names
    )
    tokenizedDataset.set_format('torch')

    train_dataset = tokenizedDataset["train"]
    eval_dataset = tokenizedDataset["eval"]
    test_dataset = tokenizedDataset["test"]

    trainDataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    evalDataLoader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2)

    device = utils.get_best_device(0)

    bestModel, bestAcc, bestEpochNum = trainHelperQA.train(model,
                                                         trainDataLoader,
                                                         evalDataLoader,
                                                         testDataLoader,
                                                         warmupSteps=WARMUP_STEPS,
                                                         numEpochs=EPOCH,
                                                         patience=PATIENCE,
                                                         optimizer=optimizer,
                                                         device=device)

    model = bestModel

    notEMData = trainHelperQA.getNotEMString(model, evalDataLoader, evalData, device=device)
    pickle.dump(notEMData, open("./misclass_data/adqa/distilBert_misclass3-5.pkl", 'wb'))
    print("end")



if __name__ == "__main__":
    main()