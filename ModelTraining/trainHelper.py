'''
Author: Ruida Wang
'''
import datetime
from copy import deepcopy
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm


def predFromLogits(logits):
    logits = logits.to(torch.device('cpu'))
    return torch.argmax(logits, dim=-1)


def eval(model,
         evalDataLoader,
         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    totalMis = 0
    totalLoss = 0.0
    evalRecordNum = evalDataLoader.dataset.num_rows

    # eval_progressBar = tqdm(range(len(evalDataLoader)))
    for batch in evalDataLoader:
        curr_batch = {}
        for k, v in batch.items():
            if k != 'text' \
                    and k != 'premise' and k != 'question':
                curr_batch[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**curr_batch)

        logits = outputs.logits
        pred = predFromLogits(logits)
        target = batch['labels']

        pred.to(torch.device('cpu'))
        target.to(torch.device('cpu'))
        totalMis += int(torch.sum(torch.abs(pred - target)))
        totalLoss += float(outputs.loss)

        # eval_progressBar.update(1)

    print("total Misclass:", totalMis)
    print("eval record Num:", evalRecordNum)

    evalAcc = (evalRecordNum - totalMis) / evalRecordNum

    return evalAcc, totalLoss


def train(model,
          trainDataLoader,
          evalDataLoader,
          testDataLoader,
          optimizer,
          warmupSteps=1000,
          numEpochs=32,
          patience=5,
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = model.to(device)

    numTrainingSteps = numEpochs * len(trainDataLoader)
    lrScheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=warmupSteps,
                                num_training_steps=numTrainingSteps - 1000)

    # records for get the best model
    bestModel = deepcopy(model).to(torch.device('cpu'))
    bestEpochNum = 0
    bestAccRate = 0.0
    bad_encounter = 0

    print("begin training")

    for epoch in range(numEpochs):

        print("Record for epoch:", epoch)
        print("--------------------------------------------------------")
        print("epoch begin at:", datetime.datetime.now())

        totalTrainLoss = 0.0
        totalTrainMiss = 0
        totalTrainNumRecord = trainDataLoader.dataset.num_rows

        # setting model's status to train
        model.train()

        # set up bar for this epoch
        progressBar = tqdm(range(len(trainDataLoader)))

        # training for this batch
        for batch in trainDataLoader:
            currBatch = {}
            for k, v in batch.items():
                if k != 'text' \
                        and k != 'premise' and k != 'question':
                    currBatch[k] = v.to(device)
            outputs = model(**currBatch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lrScheduler.step()
            optimizer.zero_grad()

            # get statistical results for this batch
            logits = outputs.logits
            pred = predFromLogits(logits)
            targets = batch['labels']

            totalTrainMiss += int(torch.sum(torch.abs(pred - targets)))
            totalTrainLoss += float(loss.item())
            progressBar.update(1)

        trainingAcc = (totalTrainNumRecord - totalTrainMiss) / totalTrainNumRecord

        print("training result: ")
        # print training record
        print("\tTraining Accuracy:", trainingAcc)
        print("\tTraining Loss:", totalTrainLoss)

        evalAcc, evalLoss = eval(model, evalDataLoader, device=device)

        testAcc, testLoss = eval(model, testDataLoader, device=device)

        print("evaluating result: ")
        print("\tEvaluation Accuracy:", evalAcc)
        print("\tEvaluation Loss:", evalLoss)

        print("")
        print("test result: ")
        print("\tTest Acc:", testAcc)
        print("\tTest Loss:", testLoss)

        if evalAcc > bestAccRate:
            bestAccRate = evalAcc
            bestModel = deepcopy(model).to(torch.device('cpu'))
            bestEpochNum = epoch
            bad_encounter = 0
        else:
            bad_encounter += 1

        if bad_encounter >= patience:
            break
    model = bestModel.to(device)
    testAcc, testLoss = eval(model, testDataLoader, device=device)

    print("-----------------------------------------------------")
    print("Best epoch:", bestEpochNum)
    print("Best eval Acc:", bestAccRate)

    print("Best model's testing acc:", testAcc)
    print("Best model's testing loss:", testLoss)

    return bestModel, bestAccRate, bestEpochNum


def getMisclassficationString(model, evalDataLoader,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    misClassData = [[], []]
    totalMissNum = 0
    for batch in evalDataLoader:
        curr_batch = {}
        for k, v in batch.items():
            if k != 'text' \
                    and k != 'premise' and k != 'question':
                curr_batch[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**curr_batch)

        logits = outputs.logits.to(torch.device('cpu'))
        pred = predFromLogits(logits)
        targets = batch['labels']

        misClassPos = torch.abs(pred - targets)
        totalMissNum += int(torch.sum(torch.abs(pred - targets)))
        for idx in range(len(misClassPos)):
            pos_label = misClassPos[idx]
            if pos_label == 1:
                misClassData[0].append(str(batch['text'][idx]))
                misClassData[1].append(int(targets[idx]))

    print("total misclass num:", totalMissNum)
    return misClassData


def getMisclassficationString_qnli(model, evalDataLoader,
                                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    misClassData = [[], [], []]
    model = model.to(device)
    totalMissNum = 0
    for batch in evalDataLoader:
        curr_batch = {}
        for k, v in batch.items():
            if k != 'text' \
                    and k != 'premise' and k != 'question':
                curr_batch[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**curr_batch)

        logits = outputs.logits.to(torch.device('cpu'))
        pred = predFromLogits(logits)
        targets = batch['labels']

        misClassPos = torch.abs(pred - targets)
        totalMissNum += int(torch.sum(torch.abs(pred - targets)))
        for idx in range(len(misClassPos)):
            pos_label = misClassPos[idx]
            if pos_label == 1:
                misClassData[0].append(str(batch['premise'][idx]))
                misClassData[1].append(str(batch['question'][idx]))
                misClassData[2].append(int(targets[idx]))

    print("total misclass num:", totalMissNum)
    return misClassData
