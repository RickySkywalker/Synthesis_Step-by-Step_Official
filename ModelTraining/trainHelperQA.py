'''
Author: Ruida Wang
'''
import datetime
from copy import deepcopy
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from collections import Counter

def _judgeExactMatch(startPosArr, endPosArr, tar_startPosArr, tar_endPosArr):
    curr_device = torch.device('cpu')
    startPosArr = startPosArr.to(curr_device)
    endPosArr = endPosArr.to(curr_device)
    tar_startPosArr = tar_startPosArr.to(curr_device)
    tar_endPosArr = tar_endPosArr.to(curr_device)

    start_eq = torch.eq(startPosArr, tar_startPosArr)
    end_eq = torch.eq(endPosArr, tar_endPosArr)

    return torch.logical_and(start_eq, end_eq)

def _getIdxFromLogits(logits):
    logits = logits.to('cpu')
    return torch.argmax(logits, dim=-1)

def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.tolist()
    ground_truth_tokens = ground_truth.tolist()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def _getF1Score(predTokenLs, trueTokenLs):
    to_return = torch.zeros(len(predTokenLs))
    for i in range(len(predTokenLs)):
        curr_predToken = predTokenLs[i]
        curr_trueToken = trueTokenLs[i]
        to_return[i] = f1_score(curr_trueToken, curr_predToken)
    return to_return

def _getAnswerToken(input_ids, predStartPos, predEndPos):
    to_return = []
    for i in range(len(input_ids)):
        to_return.append(input_ids[i, predStartPos[i] : predEndPos[i] + 1])
    return to_return


def eval(model, evalDataLoader, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model.eval()
    totalEM = 0
    totalLoss = 0.0
    totalF1 = 0.0
    evalRecordNum = evalDataLoader.dataset.num_rows

    for batch in evalDataLoader:
        currBatch = {}

        for k, v in batch.items():
            if k == "attention_mask" or k == "input_ids":
                currBatch[k] = v.to(device)

        tarStartPos = batch["start_pos"].to(device)
        tarEndPos = batch["end_pos"].to(device)

        with torch.no_grad():
            outputs = model(**currBatch, start_positions=tarStartPos, end_positions=tarEndPos)

        predStartPos = _getIdxFromLogits(outputs.start_logits)
        predEndPos = _getIdxFromLogits(outputs.end_logits)

        predAnsTokens = _getAnswerToken(batch["input_ids"], predStartPos, predEndPos)
        trueAnsTokens = _getAnswerToken(batch["input_ids"], batch["start_pos"], batch["end_pos"])

        exactMatch = _judgeExactMatch(predStartPos, predEndPos, tarStartPos, tarEndPos)
        F1 = _getF1Score(predAnsTokens, trueAnsTokens)

        totalEM += int(torch.sum(exactMatch))
        totalF1 += float(torch.sum(F1))
        totalLoss += float(outputs.loss)

    print("total Exact Match:", totalEM)
    print("eval record Num:", evalRecordNum)

    evalAcc = (totalEM)/evalRecordNum
    avgEvalLoss = totalLoss/evalRecordNum
    avgF1 = totalF1/evalRecordNum

    return evalAcc, avgF1, avgEvalLoss


def train(model,
          trainDataLoader,
          evalDataLoader,
          testDataLoader,
          optimizer,
          warmupSteps=1000,
          numEpochs=32,
          patience = 5,
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = model.to(device)

    numTrainingSteps = numEpochs * len(trainDataLoader)
    lrScheduler = get_scheduler(name='linear', optimizer=optimizer, num_warmup_steps=warmupSteps,
                                num_training_steps=numTrainingSteps - 1000)

    # records for the best model, we use EM as the judging criteria here
    bestModel = deepcopy(model).to(torch.device('cpu'))

    bestEMEpochNum = 0
    bestF1EpochNum = 0
    bestF1 = 0.0
    bestEM = 0.0
    bestEvalLoss = 0.0
    bad_counter = 0

    print("begin training")

    for epoch in range(numEpochs):
        print("Record for epoch:", epoch)
        print("--------------------------------------------------------")
        print("epoch begin at:", datetime.datetime.now())

        totalTrainLoss = 0.0
        totalTrainEM = 0
        totalTrainF1 = 0.0
        totalTrainEM = 0.0
        totalTrainNumRecord = trainDataLoader.dataset.num_rows

        model.train()

        # set up bar for this epoch
        progressBar = tqdm(range(len(trainDataLoader)))

        for batch in trainDataLoader:
            currBatch = {}

            for k, v in batch.items():
                if k == "attention_mask" or k == "input_ids":
                    currBatch[k] = v.to(device)

            tarStartPos = batch["start_pos"].to(device)
            tarEndPos = batch["end_pos"].to(device)

            outputs = model(**currBatch, start_positions=tarStartPos, end_positions=tarEndPos)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lrScheduler.step()
            optimizer.zero_grad()

            predStartPos= _getIdxFromLogits(outputs.start_logits)
            predEndPos = _getIdxFromLogits(outputs.end_logits)

            predAnsTokens = _getAnswerToken(batch["input_ids"], predStartPos, predEndPos)
            trueAnsTokens = _getAnswerToken(batch["input_ids"], batch["start_pos"], batch["end_pos"])

            exactMatch = _judgeExactMatch(predStartPos, predEndPos, tarStartPos, tarEndPos)

            F1 = _getF1Score(predAnsTokens, trueAnsTokens)

            totalTrainEM += int(torch.sum(exactMatch))
            totalTrainF1 += float(torch.sum(F1))
            totalTrainLoss += float(loss.item())

            progressBar.update(1)

        trainingAcc = (totalTrainEM)/totalTrainNumRecord
        trainingF1 = totalTrainF1/totalTrainNumRecord
        avgTrainingLoss = totalTrainLoss/totalTrainNumRecord

        print("training result: ")
        print("\tTraining Accuracy:", trainingAcc)
        print("\tAvg Training F1:", trainingF1)
        print("\tAvg Training Loss:", avgTrainingLoss)

        evalAcc, evalF1, evalLoss = eval(model, evalDataLoader, device=device)
        testAcc, testF1, testLoss = eval(model, testDataLoader, device=device)

        print("evaluation result:")
        print("\tEvaluation Accuracy:", evalAcc)
        print("\tEvaluation F1:", evalF1)
        print("\tEvaluation Loss", evalLoss)

        print("test result:")
        print("\tTest Accuracy:", testAcc)
        print("\tTest F1:", testF1)
        print("\tTest Loss", testLoss)

        if testAcc > bestEM:
            bestEM = testAcc
            bestModel = deepcopy(model).to(torch.device("cpu"))
            bestEMEpochNum = epoch
            bestEvalLoss = testLoss
            bad_counter = 0
        else:
            bad_counter += 1

        if testF1 > bestF1:
            bestF1 = testF1
            bestF1EpochNum = epoch
            bad_counter = 0

        if bad_counter >= patience:
            break
    model = bestModel.to(device)
    testAcc, testF1, testLoss = eval(model, testDataLoader, device=device)

    print("-----------------------------------------------------")
    print("Best EM epoch:", bestEMEpochNum)
    print("Best test EM:", bestEM)
    print("Best test Acc:", bestEvalLoss)
    print("")
    print("Best test F1 epoch:", bestF1EpochNum)
    print("Best test F1:", bestF1)

    print("")

    print("Best model's testing EM:", testAcc)
    print("Best model's testing F1:", testF1)
    print("Best model's testing Loss:", testLoss)

    return bestModel, bestEM, bestEMEpochNum


def getNotEMString(model, evalDataLoader, evalData, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = model.to(device)

    evalContextDic = {}
    evalQuestionDic = {}
    evalAnswerDic = {}
    evalAnswerStartDic = {}
    evalAnswerEndDic = {}
    for i in range(len(evalData[5])):
        currIdx = evalData[5][i]
        evalContextDic[currIdx] = evalData[0][i]
        evalQuestionDic[currIdx] = evalData[1][i]
        evalAnswerDic[currIdx] = evalData[2][i]
        evalAnswerStartDic[currIdx] = evalData[3][i]
        evalAnswerEndDic[currIdx] = evalData[4][i]

    totalNotEMNum = 0
    notEMData = [[], [], [], [], []]
    
    for batch in evalDataLoader:
        currBatch = {}

        for k, v in batch.items():
            if k == "attention_mask" or k == "input_ids":
                currBatch[k] = v.to(device)

        tarStartPos = batch["start_pos"].to(device)
        tarEndPos = batch["end_pos"].to(device)

        with torch.no_grad():
            outputs = model(**currBatch, start_positions=tarStartPos, end_positions=tarEndPos)

        predStartPos = _getIdxFromLogits(outputs.start_logits)
        predEndPos = _getIdxFromLogits(outputs.end_logits)

        exactMatch = _judgeExactMatch(predStartPos, predEndPos, tarStartPos, tarEndPos)

        totalNotEMNum += (len(exactMatch) - int(torch.sum(exactMatch)))

        for i in range(len(exactMatch)):
            currBool = exactMatch[i]
            if not currBool:
                curr_idx = batch["idx"][i]
                notEMData[0] += [evalContextDic[curr_idx]]
                notEMData[1] += [evalQuestionDic[curr_idx]]
                notEMData[2] += [evalAnswerDic[curr_idx]]
                notEMData[3] += [evalAnswerStartDic[curr_idx]]
                notEMData[4] += [evalAnswerEndDic[curr_idx]]


    print("total not EM num:", totalNotEMNum)
    return notEMData






