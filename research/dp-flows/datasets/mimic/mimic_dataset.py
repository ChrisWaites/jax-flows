# This script processes MIMIC-III dataset and builds a binary matrix or a count matrix depending on your input.
# The output matrix is a Numpy matrix of type float32, and suitable for training medGAN.
# Written by Edward Choi (mp2893@gatech.edu), augmented by Chris Waites (waites@stanford.edu)
# Usage: Put this script to the folder where MIMIC-III CSV files are located. Then execute the below command.
# python process_mimic.py ADMISSIONS.csv DIAGNOSES_ICD.csv <output file> <"binary"|"count">
# Note that the last argument "binary/count" determines whether you want to create a binary matrix or a count matrix.

from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import datetime
from .. import utils


def get_patient_matrix(admissionFile, diagnosisFile, binary_count):
    if binary_count != 'binary' and binary_count != 'count':
        raise Exception('You must choose either binary or count.')

    # Building pid-admission mapping, admission-date mapping
    pidAdmMap = {}
    admDateMap = {}
    infd = open(admissionFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[1])
        admId = int(tokens[2])
        admTime = datetime.strptime(tokens[3], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap: pidAdmMap[pid].append(admId)
        else: pidAdmMap[pid] = [admId]
    infd.close()

    # Building admission-dxList mapping
    admDxMap = {}
    infd = open(diagnosisFile, 'r')
    infd.readline()
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[2])
        #dxStr = 'D_' + convert_to_icd9(tokens[4][1:-1]) ############## Uncomment this line and comment the line below, if you want to use the entire ICD9 digits.
        dxStr = 'D_' + convert_to_3digit_icd9(tokens[4][1:-1])
        if admId in admDxMap: admDxMap[admId].append(dxStr)
        else: admDxMap[admId] = [dxStr]
    infd.close()

    # Building pid-sortedVisits mapping
    pidSeqMap = {}
    for pid, admIdList in pidAdmMap.items():
        #if len(admIdList) < 2: continue
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in admIdList])
        pidSeqMap[pid] = sortedList

    # Building pids, dates, strSeqs
    pids = []
    dates = []
    seqs = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)

    # Converting strSeqs to intSeqs, and making types
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    # Constructing the matrix
    numPatients = len(newSeqs)
    numCodes = len(types)
    matrix = np.zeros((numPatients, numCodes)).astype('float32')
    for i, patient in enumerate(newSeqs):
        for visit in patient:
            for code in visit:
                if binary_count == 'binary':
                    matrix[i][code] = 1.
                else:
                    matrix[i][code] += 1.

    return matrix


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4] + '.' + dxStr[4:]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3] + '.' + dxStr[3:]
        else: return dxStr


def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4: return dxStr[:4]
        else: return dxStr
    else:
        if len(dxStr) > 3: return dxStr[:3]
        else: return dxStr


@utils.constant_seed
def get_datasets(val_prop=.1, test_prop=.1):
    dataset = get_patient_matrix(
        'datasets/mimic/ADMISSIONS.csv',
        'datasets/mimic/DIAGNOSES_ICD.csv',
        'binary'
    )

    np.random.shuffle(dataset)

    val_start = int(dataset.shape[0] * (1 - (test_prop + val_prop)))
    val_end = int(dataset.shape[0] * (1 - test_prop))

    train_dataset = dataset[:val_start]
    val_dataset = dataset[val_start:val_end]
    test_dataset = dataset[val_end:]

    return dataset, train_dataset, val_dataset, test_dataset


def postprocess(data):
    return (data > 0.5).astype(np.int32)

