from multiprocessing import Pool
import threading
import os
import subprocess
import time, random

FILELIST = 'e:\\tmp\\filelist.txt'
STANFORD_CORENLP_PATH = 'C:\\stanford-corenlp-full-2015-12-09'
JAVA_HEAP_MEMORY = '2g'

PARSER_CMD = 'java -Xmx{0} ' \
             '-cp "{1}/*;." ' \
             'edu.stanford.nlp.pipeline.StanfordCoreNLP ' \
             '-annotators tokenize,ssplit,pos,depparse ' \
             '-file {2} ' \
             '-outputDirectory {3} ' \
             '-outputFormat conll'

QSUB_MEM = '5G'
QSUB_RMEM = '3G'
QSUB_MAIL_OPTION = 'n'
QSUB_MAIL_ADDRESS = 'rjain2@sheffield.ac.uk'
QSUB_CMD = 'qsub -l mem={0} ' \
           '-l rmem={1} ' \
           '-j y -o {2} ' \
           '-m {3} -M {4} ' \
           '{5}'

FILELIST = os.path.abspath(FILELIST)

def run_parser(inputFile):
    print('parsing input file', ':', inputFile)

    outputPath = os.path.dirname(inputFile)
    parserCmd = PARSER_CMD.format(JAVA_HEAP_MEMORY, STANFORD_CORENLP_PATH, inputFile, outputPath)

    print('parserCmd', ':', parserCmd)

    qsubLogFile = os.path.join(os.path.dirname(inputFile), 'log_'+os.path.basename(inputFile))
    qsubCmd = QSUB_CMD.format(QSUB_MEM, QSUB_RMEM, qsubLogFile, QSUB_MAIL_OPTION, QSUB_MAIL_ADDRESS, parserCmd)

    print('qsubCmd', ':', qsubCmd)

    parserProcess = subprocess.Popen(qsubCmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

    for line in parserProcess.stdout:
        print(line.decode('utf-8'))

    parserProcess.stdout.close()
    parserProcess.wait()

    return

def depparse_input():
    fileList = open(FILELIST)

    inputFileNames = [line.strip() for line in fileList.readlines()]
    print('inputFileNames', ':', inputFileNames)
    for file in inputFileNames:
        print(file)

    pool = Pool(16)
    pool.map(run_parser, inputFileNames[1:2])

    pool.close()
    pool.join()

    fileList.close()

if __name__ == '__main__':
    depparse_input()