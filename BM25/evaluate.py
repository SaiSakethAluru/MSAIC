#!/usr/bin/env python
import sys, os, os.path
import numpy as np

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

print(submit_dir, truth_dir)

if not os.path.isdir(submit_dir):
	print("%s doesn't exist" % submit_dir)

if not os.path.isdir(truth_dir):
	print("%s doesn't exist" % truth_dir)

# if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

submission = open(os.path.join(submit_dir,os.listdir(submit_dir)[0]), "r")
reference = open(os.path.join(truth_dir, os.listdir(truth_dir)[0]), "r")
print("files opened")

    # submission = open("answer.tsv", "r").readlines()
    # reference = open("reference.tsv", "r").readlines()
    # assert len(submission) == len(reference), "no. of lines in submission file does not match the same in ground truth file"

preds = dict()
truths = dict()
for sub in submission:
    sub = list(map(float, sub.strip("\n").split("\t")))
    preds[int(sub[0])] = sub[1:]

for ref in reference:
    ref = list(map(float, ref.strip("\n").split("\t")))
    truths[int(ref[0])] = ref[1:]

scores = []
# print(len(truths.keys()))
cas = 0
tot = 0
for q_id in truths:
    tot+=1
    if q_id not in preds or np.sum(truths[q_id]) == 0:
        cas+=1
        scores.append(0)
    else:
        # print(np.sum(truths[q_id]))
        selected_psg = np.nonzero(truths[q_id])[0][0]
        # print(selected_psg)

        sorted_preds = np.argsort(preds[q_id])[::-1]
        rank = np.where(sorted_preds==selected_psg)[0][0] + 1
        scores.append(1.0/rank)
        # print(rank)
print(tot)
print(cas)
score = np.mean(scores)
print(score)

output_filename = os.path.join(output_dir, 'scores.txt')              
output_file = open(output_filename, 'w')
output_file.write("Difference: %f" % score)
output_file.close()