import pandas as pd
import math
import copy

df = pd.read_csv("other_user_for_pro.csv") #ランダム生成した穴あきデータ
pre_score = pd.read_csv("result_data_pre.csv").to_numpy().tolist() #ジャンルの好みだけ用いて協調フィルタリングした推定値
ans_score = pd.read_csv("other_user.csv").to_numpy().tolist() #答え
  
category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95]
p_num = 1
acc_result = []
border = 7
weight_sub = 0
for weight_main in range(0, 100):
  weight_main = weight_main / 10
  for weight_sub in range(0,100):
    weight_sub = weight_sub / 10
    pre_score_p = copy.deepcopy(pre_score)
    for j in range(len(pre_score_p)):
      for i in range(len(pre_score_p[j])):
        if i+1 in [3,4,5,6,7] and  ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 4) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [3,4,5,6,7] and  ans_score[j][i-1] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 4) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [9,10,11,12,13,14] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 5) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [9,10,11,12,13,14] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 5) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [19,20,21] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [19,20,21] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [23,24,25,26,27] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 4) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [23,24,25,26,27] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 4) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [29,30,31] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [29,30,31] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [33,34] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 1) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [33,34] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 1) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [36,37,38,39] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [36,37,38,39] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [41,42,43] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [41,42,43] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [45,46,47,48] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [45,46,47,48] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [50,51,52,53] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [50,51,52,53] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [56,57,58,59] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [56,57,58,59] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [61,62] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 1) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [61,62] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 1) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [65,66,67,68,69,70] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 5) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [65,66,67,68,69,70] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 5) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [72,73,74,75] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [72,73,74,75] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [78,79,80,81] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [78,79,80,81] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [83,84,85,86] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [83,84,85,86] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 3) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [89,90,91,92,93,94] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 5) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [89,90,91,92,93,94] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 5) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)

        elif i+1 in [96,97,98] and ans_score[j][i] >= border:
          pre_score_p[j][i-1] += weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = min(pre_score_p[j][i-1], 9)
        elif i+1 in [96,97,98] and  ans_score[j][i] < border:
          pre_score_p[j][i-1] -= weight_main * p_num + (weight_sub * 2) * p_num
          pre_score_p[j][i-1] = max(pre_score_p[j][i-1], 1)
    acc_pro = []
    item_num = 74
    border_ans = 7
    for target_num in range(1000):
      target = df[df["ID"] == target_num].to_numpy().tolist()
      acc_cal_positive = 0
      acc_cal_negative = 0
      item_zero = 0
      for item in range(1,len(target[0])):
        if target[0][item] == 0 and item not in category_id:
          item_zero += 1
        if target[0][item] >= border_ans and pre_score_p[target_num][item-1] >= border_ans and item not in category_id and target[0][item] != 0:
          acc_cal_positive += 1 #伝達された情報∧対象者が好きである情報
        elif target[0][item] < border_ans and pre_score_p[target_num][item-1] < border_ans and item not in category_id and target[0][item] != 0:
          acc_cal_negative += 1 #伝達されなかった情報∧対象者が好きでない情報
      acc_pro.append((acc_cal_positive + acc_cal_negative) / (item_num - item_zero))
    acc_result.append([weight_main, weight_sub, sum(acc_pro)/1000, acc_pro])
    print("weight_main =", weight_main, "weight_sub =", weight_sub, ", acc_ave =", sum(acc_pro)/1000)

acc_result = pd.DataFrame(acc_result)
acc_result.to_csv("result_data.csv", index=False)
print('finish the program!!')