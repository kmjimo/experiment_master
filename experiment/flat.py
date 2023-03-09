import numpy as np
import pandas as pd
import math
import random

other = pd.read_csv("other_user.csv").to_numpy().tolist()

category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95]
store_id = [i for i in range(1, 99)]
store_id = list(set(store_id) ^ set(category_id)) #店id

#推定するデータの作成 list型に変換（中身（id）は1～98の値）
item_list = pd.read_csv("predict_item_5.csv").to_numpy().tolist()
item_list_revise = []
for i in range(len(item_list)):
  item_list_revise.append(item_list[i][0])

rand_list = []
for i in range(len(other)):
  rand_item = list(random.sample(item_list_revise, k=50)) #店のデータから50個ランダムで重複なしで抽出 やっていいることは実質ランダム並び替え
  rand_list.append(rand_item) #検出した順番をランダムで決定．それを1~1000番目の推薦対象者に対して行う

item_id = [list(set(item_list_revise) ^ set(store_id)) for _ in range(1000)] #推定するのに参考にする店のid（24個）

other_all = pd.read_csv("other_user.csv")
other_all_for_ans = pd.read_csv("other_user_for_ans.csv")

category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95]
ans = other_all_for_ans.to_numpy().tolist()#正解データ
change = []

for detect in range(-1, 50, 1): #一週目は正解入力なし
  pre_score = []
  for target_num in range(1000): #target_numは推薦対象者のID
    target = other_all[other_all["ID"] == target_num].to_numpy().tolist()[0]
    other = other_all[other_all["ID"] != target_num].to_numpy().tolist()
    if detect >= 0:
      item_id[target_num].append(rand_list[target_num][detect]) #検出した順番に正解データを入れていく

    #評価値の平均を計算
    ave_target = 0
    ave_other = []

    target_zero = 0
    for i in range(1, len(target)):
      if target[i] == 0 and i in set(item_id[target_num]): #好みが0（わからない）なら無視
        target_zero += 1
      elif target[i] != 0 and i in set(item_id[target_num]):
        ave_target += target[i]
    if target_zero == len(item_id[target_num]):
      ave_target = 0
    else:
      ave_target = ave_target / (len(item_id[target_num]) - target_zero)
    
    ave_other_cal = 0
    for i in range(len(other)):
      other_zero = 0
      for j in range(1, len(other[i])):
        if other[i][j] == 0 and j not in set(category_id):
          other_zero += 1
        elif other[i][j] != 0 and j not in set(category_id):
          ave_other_cal += other[i][j]
      if other_zero == 74:
        ave_other_cal = 0
      else:
        ave_other_cal = ave_other_cal / (74 - other_zero)
      ave_other.append(ave_other_cal)

    #s_xの計算
    s_x = 0
    for i in range(1, len(target)):
      if target[i] != 0 and i in set(item_id[target_num]): #好みが0（わからない）なら無視
        s_x += (target[i] - ave_target) ** 2
    s_x = math.sqrt(s_x)
    
    #s_yの計算
    s_y = []
    for i in range(len(other)):
      s_y_cal = 0
      for j in range(1, len(other[i])):
        if other[i][j] != 0 and j not in set(category_id):
          s_y_cal += (other[i][j] - ave_other[i]) ** 2
      s_y.append(math.sqrt(s_y_cal))
    
    #s_xyの計算
    s_xy = []
    for i in range(len(other)):
      s_xy_cal = 0
      for j in range(1, len(other[i])):
        if target[j] != 0 and other[i][j] != 0 and j in set(item_id[target_num]):
          s_xy_cal += (target[j] - ave_target) * (other[i][j] - ave_other[i])
      s_xy.append(s_xy_cal)

    #simの計算
    sim = []
    for i in range(len(other)):
      if s_x == 0:
        sim.append(0)
      else:
        sim.append(s_xy[i] / (s_x * s_y[i]))
    
    #Σsimの計算
    all_sim = 0
    for i in range(len(other)):
      all_sim += sim[i]
    
    #推定値の計算
    pre = []
    for i in range(1, len(other[0])):
      cal_for_pre = 0
      if i not in set(item_id[target_num]) and i not in category_id:
        for j in range(len(other)):
          if other[j][i] != 0:
            cal_for_pre += sim[j] * (other[j][i] - ave_other[j])
        if all_sim == 0:
          pre.append(0)
        else:
          pre.append(ave_target + (cal_for_pre / all_sim))
      else:
        pre.append(target[i])
    pre_score.append(pre)

  #accuracy計算
  acc_cf = []
  positive = []
  F_P = []
  GIVE = []
  item_num = 50
  border = 7
  for target_num in range(1000):
    target = ans[target_num] # 答え

    acc_cal_positive = 0
    acc_cal_negative = 0
    f_p = 0
    give = 0
    item_zero = 0
    for item in range(1,len(target)):
      if target[item] == 0 and item not in set(category_id) and item in set(rand_list[target_num]):
        item_zero += 1
      if target[item] >= border and pre_score[target_num][item-1] >= border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        acc_cal_positive += 1 #伝達された情報∧対象者が好きである情報
      elif target[item] < border and pre_score[target_num][item-1] < border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        acc_cal_negative += 1 #伝達されなかった情報∧対象者が好きでない情報
      elif target[item] < border and pre_score[target_num][item-1] >= border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        f_p += 1
      elif target[item] == 0 and pre_score[target_num][item-1] >= border and item not in set(category_id) and item in set(rand_list[target_num]):
        give += 1

    acc_cf.append((acc_cal_positive + acc_cal_negative) / (item_num-item_zero))
    positive.append(acc_cal_positive)
    F_P.append(acc_cal_negative)
    GIVE.append(give+acc_cal_positive+acc_cal_negative)
  print("accuracy :", sum(acc_cf)/1000)
  print("True_positive :", sum(positive)/1000)
  print("False_Positive :", sum(F_P)/1000)
  print("all give :", sum(GIVE)/1000)
  change.append(sum(acc_cf)/1000)