import numpy as np
import pandas as pd
import math
import random

other = pd.read_csv("other_user.csv").to_numpy().tolist()

category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95]
store_id = [i for i in range(1, 99)]
store_id = list(set(store_id) ^ set(category_id)) #店の好みを持つ要素が格納されているセルのid

#ステイするデータの作成 list型に変換（中身（id）は1～98の値）
item_list = pd.read_csv("predict_item/predict_item_2.csv").to_numpy().tolist() #input data of other users
item_list_revise = []
for i in range(len(item_list)):
  item_list_revise.append(item_list[i][0])

#This part is for pre-experiment so you can change this part according your mission
rand_list = []
for i in range(len(other)):
  rand_item = list(random.sample(item_list_revise, k=50)) #店のデータから50個ランダムで重複なしで抽出 やっていいることは実質ランダム並び替え
  rand_list.append(rand_item) #検出した順番をランダムで決定．それを1~1000番目の推薦対象者に対して行う

other_all = pd.read_csv("other_user.csv")
other_all_for_ans = pd.read_csv("other_user_for_ans.csv")

category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95] #the ID of categories
item_id = [[1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95] for i in range(1000)] #the ID of stores
id_d_1 = [1,54,63,76,87] #the ID of item whose depth is 1
id_d_2 = [2,8,15,18,22,28,32,35,40,44,49,55,60,64,71,77,82,88,95] #the ID of item whose depth is 2
id_d_3 = list(set([i for i in range(1,99)]) ^ set(category_id)) #the ID of item whose depth is 3
ans = other_all_for_ans.to_numpy().tolist()#正解データ
change = []

#rand_list[target_num][detect]で検出順にアクセスできる
for detect in range(-1, 50, 1): #一週目は正解入力なし
  #The "detect" is the number of iter. So, if you want to update 100times, you change "detect" to 100.
  pre_score_p = []
  for target_num in range(1000): #target_num is the ID of users
    target = other_all[other_all["ID"] == target_num].to_numpy().tolist()[0] #推薦対象者データ（最初に使うのはジャンルの好みだけ）
    other = other_all[other_all["ID"] != target_num].to_numpy().tolist() #類似度を計算するのに参考にする他ユーザデータ
    if detect >= 0:
      item_id[target_num].append(rand_list[target_num][detect])

    #評価値の平均を計算
    #calculate the average
    ave_target = 0
    ave_other = []

    target_zero = 0
    for i in range(1, len(target)):
      if target[i] == 0 and i in set(item_id[target_num]): #好みが0（わからない）なら無視
        target_zero += 1
      elif target[i] != 0 and i in set(item_id[target_num]):
        ave_target += target[i]
    ave_target = ave_target / (len(set(item_id[target_num])) - target_zero)
    
    ave_other_cal = 0
    for i in range(len(other)):
      other_zero = 0
      for j in range(1, len(other[i])):
        if other[i][j] == 0 and (j in set(item_id[target_num]) or j in set(rand_list[target_num])):
          other_zero += 1
        if other[i][j] != 0 and (j in set(item_id[target_num]) or j in set(rand_list[target_num])):
          ave_other_cal += other[i][j]
      ave_other_cal = ave_other_cal / (74 - other_zero)
      ave_other.append(ave_other_cal) #推薦対象者以外のユーザの好みの平均

    #calcuate the sim↓
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
        if other[i][j] != 0 and (j in set(item_id[target_num]) or j in set(rand_list[target_num])):
          s_y_cal += (other[i][j] - ave_other[i]) ** 2
      s_y.append(math.sqrt(s_y_cal))
    
    #s_xyの計算かつ各階層の好みの似ている情報数の計算
    s_xy = []
    sim_d_1 = []
    sim_d_2 = [] #階層2で好みが近い（差が1以下）情報数を他ユーザ分（999人）格納
    sim_d_3 = [] #階層3で好みが近い（差が1以下）情報数を他ユーザ分（999人）格納
    for i in range(len(other)):
      s_xy_cal = 0
      sim_d_1_cal = 0
      sim_d_2_cal = 0
      sim_d_3_cal = 0
      for j in range(1, len(other[i])):
        if target[j] != 0 and other[i][j] != 0 and j in set(item_id[target_num]):
          s_xy_cal += (target[j] - ave_target) * (other[i][j] - ave_other[i])
        if target[j] != 0 and other[i][j] != 0 and j in id_d_1:
          if abs(target[j] - other[i][j]) <= 1:
            sim_d_1_cal += 1
        if target[j] != 0 and other[i][j] != 0 and j in id_d_2:
          if abs(target[j] - other[i][j]) <= 1:
            sim_d_2_cal += 1
        if target[j] != 0 and other[i][j] != 0 and j in id_d_3 and j in set(item_id[target_num]): #正解がわかっているid格納リストに入っている∧depth=3のアイテムid格納リストに入っている
          if abs(target[j] - other[i][j]) <= 1:
            sim_d_3_cal += 1
      sim_d_1.append(sim_d_1_cal)
      sim_d_2.append(sim_d_2_cal)
      sim_d_3.append(sim_d_3_cal)
      s_xy.append(s_xy_cal)

    #simの計算 ここが新しい計算処理
    sim = []
    for i in range(len(other)):
      if s_x == 0:
        s_x = 0.001
      if (s_xy[i] / (s_x * s_y[i])) < 0:
        sim.append(s_xy[i] / (s_x * s_y[i]))
      else:
        sim.append((s_xy[i] / (s_x * s_y[i])) * (1 + (3*sim_d_3[i])/len(sim_d_3)))
        #sim.append((s_xy[i] / (s_x * s_y[i])) * (1 + (2*sim_d_2[i])/len(sim_d_2) + (3*sim_d_3[i])/len(sim_d_3)))
    
    #Σsimの計算
    all_sim = 0
    for i in range(len(other)):
      all_sim += sim[i]
    
    #calcultate the pre↓
    #推定値の計算
    pre = []
    for i in range(1, len(other[0])): #アイテム
      cal_for_pre = 0
      if i not in set(item_id[target_num]): #ユーザ
        if i in set(rand_list[target_num]):
          for j in range(len(other)):
            if other[j][i] != 0:
              cal_for_pre += sim[j] * (other[j][i] - ave_other[j])
          if all_sim == 0:
            pre.append(0)
          else:
            pre.append(ave_target + (cal_for_pre / all_sim))
        else:
          pre.append(0)
      else:
        pre.append(target[i])
    pre_score_p.append(pre)

  #calculate the acc, TP, FP and the number of all given information
  #accuracy計算
  acc_pro = []
  positive = []
  F_P = []
  GIVE = []
  item_num = 50 #the number of items to calculate the acc
  border = 7 #the threshold when you decide if you recommend the item or not
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
      if target[item] >= border and pre_score_p[target_num][item-1] >= border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        acc_cal_positive += 1 #伝達された情報∧対象者が好きである情報
      elif target[item] < border and pre_score_p[target_num][item-1] < border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        acc_cal_negative += 1 #伝達されなかった情報∧対象者が好きでない情報
      elif target[item] < border and pre_score_p[target_num][item-1] >= border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        f_p += 1
      if pre_score_p[target_num][item-1] >= border and item not in set(category_id) and item in set(rand_list[target_num]):
        give += 1

    acc_pro.append((acc_cal_positive + acc_cal_negative) / (item_num-item_zero))
    positive.append(acc_cal_positive)
    F_P.append(f_p)
    GIVE.append(give)
  print("accuracy :", sum(acc_pro)/1000)
  print("Ture_Positive :", sum(positive)/1000)
  print("False_Positive :", sum(F_P)/1000)
  print("all give :", sum(GIVE)/1000)
  change.append(sum(acc_pro)/1000)

change = pd.DataFrame(change)
change.to_csv("1.csv", index=False)