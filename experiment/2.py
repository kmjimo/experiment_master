import numpy as np
import pandas as pd
import math
import random

other_50 = pd.read_csv("other_user_flat.csv").to_numpy().tolist()
other = pd.read_csv("other_user.csv").to_numpy().tolist()

category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95]
store_id = [i for i in range(1, 99)]
store_id = list(set(store_id) ^ set(category_id)) #店の好みを持つ要素が格納されているセルのid

#推定するデータの作成 list型に変換（中身（id）は1～98の値）
item_list = pd.read_csv("predict_item.csv").to_numpy().tolist()
item_list_revise = []
for i in range(len(item_list)):
  item_list_revise.append(item_list[i][0])

rand_list = []
for i in range(len(other)):
  rand_item = list(random.sample(item_list_revise, k=50)) #店のデータから50個ランダムで重複なしで抽出 やっていいることは実質ランダム並び替え
  rand_list.append(rand_item) #検出した順番をランダムで決定．それを1~1000番目の推薦対象者に対して行う

other_all = pd.read_csv("other_user.csv")
other_all_for_ans = pd.read_csv("other_user_for_ans.csv")
other_50 = pd.read_csv("other_user_for_pro.csv")

category_id = [1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95]
item_id = [[1,2,8,15,18,22,28,32,35,40,44,49,54,55,60,63,64,71,76,77,82,87,88,95] for i in range(1000)]
id_d_2 = [2,8,15,18,22,28,32,35,40,44,49,55,60,64,71,77,82,88,95]
id_d_3 = list(set([i for i in range(1,99)]) ^ set(category_id))
ans = other_all_for_ans.to_numpy().tolist()#正解データ
change = []

#rand_list[target_num][detect]で検出順にアクセスできる
for detect in range(-1, 50, 1): #一週目は正解入力なし
  pre_score_p = []
  for target_num in range(1000): #target_numは推薦対象者のID
    target = other_all[other_all["ID"] == target_num].to_numpy().tolist()[0] #推薦対象者データ（最初に使うのはジャンルの好みだけ）
    other = other_50[other_all_for_ans["ID"] != target_num].to_numpy().tolist() #類似度を計算するのに参考にする他ユーザデータ
    if detect >= 0:
      item_id[target_num].append(rand_list[target_num][detect])

    #評価値の平均を計算
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
    
    #s_xyの計算 ここが新しい計算処理
    s_xy = []
    for i in range(len(other)):
      s_xy_cal = 0
      for j in range(1, len(other[i])):
        if target[j] != 0 and other[i][j] != 0 and j in id_d_2 and j in set(item_id[target_num]):
          s_xy_cal += (target[j] - ave_target) * (other[i][j] - ave_other[i]) * 1/max(1,abs(target[j] - other[i][j])) * 2
        if target[j] != 0 and other[i][j] != 0 and j in id_d_3 and j in set(item_id[target_num]):
          s_xy_cal += (target[j] - ave_target) * (other[i][j] - ave_other[i]) * 1/max(1,abs(target[j] - other[i][j])) * 3
      s_xy.append(s_xy_cal)

    #simの計算
    sim = []
    for i in range(len(other)):
      if s_x == 0:
        s_x = 0.001
      sim.append(s_xy[i] / (s_x * s_y[i]))
    
    #Σsimの計算
    all_sim = 0
    for i in range(len(other)):
      all_sim += sim[i]
    
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


  #accuracy計算
  acc_pro = []
  item_num = 50
  border = 7
  for target_num in range(1000):
    target = ans[target_num] # 答え

    acc_cal_positive = 0
    acc_cal_negative = 0
    item_zero = 0
    for item in range(1,len(target)):
      if target[item] == 0 and item not in set(category_id) and item in set(rand_list[target_num]):
        item_zero += 1
      if target[item] >= border and pre_score_p[target_num][item-1] >= border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        acc_cal_positive += 1 #伝達された情報∧対象者が好きである情報
      elif target[item] < border and pre_score_p[target_num][item-1] < border and item not in set(category_id) and target[item] != 0 and item in set(rand_list[target_num]):
        acc_cal_negative += 1 #伝達されなかった情報∧対象者が好きでない情報

    acc_pro.append((acc_cal_positive + acc_cal_negative) / (item_num-item_zero))
  print("accuracy :", sum(acc_pro)/1000)
  change.append(sum(acc_pro)/1000)

change = pd.DataFrame(change)
change.to_csv("2.csv", index=False)