#-*- coding:utf-8 -*- coding
#@Author:Jiang
#@Date:2020/4/7
#@Time:下午 3:25
#@User:Administrator
import os

names = ['传媒','公益','娱乐','教育','时尚','互联网','军事','奥运','汽车']
for name in names:
    base_dir = "data/sougou_all/"+name
    file_paths = os.listdir("data/sougou_all/"+name)
    f1 = open ( "data/text.txt", "a+", encoding="gbk" )
    for file_path in file_paths:
       #print("dealing with"+ name)
        try:
            with open(os.path.join(base_dir,file_path),"r",encoding="gbk") as f:
                print(os.path.join(base_dir,file_path))
                content = f.readlines()[0]
                #print(content)
                f1.write(content)
                f1.write("\n")
        except:
            print("错误！")

    f1.close()