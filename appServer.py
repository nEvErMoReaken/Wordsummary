import joblib
import w2v_train
import text_summary
import GlobalParameters
from flask import Flask,request,render_template
import os
app = Flask(__name__)
@app.route("/")
def index():
    return render_template("root.html")
@app.route("/get_summary",methods=["POST"])
def get_summary():
    # 加载停用词
    stopwords = w2v_train.read_stopwords ()
    # 加载模型
    model = joblib.load ( GlobalParameters.w2v_model_path )
    #获取用户数据
    content = request.form.get("ori")
    print(content)
    #获取初次摘要列表
    final_lis = text_summary.get_first_summaries(content,stopwords,model)
    #获取最终摘要列表
    summaries = text_summary.get_last_summaries(content,final_lis,stopwords,model)
    #将获得摘要拼接
    summary = ",".join(summaries)
    return render_template("index.html",summary=summary,content=content)

@app.route("/set_configs",methods=["POST"])
def set_config():
    GlobalParameters.locFirst_weight=float(request.form.get("item1"))
    GlobalParameters.locLast_weight=float(request.form.get("item2"))
    GlobalParameters.summary_len=int(request.form.get("item3"))
    GlobalParameters.last_num=int(request.form.get("item4"))
    alpha = float(request.form.get("item5"))
    if alpha:
         GlobalParameters.alpha=alpha
    return render_template ( "root.html" )
if __name__ == "__main__":
    app.run("127.0.0.1","5001",debug = True)
