#from flask import Flask
#from flask_restx import Resource, Api

#app = Flask(__name__)
#api = Api(app)


#@api.route('/hello')
#class HelloWorld(Resource):
#    def get(self):
#        return {'hello': 'world'}


#if __name__ == '__main__':
#    app.run(debug=True)


from flask import Flask
from flask import request
from joblib import dump, load
import numpy as np
app = Flask(__name__)
best_model_path='./mymodel_0.1_val_0.1_rescale_1_gamma_0.01/model.joblib'
best_model_path2='./mymodel_DT/DT.joblib'


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#curl http://localhost:5000/predict -X POST  -H 'Content-Type: application/json' -d '{"image": ["1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","1.0", "2.0", "3.0","4.0","1.0","2.0","3.0","4.0","1.0","2.0"]}'
@app.route("/svmpredict",methods=['POST'])
def svmpredict():
    clf=load(best_model_path)
    input_json=request.json
    print(clf)
    image=input_json['image']
    image=np.array(image).reshape(1,-1)
    predicted=clf.predict(image)
    outputstr="The output from SVM is "+str(predicted[0])+"\n"
    return outputstr
    

@app.route("/decisiontreepredict",methods=['POST'])
def decisiontreepredict():
    clf=load(best_model_path2)
    input_json=request.json
    print(clf)
    image=input_json['image']
    image=np.array(image).reshape(1,-1)
    predicted=clf.predict(image)
    outputstr="The output from DT is "+str(predicted[0])+"\n"
    return outputstr


if __name__ == '__main__':
   app.run(debug=True,host='0.0.0.0')
