from flask import Flask,request
from readfile import procesar_texto, procesar_archivo,procesar_uno_vs_all
from functools import lru_cache
import json
import numpy as np

from lista import lista
app = Flask(__name__)  


clf,tfidf=procesar_texto(lista)


@app.route('/texto', methods = ['GET'])
def pro_texto():
    data = [request.args.get('texto')]
    print(data)
    #data=['this is a test']
    textopre=tfidf.transform(data)
    result =clf.predict(textopre)[0]
    json_str = json.dumps({'result': int(result)})
    return [json_str]

@app.route('/archivo', methods = ['GET'])
def pro_archivo():
    id1 = request.args.get('id1')
    id2 = request.args.get('id2')
    ##json_str = json.dumps({'result': int(result)})
    #return [json_str]
    result =procesar_archivo(id1,id2)
    json_str = json.dumps({'result': str(result)}) 
    return [json_str]


@app.route('/unovsall', methods = ['GET'])
def unovsall():
    id = request.args.get('id')
    ##json_str = json.dumps({'result': int(result)})
    #return [json_str]
    result =procesar_uno_vs_all(id)
    json_str = json.dumps({'result': str(result)}) 
    return [json_str]


if __name__ == '__main__':
    app.run(debug=True, port=8000)