
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd



from protlearn.features import aac



def solve(a):
    name_to_sequence = {}
    sequence_to_name = {}
    names = set()
    data = set()
    new = a.split()
    a1 = 1
    s = ""
    sequence = ""
    for i in new:
        if a1 % 2 == 1:
            s = i
            #print(s)
        else:
            sequence = i
            #print(sequence)

            names.add(s)

            data.add(sequence)

            if s not in name_to_sequence:
                name_to_sequence[s] = set()
            name_to_sequence[s].add(sequence)

            if sequence not in sequence_to_name:
                sequence_to_name[sequence] = set()
            sequence_to_name[sequence].add(s)
        a1 = a1+1
    data_list = list(data)
    comp, aa = aac(data_list, remove_zero_cols=False)
    column_names = ['AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H', 'AAC_I', 'AAC_K', 'AAC_L', 
                'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T', 'AAC_V', 'AAC_W', 'AAC_Y']
    df = pd.DataFrame(comp, columns=column_names)


    return df





def solvea(a):
    
    new = a.split()
    comp, aa = aac(new, remove_zero_cols=False)
    column_names = ['AAC_A', 'AAC_C', 'AAC_D', 'AAC_E', 'AAC_F', 'AAC_G', 'AAC_H', 'AAC_I', 'AAC_K', 'AAC_L', 
                'AAC_M', 'AAC_N', 'AAC_P', 'AAC_Q', 'AAC_R', 'AAC_S', 'AAC_T', 'AAC_V', 'AAC_W', 'AAC_Y']
    df = pd.DataFrame(comp, columns=column_names)
    



    return df




def answer(data):
    

      output_ans = []
      for col_idx, column in enumerate(zip(*data)):
          zero_count = column.count(0)
          one_count = column.count(1)

          if zero_count > 6:
              output_ans.append("negative")

          else:
              output_ans.append("positive")

      return list(output_ans)
  
def examples_data():
      sequences = {
    ">AMP1": "EYHLMNGANGYLTRVNGKTVYRVTKDPVSAVFGVISNCWGSAGAGFGPQH",
    ">AMP2": "KLESIGYLYEPLSEEYRRVIDFSDMKNLRSMFNKITIHVSDKCIQVNKGYLSDFVTSLIRLSDSDINTYDSFDITYIDPRRHITWNNILSILNEK",
    ">AMP3": "MLRRKPTRLELKLDDTEEFESVKKELESRKKQRDEVDVVGVATSSEMSGAAGGTADGKTREQMIHERIGYKPHPKPNTLPSLFGNLQF",
    ">AMP4": "GSSFLSPEHQRVQQRKESKKPPAKLQPR"

    }
      return sequences
 
examples_data()

loaded_models = joblib.load('all_models.pkl')


def accuracy(datas):
      dataset = solve(datas)
      modeles = []
      predicti = []
      for model_name, model in loaded_models.items():
          predictions = model.predict(dataset)
   
          modeles.append(model_name)
          predicti.append(predictions)
      out = answer(predicti)
      return out
  
  
  
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/analyze', methods=['POST'])
def analyze():
    
    data = request.get_json()
    
    user_input = data.get('userInput')
    sol = accuracy(user_input)

    
    return jsonify(sol)


if __name__ == '__main__':
    app.run(debug=True)
  