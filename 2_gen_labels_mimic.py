#%%
import base64
from mimetypes import guess_type
import requests
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import re
from ast import literal_eval
from openai import AzureOpenAI

CXR_LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", 
"Lung Opacity", "Edema", "Consolidation", "Pneumonia", 
"Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

ls_label_prompt = ['1_4', '2_1', '2_2'] # prompts to gen label
ls_repo_prompt = ['1_4', '3_1'] # prompts to gen repo
ls_gt_prompt = ['3_1'] # for these prompts, no image attached

def parse_args():
    """
    Parse the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=str, dest='api_key', 
                        default=None, 
                        help="API key for Azure API.")
    parser.add_argument("--endpoint", type=str, dest='endpoint', 
                        default=None, 
                        help="Endpoint for Azure API.")
    parser.add_argument("--m", type=str, 
                        dest='image_dir', default=None, 
                        help="Directory path to image files.")
    parser.add_argument("--i", type=str, 
                        dest="input_dir", default=None, 
                        help="Directory path to input csv(s).")
    parser.add_argument("--o", type=str, 
                        dest='output_dir', default=None, 
                        help="Directory path to output csv(s).")
    parser.add_argument("--p", type=str, 
                        dest="prompt_dir", default=None, 
                        help="Directory path to prompts.")
    parser.add_argument("--type", type=str, default='1', 
                        help="Choose the type of prompts you want to input.")
    parser.add_argument("--label", type=str, default='', 
                        help="Choose the type of labelling.")
    args = parser.parse_known_args()

    return args


# (In) gt_indc_sample.csv + ind_list_sample.csv 
# (Out) gen_findings_sample.csv + gen_imp_sample.csv + err_list.txt : study_id (str), report (str)
class GenLabelAzureGPT:
  def __init__(self, image_dir, api_key, endpoint, input_dir, output_dir, prompt_dir, max_retries=3, prompt='1', labels=CXR_LABELS, num_class=''):
      self.prompt_label = prompt
      self.class_type = num_class
      self.image_dir = image_dir
      self.api_key = api_key
      self.endpoint = endpoint
      self.in_ind_pth = input_dir + 'ind_list_sample.csv'
      self.in_indc_pth = input_dir + 'gt_indc_sample.csv'
      self.in_label_pth = input_dir + 'gt_labels_sample{}.csv'.format(self.class_type)
      self.out_dir = output_dir
      self.out_findings_pth = output_dir + 'gen_findings_sample_{}.csv'.format(self.prompt_label)
      self.out_imp_pth = output_dir + 'gen_imp_sample_{}.csv'.format(self.prompt_label)
      self.out_labels_pth = output_dir + 'gen_labels_sample_{}.csv'.format(self.prompt_label)
      self.max_retries = max_retries
      self.prompt_dir = prompt_dir
      self.labels = labels


  # Function to encode a local image into data URL 
  def encode_image(self, image_path):
      mime_type, _ = guess_type(image_path)
      if mime_type is None:
          mime_type = 'application/octet-stream'

      with open(image_path, "rb") as image_file:
          base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

      return f"data:{mime_type};base64,{base64_encoded_data}"


  def gen_per_request(self, ls_base64_image, ls_pos, ls_neg, api_version='2023-12-01-preview', deployment_name='gpt4v'):
      '''
      Report Generation for each request.
      '''

      client = AzureOpenAI(
          api_key=self.api_key,  
          api_version=api_version,
          base_url=f"{self.endpoint}/openai/deployments/{deployment_name}"
      )

      # user text prompt
      with open(os.path.join(self.prompt_dir, self.prompt_label+'.txt'), 'r') as file:
          prompt = file.read()
      
      ls_content = [
      {
              "type": "text",
              "text": prompt.format(ls_pos, ls_neg) ###
              }
      ]

      # user image prompt
      if self.prompt_label not in ls_gt_prompt:
        for i in range(len(ls_base64_image)):
          ls_content.append(
              {
                  "type": "image_url",
                  "image_url": {
                    "url": ls_base64_image[i]
                  }
                }
          )

      # text+image prompt
      apiresponse = client.chat.completions.with_raw_response.create(
          model=deployment_name,
          messages=[
              {
                  "role": "system", 
                  "content": "You are a professional chest radiologist that reads chest X-ray image."},
              {
                  "role": "user",
                  "content": ls_content
              }
          ],
          max_tokens=4096
      )

      debug_sent = apiresponse.http_request.content
      chat_completion = apiresponse.parse()
      
      return chat_completion


  def gen_per_study(self, id, df_label_s, df_ind_s, dict_ind_error):
    ls_pos = []
    ls_neg = []
    ls_pth_img = list(df_ind_s[df_ind_s['study_id'] == id]['path_image'])

    # Prepare encoded images
    ls_base64_image = []
    for pth in ls_pth_img:
        base64_image = self.encode_image(self.image_dir+pth)
        ls_base64_image.append(base64_image)

    # Prepare dict of labels
    for key in CXR_LABELS:
        value = df_label_s[df_label_s['study_id'] == id][key]

        if not pd.isna(value).any():
            if int(value) == 1:
                ls_pos.append(key)
            elif int(value) == 0:
                ls_neg.append(key)

    # Generate reports with max_retries = 3
    temp_repo = ''
    for n in range(self.max_retries):
      temp_resp = self.gen_per_request(ls_base64_image, ls_pos, ls_neg)
      try:
        temp_repo = temp_resp.choices[0].message.content
      except KeyError as e:
        if n == self.max_retries-1:
          print(f's{id} still encounters request rejection after {n+1} tries:', e)
          dict_ind_error[id] = temp_resp
      
      if '<REPORT>' in temp_repo or '<LABEL>' in temp_repo:
        break
  
    # Save temp results and update result dataframe for each request
    if not os.path.exists(self.out_dir + f'cache/{self.prompt_label}'):
      os.makedirs(self.out_dir + f'cache/{self.prompt_label}', exist_ok=True)

    with open(self.out_dir + f'cache/{self.prompt_label}/s{id}.txt', 'w') as file:
      file.write(temp_repo)

    return n, temp_repo, dict_ind_error



  def label_split(self, repo, id, dict_ind_error, df_labels):
    match = re.search(r"<LABEL>\s*(\[.*?\])\s*</LABEL>", repo, re.DOTALL)
    if match:
        data_str = match.group(1)
        data_list = literal_eval(data_str)

        conditions_dict = {condition: [value] for condition, value in data_list}
        df = pd.DataFrame(conditions_dict)

        df['study_id'] = id

        df_labels = pd.concat([df_labels, df])
    else:
        print(f's{id} fails to generate LABEL.')
        dict_ind_error[id] = 'Format'
    
    return df_labels, dict_ind_error


  def repo_split(self, repo, id, dict_ind_error, df_gen_findings_s, df_gen_imp_s):
    match = re.search(r"<REPORT>(.*?)</REPORT>", repo, re.DOTALL)
    df_gen_findings = pd.DataFrame(columns=['study_id', 'report'], index=[0])
    df_gen_imp = pd.DataFrame(columns=['study_id', 'report'], index=[0])
    if match:
        repo_str = match.group(1)
        try:
            if "FINDINGS:" in repo:
                temp_repo = repo_str.split("FINDINGS:")[1].split("IMPRESSION:")
                repo_imp = temp_repo[1].strip()
                cleaned_imp = " ".join(repo_imp.split())
                repo_findings = temp_repo[0].strip()
                cleaned_findings = " ".join(repo_findings.split())
            elif "IMPRESSION:" in repo:
                temp_repo = repo_str.split("IMPRESSION:")[1]
                repo_imp = temp_repo.strip()
                cleaned_imp = " ".join(repo_imp.split())
                cleaned_findings = np.nan
            else:
                cleaned_findings = np.nan
                cleaned_imp = np.nan 
        except IndexError as i:
            print(f's{id} encounters format error:', i)
            dict_ind_error[id] = i
            cleaned_findings = np.nan
            cleaned_imp = np.nan
        
        df_gen_findings['study_id'] = id
        df_gen_findings['report'] = cleaned_findings
        df_gen_imp['study_id'] = id
        df_gen_imp['report'] = cleaned_imp

        df_gen_findings_s = pd.concat([df_gen_findings_s, df_gen_findings])
        df_gen_imp_s = pd.concat([df_gen_imp_s, df_gen_imp])
    else:
        print(f's{id} fails to generate REPORT.')
        dict_ind_error[id] = 'Format'

    return df_gen_findings_s, df_gen_imp_s, dict_ind_error

  

  def run(self):
    '''
    Input Data Files:
    - ind_list_sample.csv: must contain 'study_id' and 'path_image'
    - gt_labels_sample.csv: must contain 'study_id' and 14 CXR conditions (1|0|-1|2)
    
    Output Data Files:
    - err_list_{self.prompt_label}.csv: 'study_id', 'error'
    - gen_findings_sample_{self.prompt_label}.csv: 'study_id', 'report'
    - gen_imp_sample_{self.prompt_label}.csv: 'study_id', 'report'
    - gen_labels_sample_{self.prompt_label}.csv: 'study_id', 14 CXR conditions
    '''
    # 1. read data files
    df_ind_s = pd.read_csv(self.in_ind_pth).sort_values(by='study_id').reset_index(drop=True)
    ls_id = list(np.unique(df_ind_s['study_id']))
    df_label_s = pd.read_csv(self.in_label_pth).sort_values(by='study_id').reset_index(drop=True)

    # 2. prepare empty dataframes and dicts
    df_labels = pd.DataFrame(columns=self.labels)
    df_gen_findings_s = pd.DataFrame(columns=['study_id', 'report'])
    df_gen_imp_s = pd.DataFrame(columns=['study_id', 'report'])
    dict_ind_error = {}

    # 3. generate and process reports
    for i in tqdm(range(len(ls_id))):
      id = ls_id[i]
      n, repo_gen, dict_ind_error = self.gen_per_study(id, df_label_s, df_ind_s, dict_ind_error)

      if n == self.max_retries-1:
        if '<LABEL>' not in repo_gen or '<LABEL>' not in repo_gen:
          if id not in dict_ind_error:
            print(f's{id} still encounters request rejection after {n+1} tries: APIError')
            dict_ind_error[id] = 'APIError'
            continue

      if self.prompt_label in ls_label_prompt:
        df_labels, dict_ind_error = self.label_split(repo_gen, id, dict_ind_error, df_labels)
      if self.prompt_label in ls_repo_prompt:
        df_gen_findings_s, df_gen_imp_s, dict_ind_error = self.repo_split(repo_gen, id, dict_ind_error, df_gen_findings_s, df_gen_imp_s)
      
    # 4. save error information
    temp_tuples = list(dict_ind_error.items())
    df_ind_err = pd.DataFrame(temp_tuples, columns=['study_id', 'error'])
    df_ind_err.to_csv(self.out_dir+f'err_list_{self.prompt_label}.csv', index=False)

    # 5. save generated results
    if self.prompt_label in ls_label_prompt:
      df_labels['study_id'] = df_labels['study_id'].apply(int).apply(str)
      df_labels.to_csv(self.out_labels_pth, index=False)    
    if self.prompt_label in ls_repo_prompt:
      df_gen_findings_s['study_id'] = df_gen_findings_s['study_id'].apply(str)
      df_gen_imp_s['study_id'] = df_gen_imp_s['study_id'].apply(str)
      df_gen_findings_s.to_csv(self.out_findings_pth, index=False)
      df_gen_imp_s.to_csv(self.out_imp_pth, index=False)            

    

if __name__ == '__main__':
  args, _ = parse_args()

  GenRepo = GenLabelAzureGPT(image_dir=args.image_dir, 
                            api_key=args.api_key,
                            endpoint=args.endpoint, 
                            input_dir=args.input_dir,
                            output_dir=args.output_dir, 
                            prompt_dir=args.prompt_dir, 
                            prompt=args.type, 
                            max_retries=3, 
                            num_class=args.label)
  GenRepo.run()