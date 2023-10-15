import pandas as pd
import numpy as np
import regex as re

from decimal import Decimal

from Data_Merger_Pipeline import DataMerger

def load_data():
    web_data_df = pd.read_csv("Data/Loaded_Data/Web_Data.csv")
    keV_index = web_data_df['mass'][web_data_df['mass'].str.contains('keV') == True].index
    web_data_df = web_data_df.drop(keV_index)
    web_data_df.reset_index(drop=True)

    github_data_df = pd.read_csv("Data/Loaded_Data/Github_Data.csv")

    return web_data_df, github_data_df

def convert_data_to_seconds(time):
    if type(time) == type(np.nan):
        return time
    if(time == '-1.0'):
        return round(-1)

    time = '%s'%(time)
    mes = re.findall("['A','M','S','Y','H','D','P','U','N']+", time)[0]
    value = float(time.split(' ')[0][:-len(mes)])

    if(mes == 'S'):
        return value

    if(mes == 'AS'):
        a = 41341373336493000000
        return convert_data_to_seconds('%sMS'%(value/a))

    if(mes == 'P'):
        return convert_data_to_seconds('%sN'%(value/1000))
    if(mes == 'N'):
        return convert_data_to_seconds('%sU'%(value/1000))
    if(mes == 'U'):
        return convert_data_to_seconds('%sMS'%(value/1000))
    if(mes == 'MS'):
        return convert_data_to_seconds('%sS'%(value/1000))

    if(mes == 'M'):
        return convert_data_to_seconds('%sS'%(value*60))
    if(mes == 'H'):
        return convert_data_to_seconds('%sM'%(value*60))
    if(mes == 'D'):
        return convert_data_to_seconds('%sH'%(value*24))
    if(mes == 'Y'):
        return convert_data_to_seconds('%sD'%(value*365))

def clean_data(web_data_df, github_data_df):
    cols_to_keep = ['Z','N','Mass','Half Life']
    data_merger = DataMerger(cols_to_keep)

    data = data_merger.transform(web_data_df, github_data_df)[1:].reset_index(drop=True)


    data['Half Life (Seconds)'] = [convert_data_to_seconds(n) for n in data['Half Life']]
    data = data.drop('Half Life', axis=1)
    #Add Data Here
    custom_unstable1 = generate_unstable_data(data, 30, 35, 60, 100)
    #custom_unstable2 = generate_unstable_data(data, 150, 155, 20, 30) #Data is for min_half-life
    data = pd.concat([custom_unstable1, custom_unstable2, data])

    return data

#eb_data_df, github_data_df = load_data()

#df = clean_data(web_data_df, github_data_df)
def generate_unstable_data(df, min_proton, max_proton, min_neutron, max_neutron):
    z, n = [],[]
    for i in range(min_proton,max_proton+1):
        for j in range(min_neutron,max_neutron+1):
            z.append(i)
            n.append(j)

    test_df = pd.DataFrame({'Z':z, 'N':n})
    test_df['Mass'] = test_df['N'] + test_df['Z']
    test_df['Mass'] = [Decimal(test_df.loc[i,'Z']*0.00054386734)+test_df.loc[i,'Mass'] for i in range(test_df.shape[0])]
    test_df['Mass'] = [float(n) for n in test_df['Mass']]
    min_half_life = min([abs(x) for x in df['Half Life (Seconds)'].dropna()])
    test_df['Half Life (Seconds)'] = min_half_life

    return test_df
