import math
from scipy.stats import binom
import pandas as pd
import numpy as np
import os 
import plotly.express as px
import plotly.graph_objects as go # needs pip install --upgrade "kaleido==0.1.*" to fix export bug

dir_path = os.path.dirname(os.path.realpath(__file__))


#parameters

#min pulls
minCt = 1
#max pulls 5
maxCt = 100

#group,quantity, total released
data = { 
'common':[0,11,13000],
'uncommon':[1,7,12000],
'rare':[2,3,9000],
'super rare':[2,2,6000],
'ultra rare':[2,1,2500],
}

#graph params
height = 1200
width = 1200
upscale = 3
markerSize = 5
lineWidth = 0.5


#calculate drop rates via bionomial distribution
dataF = pd.DataFrame.from_dict(data).T
dataF.columns=['pack group','quantity','total issued']

def calcProb(df,row):
    group= row['pack group']
    groupdata = df[df['pack group']==group]
    totalItems = (groupdata['quantity']*groupdata['total issued']).sum()
    row['total items in group']=totalItems
    row['probability']=row['total issued']/totalItems*100
    return(row)
dataF = dataF.apply(lambda row: calcProb(dataF,row),axis=1)

calculatedPulls=pd.DataFrame()

for row in dataF.iterrows():
    rarityPulls=[]
    for pack in range(minCt,maxCt+1):
        k = 0
        p = row[1]['probability']/100
        n = pack
        mean, var = binom.stats(n, p)
        dist = binom.pmf(k, n, p)
        rarityPulls.append((1-dist)*100)
    if len(calculatedPulls)==0:
        calculatedPulls = pd.DataFrame(rarityPulls,columns = [row[0]])
    else:
        calculatedPulls[row[0]]=rarityPulls
calculatedPulls.index =np.linspace(minCt, maxCt, num=(maxCt-minCt+1)).astype('int')
calculatedPulls.index.name = 'packs'

calculatedPulls.to_csv(os.path.join(dir_path,'min_prob_stats.csv'))

fig = go.Figure()
for col in calculatedPulls.columns:
    x=calculatedPulls.index
    y=calculatedPulls[col].values
    fig.add_trace(go.Scatter(x=x, y=y,mode='markers+lines',name=col,marker_size = markerSize,line_width=lineWidth))

fig.update_traces(textposition='top left',textfont_size=6)
fig.update_layout(
title ='Minimum drop rate calculated by binomial distribution',
legend_title ='rarity',
xaxis_title ='packs',
yaxis_title ='min drop chance (%)', 
height = height,
width = width,
)

#fig.show()
fig.write_image(os.path.join(dir_path,"Minimum_dropRate_binomial_dist.png"),scale=upscale)
print('done! all figures saved to '+ dir_path)