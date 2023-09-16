#modules
import pandas as pd
import numpy as np
import random
from numpy import random as rand
import os 
import plotly.express as px
import plotly.graph_objects as go # needs pip install --upgrade "kaleido==0.1.*" to fix export bug

dir_path = os.path.dirname(os.path.realpath(__file__))


#parameters
ultraRare =[1] #UR IDS
superRare =[2,24] #SR IDS
rare =[7,12,15] #R IDS
uncommon =[5,6,10,11,14,17,19] #UC IDS
common =[13,16,18,20,21,22,23,3,4,8,9] #C IDS

#quantities of each rarity
SRDrops = 2500
URDrops = 6000
RDrops = 9000
UCDrops = 12000
CDrops = 13000

#simulate pulls (higher numbers will take longer to calculate but will have a better distribution)
iterations = 500

minCount = 4 #min packs
maxCount = 50 #max packs

# generate packs
packsToSimulate = 41500

#graph params
height = 1200
width = 1200
upscale = 3
markerSize = 5
lineWidth = 0.5

#main simulation code
totalRaritiesDrop  = SRDrops*len(ultraRare)+URDrops*len(superRare)+RDrops*len(rare)

ultraRarePercent = URDrops/totalRaritiesDrop
superRareRatePrecent = SRDrops/totalRaritiesDrop
rarePercent = RDrops/totalRaritiesDrop

#bias random to account for quantities
weights = [superRareRatePrecent]*len(ultraRare)+[ultraRarePercent]*len(superRare)+ [rarePercent]*len(rare)

totalRarities = ultraRare+rare+uncommon+common+superRare
totalRarities.sort()

def pull():
    global common,uncommon,rare,ultraRare
    pulls =[]
    #3 commons
    pulls.append(random.choice(common))
    pulls.append(random.choice(common))
    pulls.append(random.choice(common))
    #2 uncommons
    pulls.append(random.choice(uncommon))
    pulls.append(random.choice(uncommon))
    #1 mixed rare
    pulls.append(*random.choices(ultraRare+superRare+rare,weights=weights))
    return pulls
    
def simulatePacks(packcount):
    packs =[]
    for i in range(0,packcount):
        packs.append(pull())
    df =pd.DataFrame(packs)
    df.columns = 'pull '+(df.columns+1).astype(str)
    return df

#random pool of packs to draw from
print('creating ' + str(iterations)+ ' pack pulls, this may take some time')
simulatedPacks = simulatePacks(packsToSimulate)


#simulate pack pulls
print('Simulating dataset of ' + str(packsToSimulate)+ ' packs')
calculatedPulls = pd.DataFrame(columns=totalRarities)
for count in range(minCount,maxCount+1):
    print('Simulating ' + str(count)+ ' pack pulls')
    unique=[]
    for i in range(0,iterations):
        sample = simulatedPacks.sample(n=count)#grab random sample
        uniqueVals = sample.melt()['value'].value_counts().sort_index()#keep track of quantities pulled for each item type
        minCt = pd.Series(uniqueVals.unique()).min()# minimum quantity pulled from set
        uniqueCt = len(sample.melt()['value'].unique())*minCt/len(totalRarities)#calculate total unique cars from set pulled normalized for packs drawn out of full set
        uniqueVals['completed sets'] = uniqueCt
        unique.append(uniqueVals)
    tempDF = pd.DataFrame(unique,columns = totalRarities+['completed sets'])
    tempDF['packs'] = count
    tempDF['simulations'] = iterations
    calculatedPulls = pd.concat([calculatedPulls,tempDF])

calculatedPulls.reset_index(inplace=True,drop=True)

calculatedPulls.to_csv(os.path.join(dir_path,'simulated_pulls.csv'))

#simulate pull from pack set
pullProbabilities =pd.DataFrame()
for pullSet in calculatedPulls.groupby(['packs']):
    packs =pullSet[0]
    packsPulledDf = pullSet[1].drop(columns=['packs','simulations']).reset_index(drop=True)
    packsContents = packsPulledDf.drop(columns=['completed sets'])
    probability = packsContents*(packsContents.count()/iterations)*100
    probability['completed sets']=packsPulledDf['completed sets']
    stats = probability.describe().T
    stats['packs']=[packs[0]]*len(stats)
    stats = stats.T
    pullProbabilities= pd.concat([pullProbabilities,stats])

pullProbabilities.to_csv(os.path.join(dir_path,'simulated_pulls_stats.csv'))


#graphing
pallette =px.colors.qualitative.Alphabet

figMin = go.Figure()
figMean = go.Figure()
for i,col in enumerate(pullProbabilities.drop(columns=['completed sets']).columns):
    subset = pullProbabilities[col]
    x=subset.T['packs'].values
    min=subset.T['min'].values
    mean=subset.T['mean'].values
    figMin.add_trace(go.Scatter(x=x, y=min,mode='markers+lines',name=col,marker_size = markerSize,line_width = lineWidth,line_color=pallette[i]))
    figMean.add_trace(go.Scatter(x=x, y=mean,mode='markers+lines',name=col,marker_size = markerSize,line_width = lineWidth,line_color=pallette[i]))


figSets = go.Figure()
subset = pullProbabilities['completed sets']
x=subset.T['packs'].values
min=subset.T['min'].values
mean=subset.T['mean'].values

figSets.add_trace(go.Scatter(x=x, y=min,mode='markers+lines',name='min',marker_size = markerSize,line_width = lineWidth,line_color='red'))
figSets.add_trace(go.Scatter(x=x, y=mean,mode='markers+lines',name='mean',marker_size = markerSize,line_width = lineWidth,line_color='green'))

figSets.update_traces(textposition='top left',textfont_size=6)
figSets.update_layout(
title ='Set completion (n = '+str(iterations)+')',
legend_title ='parameter',
xaxis_title ='packs',
yaxis_title ='completed sets', 
height = height,
width = width,
)

#figSets.show()
figSets.write_image(os.path.join(dir_path,"NFT_Sets.png"),scale=upscale)


figMin.update_traces(textposition='top left',textfont_size=6)
figMin.update_layout(
title ='Minimum drop rate (n = '+str(iterations)+')',
legend_title ='car id',
xaxis_title ='packs',
yaxis_title ='drop chance (%)', 
height = height,
width = width,
)

#figMin.show()
figMin.update_layout(yaxis_range=[0,110])
figMin.write_image(os.path.join(dir_path,"MinDropRate.png"),scale=upscale)

figMean.update_traces(textposition='top left',textfont_size=6)
figMean.update_layout(
title ='Mean drop rate (n = '+str(iterations)+')',
legend_title ='car id',
xaxis_title ='packs',
yaxis_title ='drop chance (%)', 
height = height,
width = width,
)

#figMean.show()
figMean.write_image(os.path.join(dir_path,"meanDropRate.png"),scale=upscale)

print('done! all figures saved to '+ dir_path)