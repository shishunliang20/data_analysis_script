import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import pyplot
import numpy as np 
from matplotlib.pyplot import MultipleLocator, subplot
import numpy as np 
import math
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import numpy as np 
import math
from matplotlib.ticker import MaxNLocator

od =  pd.read_csv('/Users/liangshishun/Library/CloudStorage/OneDrive-ImperialCollegeLondon/phd/Data/biosensor/230504_wt_tnac_4c8m_od.csv')
gfp = pd.read_csv('/Users/liangshishun/Library/CloudStorage/OneDrive-ImperialCollegeLondon/phd/Data/biosensor/230504_wt_tnac_4c8m_gfp.csv')



# od = od.iloc[7:103, 3:32]
# gfp = gfp.iloc[7:103, 3:32]

od = od.iloc[6:103, 3:99]
gfp = gfp.iloc[6:103, 3:99]

# od.columns = column_name

# print(od)
#convert the od to numeric type
for i in range (len (od.iloc[0])):
    od.iloc[i] = od.iloc[i].astype('float')

for i in range (len (gfp.iloc[0])):
    gfp.iloc[i] = gfp.iloc[i].astype('float')

def Gather_1_sample_mean(start_row,duplicate):# caculate the mean
    data1 = od.iloc[start_row]
    data2 = od.iloc[start_row + 1]
    data3 = od.iloc[start_row + 2]

    data4 = gfp.iloc[start_row]
    data5 = gfp.iloc[start_row + 1]
    data6 = gfp.iloc[start_row + 2]
    print (data1,data2,data3)
    # sample_data = pd.DataFrame([data1,data2,data3])
    # sample_data = pd.DataFrame([data1])

    sample_data_od = od.iloc[range(start_row,start_row + duplicate)].astype('float')
    sample_data_gfp = gfp.iloc[range(start_row,start_row + duplicate)].astype('float')

    # print(sample_data)
    mean_od = []
    mean_gfp = []
    mean = []
    for i in range (len(sample_data_od.iloc[0])):
         mean_od.append(sample_data_od.iloc[:,i].mean())
         mean_gfp.append(sample_data_gfp.iloc[:,i].mean())
         mean.append(sample_data_gfp.iloc[:,i].mean()/sample_data_od.iloc[:,i].mean())

    return mean

def Gather_1_sample_std(start_row,duplicate):# caculae the std
    
    data1 = od.iloc[start_row]
    data2 = od.iloc[start_row + 1]
    data3 = od.iloc[start_row + 2]
    
    # print (data1,data2,data3)
    sample_data = pd.DataFrame([data1,data2,data3])
    # sample_data = pd.DataFrame([data1])
    
    std = []
    for i in range (len(sample_data.iloc[0])):
         std.append(sample_data.iloc[:,i].std())
    return std

plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

    
media_type = {0:'indole',1:'chorismate',2:'phenylalanine',3:'phenylpyruvate',4:'trytophan',5:'tyrosine',6:'shikimate',7:'no_m'}
# media_type = {0:'indole',1:'phenylalanine',2:'phenylpyruvate',3:'tyrosine',4:'no_m'}

def X_axis (total_cycle):
    hour = total_cycle * 15 / 60
    x_axis = np.arange(0,hour,0.25)
    # x_axis = (np.arange(0,12.5,0.25))

    return x_axis


def Plot_setting(start_row,duplicate,condition):
    #fig=plt.figure(figsize=(20,10))
    # iters=list(range(29))
    iters=list(range(96))

    color=palette(condition)
    # ax=fig.add_subplot(2,4,condition+1) # the map of the plot  
    
    avg= Gather_1_sample_mean(start_row,duplicate)
    # std= Gather_1_sample_std(start_row,duplicate)

    # r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    # r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    media_type = {0:'indole',1:'chorismate',2:'phenylalanine',3:'phenylpyruvate',4:'trytophan',5:'tyrosine',6:'shikimate',7:'no_m'}
    # media_type = {0:'indole',1:'phenylalanine',2:'phenylpyruvate',3:'tyrosine',4:'no_m'}

    ax.plot(iters, avg, color=color,label=media_type[condition],linewidth=3.0)
    # ax.fill_between(iters, r1, r2, color=color, alpha=0.2)


    #ax.legend(loc='lower right',prop=font1,fontsize = 2)
    ax.legend(fontsize = 8)
    # ax.set_xlabel('time/hour',fontsize=12,labelpad=0)
    ax.set_ylabel('FI/OD',fontsize=15)
    # plt.ylim((0,10000))

    total_cycle = 96
    # total_cycle = 29


    a = X_axis(total_cycle)
    plt.xticks(iters,a)
    x_major_locator = MultipleLocator(8) #x axis step length
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


fig,ax= plt.subplots()
ax =  fig.add_subplot(331) 


start_pos = 0  # change
duplicate = 3
metabolite_num = 8
# 4 24 28 48 52 72 76
for condition in range(len(media_type)):
    start_row = start_pos + condition * duplicate
    print(start_row)
    Plot_setting(start_row,duplicate,condition)
name = 'fructose' # change
plt.title(name,fontsize=15)

ax =  fig.add_subplot(332) 

start_pos = 0 + metabolite_num * duplicate* 1 # change
duplicate = 3
# 4 24 28 48 52 72 76
for condition in range(len(media_type)):
    start_row = start_pos + condition * duplicate

    Plot_setting(start_row,duplicate,condition)
name = 'glucose' # change
plt.title(name,fontsize=15)


ax =  fig.add_subplot(333) 

start_pos = 0 + metabolite_num * duplicate* 2 # change
duplicate = 3
# 4 24 28 48 52 72 76
for condition in range(len(media_type)):
    start_row = start_pos + condition * duplicate

    Plot_setting(start_row,duplicate,condition)
name = 'glycerol' # change
plt.title(name,fontsize=15)

ax =  fig.add_subplot(334) 

start_pos = 0 + metabolite_num * duplicate* 3 # change
duplicate = 3
# 4 24 28 48 52 72 76
for condition in range(len(media_type)):
    start_row = start_pos + condition * duplicate

    Plot_setting(start_row,duplicate,condition)
name = 'maltose' # change
plt.title(name,fontsize=15)

# ax =  fig.add_subplot(336) 

# start_pos = 0 + metabolite_num * duplicate* 4 # change
# duplicate = 3
# # 4 24 28 48 52 72 76
# for condition in range(len(media_type)):
#     start_row = start_pos + condition * duplicate

#     Plot_setting(start_row,duplicate,condition)
# name = 'wt glucose' # change
# plt.title(name,fontsize=15)

plt.show()
# plt.savefig('{}.png'.format(name))    
