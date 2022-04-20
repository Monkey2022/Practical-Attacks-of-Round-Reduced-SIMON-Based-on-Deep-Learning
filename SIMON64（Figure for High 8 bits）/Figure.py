# -*- coding: utf-8 -*-
"""

@author: deeplearning

The code is related to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning', which is submitted to The Computer Journal.
The code is also related to 'Improve Neural Distinguisher for Cryptanalysis' (https://eprint.iacr.org/2021/1017).
If you want to use the code, please refer to 'Practical Attacks of Round-Reduced SIMON Based on Deep Learning' or 'Improve Neural Distinguisher for Cryptanalysis'.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from math import ceil,floor

high_bits=8


Wrong_bit=5

Data= np.load('Result_wrongbit_'+str(Wrong_bit)+'.npy')

x=np.arange(len(Data[0]))
x=x+1

figsize = 20,11
#fig=plt.figure()
plt.subplots(figsize=figsize)

ax = plt.subplot(1, 1, 1)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 35,
'style':'oblique',
}
plt.xlabel('$\Delta$',fontdict = {'family' : 'Times New Roman',
                                  'weight' : 'normal',
                                  'size' : 35,
                                  'style':'oblique',
                                  })
plt.ylabel('$Score$',fontdict = {'family' : 'Times New Roman',
                                 'weight' : 'normal',
                                 'size' : 30,
                                 'style':'oblique',
                                 })
ax.yaxis.set_major_locator(MultipleLocator(int((ceil(max(Data.flatten()))-floor(min(Data.flatten())))/16)))
ax.xaxis.set_major_locator(MultipleLocator(16))
plt.ylim(floor(min(Data.flatten())),ceil(max(Data.flatten())))
plt.xlim(0,2**high_bits)

#fig.autofmt_xdate()

frame = plt.gca()
#frame.axes.get_yaxis().set_visible(False)#取消y轴显示
Plot_Parameter={
    'marker':['v','|','d','*','o','p','x','^','.','8','s'],
    'color':['orange','indigo','green','red','lime','navy','tan','skyblue'],
    'linestyle':['-','--',':','-.','-','--',':','-.'],
    'Test':['Test0','Test1','Test2','Test3','Test4','Test5']
    }

for i in range(len(Data)):
    plt.plot(x,Data[i], marker=Plot_Parameter['marker'][i],markersize=6,color=Plot_Parameter['color'][i], linewidth=2,linestyle=Plot_Parameter['linestyle'][i]) # 绘制，指定颜色、标签、线宽，标签采用latex格式


plt.tick_params(labelsize=23) 
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23,}
plt.legend(Plot_Parameter['Test'][:len(Data)],prop=font1,bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)
#print labels [label.set_fontname('Times New Roman') for label in labels]
plt.title('Number of wrong bits (low 24 bits)='+str(Wrong_bit),fontdict={'family': 'Times New Roman','weight':'normal','size': 35,'style':'oblique',})
plt.grid()
#plt.savefig('Number of wrong bits(low 24 bits)='+str(Wrong_bit)+'.jpg', bbox_inches='tight',dpi=1000) # 
plt.savefig('Number of wrong bits(low 24 bits)='+str(Wrong_bit)+'.pdf', bbox_inches='tight',dpi=1000) # 
plt.show()