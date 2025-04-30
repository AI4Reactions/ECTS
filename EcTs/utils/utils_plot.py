import numpy as np
import copy
from tqdm import tqdm 

def Define_box_grid_figure(datadict,figtext={},size=(12,12),filename='plot.png',wspace=0.3,hspace=0.2,dpi=300,grids=None,grids_arrange=True):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import numpy as np

    def set_ax_frame(ax):
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        return

    clist='abcdefghijklmnopqrstuvwxyz'
    colors=['red','green','orange','blue','pink','purple','yellow']
    plt.rc('font',size=12)
    plt.rcParams['xtick.direction']='in'
    plt.rcParams['ytick.direction']='in'
    figure=plt.figure(figsize=size)

    if grids_arrange:
        xgrids=grids[0]
        ygrids=grids[1]
        gs=gridspec.GridSpec(xgrids,ygrids)

    ndatas=len(datadict.keys())
    for pid,key in enumerate(datadict.keys()):
        print (key,'!'*20)

        if grids_arrange:
            xid=pid//ygrids
            yid=pid%ygrids
            ax=plt.subplot(gs[xid,yid])
        else:
            assert 'subplot' in datadict[key].keys() ,"subplot should be specified when grids_arrange==False"
            ax=plt.subplot(*datadict[key]['subplot'])

        subdatadict=datadict[key]
        if 'xlabel' in datadict[key].keys():
            xlabel=datadict[key]['xlabel']
        else:
            xlabel=''

        if 'ylabel' in datadict[key].keys():
            ylabel=datadict[key]['ylabel']
        else:
            ylabel=''
        style=datadict[key]['style']

        if 'xlim' in datadict[key].keys():
            xlim=datadict[key]['xlim']
        else:
            xlim=None
        if 'ylim' in datadict[key].keys():
            ylim=datadict[key]['ylim']
        else:
            ylim=None
        if 'xticks' in datadict[key].keys():
            xticks=np.arrange(*datadict[key]['xticks'])
        else:
            xticks=None

        if 'yticks' in datadict[key].keys():
            yticks=np.arrange(*datadict[key]['yticks'])
        else:
            yticks=None 

        if "y_ticks" in datadict[key].keys():
            y_ticks=datadict[key]["y_ticks"]
        else:
            y_ticks=None

        if 'y_ticklabels' in datadict[key].keys():
            y_ticklabels=datadict[key]['y_ticklabels']
        else:
            y_ticklabels=None

        if 'ylabel' in datadict[key].keys():
            ylabel=datadict[key]['ylabel']
        else:
            ylabel=''
        if 'xlabel' in datadict[key].keys():
            xlabel=datadict[key]['xlabel']
        else:
            xlabel=''
        if 'legend' in datadict[key].keys():
            iflegend=datadict[key]['legend']
        else:
            iflegend=False
        if 'xticks_rotation' in datadict[key].keys():
            xticks_rotation=datadict[key]['xticks_rotation']
        else:
            xticks_rotation=0
        if 'yticks_rotation' in datadict[key].keys():
            yticks_rotation=datadict[key]['yticks_rotation']
        else:
            yticks_rotation=0
        if "markersize" in datadict[key].keys():
            markersize=datadict[key]["markersize"]
        else:
            markersize=8

        char=clist[pid]
        for did,subkey in tqdm(enumerate(subdatadict['data'].keys())):
            if 'pointline' in style:
                plt.plot(subdatadict['data'][subkey][0],subdatadict['data'][subkey][1],label=subkey,color=colors[did],marker='o',linewidth=2,markersize=markersize)
            if 'scatter' in style:
                plt.plot(subdatadict['data'][subkey][0],subdatadict['data'][subkey][1],label=subkey,color=colors[did],marker='o',markersize=markersize)
            if 'regplot' in style:
                sns.regplot(ax=ax,x=subdatadict['data'][subkey][0],y=subdatadict['data'][subkey][1],label=subkey,color=colors[did],scatter=True)
            if 'distplot' in style:
                sns.histplot(subdatadict['data'][subkey],ax=ax,bins=50,label=subkey,color=colors[did],kde=True,legend=True)
                #sns.histplot(subdatadict['data'][subkey],ax=ax,bins=50,label=subkey)

            if 'bar'==style:
                plt.bar(subdatadict['data'][subkey][0]+(did-1)*subdatadict['bar_width'],subdatadict['data'][subkey][1],subdatadict['bar_width'],align="edge",label=subkey)
                for i in subdatadict['data'][subkey][0]:
                    plt.text(i+0.5*(did)-0.2,
                            subdatadict['data'][subkey][1][i]+0.5,
                            f"{subdatadict['data'][subkey][1][i]:.1f}%",
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            )
            if 'barh'==style:
                plt.barh(subdatadict['data'][subkey][0]+(did-1)*subdatadict['bar_width'],subdatadict['data'][subkey][1],subdatadict['bar_width'],align="edge",label=subkey)
                for i in subdatadict['data'][subkey][0]:
                    plt.text(
                            subdatadict['data'][subkey][1][i-1]+7.5,
                            i+subdatadict["bar_width"]*(did-1),
                            f"{subdatadict['data'][subkey][1][i-1]:.1f}%",
                            ha='center',
                            va='bottom',
                            fontsize=10,
                            )
                
        if 'boxplot' == style:
            if 'hue' in subdatadict['data'].keys():
                sns.boxplot(y=subdatadict['data']['label'],x=subdatadict['data']['value'],hue=subdatadict['data']['hue'],ax=ax,showfliers=False,showcaps=True,notch=True,showmeans=True,meanline=True,meanprops={'linewidth':1.5})
                print ('Here2')
            else:
                sns.boxplot(y=subdatadict['data']['label'],x=subdatadict['data']['value'],ax=ax,showfliers=False,showcaps=True,notch=True,showmeans=True,meanline=True,meanprops={'linewidth':3})

        if "heatmap" == style:
            sns.heatmap(
                subdatadict['data']["pivot"], 
                annot=subdatadict['data']["annot"],
                annot_kws=subdatadict['data']["annot_kws"],
                fmt="", 
                cmap='viridis', 
                cbar_kws={'label':subdatadict['cbar_label']},
                cbar=True,
                ax=ax,
                linewidths=0.5,        # 设置单元格之间的线宽
                linecolor='white'      # 设置单元格之间的线颜色
                )
            
        #if iflegend:
        leg=plt.legend(fancybox=True,framealpha=0,fontsize=12,markerscale=0.5)

        if xlim:
            plt.xlim(*xlim)
        if ylim:
            plt.ylim(*ylim)

        if y_ticklabels:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticklabels,fontsize=12)

        plt.tick_params(length=5,top=True,bottom=True,left=True,right=True)

        plt.xlabel(xlabel,fontsize=16)

        plt.ylabel(ylabel,fontsize=16)

        if 'text' in datadict[key].keys():
            for textkey in datadict[key]['text'].keys():
                plt.text(datadict[key]['text'][textkey][0],datadict[key]['text'][textkey][1],s=textkey,transform=ax.transAxes, fontsize=12)

        #plt.text(0.1,0.85,s=f'({clist[pid]})',transform=ax.transAxes,fontsize=12)
        
        set_ax_frame(ax)

    for key in figtext.keys():
        pos=figtext[key]
        figure.text(pos[0],pos[1],key,fontsize=16,rotation=pos[2])
    #figure.tight_layout()
    plt.subplots_adjust(wspace=wspace,hspace=hspace)
    plt.savefig(filename,dpi=dpi,bbox_inches='tight')
    #plt.show()
    #figure.savefig(filename,format='svg',dpi=300)