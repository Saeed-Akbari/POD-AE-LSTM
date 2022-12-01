import os, os.path

import matplotlib as matplt
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors

import numpy as np

def contourSubPlot(X, Y, phi1, phi2, phi3, phi4, phi5, phi6, time, figTimeTest,\
                    testStartTime, barRange, fileName='filename', figSize=(14,7)):

    fig, axs = plt.subplots(3,2,figsize=figSize)
    phi = [phi1, phi2, phi3, phi4, phi5, phi6]
    i = 0
    for ax in axs.flat:
        cntr1 = ax.contourf(X, Y, phi[i], barRange, cmap="RdBu_r")
        ax.set_aspect('equal')
        ax.set_ylabel('$y$')
        if i-len(axs.flat) > -3:
            ax.set_xlabel('$x$')
        i = i + 1
    cb1 = fig.colorbar(cntr1, ax=axs, shrink=0.8, orientation='vertical')

    axs[0][0].set_title(r'$t={:.0f}$'.format(time[figTimeTest[0]+testStartTime]))
    axs[0][1].set_title(r'$t={:.0f}$'.format(time[figTimeTest[1]+testStartTime]))
    #fig.colorbar(cs, ax=axs, shrink=0.8, orientation='vertical')
    
    plt.text(-3.9, 2.9, r'\bf{FOM}', va='center',fontsize=18)
    plt.text(-3.85, 1.68, r'\bf{TP}', va='center',fontsize=18)
    plt.text(-4.05, 0.48, r'\bf{NLPOD}', va='center',fontsize=18)

    #fig.tight_layout()
    fig.savefig(fileName, bbox_inches = 'tight', pad_inches = 0.1, dpi = 400)
    fig.clear(True)

def animationGif(X, Y, phi, fileName='filename', figSize=(14,7)):

    fig = plt.figure(figsize=figSize)

    plt.xticks([])
    plt.yticks([])
    
    def animate(i):
        cont = plt.contourf(X,Y,phi[:,:,i],120,cmap='jet')
        return cont  
    
    anim = animation.FuncAnimation(fig, animate, frames=50)
    fig.tight_layout()
    # anim.save('animation.mp4')
    writergif = animation.PillowWriter(fps=10)
    anim.save(fileName+'.gif',writer=writergif)

    fig.clear(True)

def plotPODcontent(RICd, AEmode, dirPlot, myCont):

    fig = plt.figure()
        
    nrplot = 2 * np.min(np.argwhere(RICd>myCont))
    index = np.arange(1,nrplot+1)
    newRICd = [0, *RICd]
    plt.plot(range(len(newRICd)), newRICd, 'k')
    x1 = 1
    x2 = AEmode
    y1 = RICd[x2-1]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    xave = np.exp(0.75*np.log(x1) + 0.25*np.log(x2))
    plt.text(xave, y2+0.4, r'$'+str(np.round(RICd[x2-1],decimals=2))+'\%$',fontsize=10)
    plt.fill_between(index[:x2], RICd[:x2], RICd[0],alpha=0.6,color='orange')

    x1 = AEmode
    x2 = AEmode
    y1 = RICd[0]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    yave = np.exp(0.5*np.log(y1) + 0.5*np.log(y2))
    plt.text(x2+0.1, yave, r'r=$'+str(x2)+'$',fontsize=10)

    x1 = 1
    x2 = np.min(np.argwhere(RICd>myCont))
    y1 = RICd[x2-1]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    xave = np.exp(0.5*np.log(x1) + 0.5*np.log(x2))
    plt.text(xave, y2+0.4, r'$'+str(np.round(RICd[x2-1],decimals=2))+'\%$',fontsize=10)
    plt.fill_between(index[AEmode-1:x2], RICd[AEmode-1:x2], RICd[0],alpha=0.2,color='blue')

    x1 = np.min(np.argwhere(RICd>myCont))
    x2 = np.min(np.argwhere(RICd>myCont))
    y1 = RICd[0]
    y2 = RICd[x2-1]
    plt.plot([x1,x2], [y1,y2], color='k', linestyle='--')
    yave = np.exp(0.5*np.log(y1) + 0.5*np.log(y2))
    plt.text(x2+2, yave, r'r=$'+str(x2)+'$',fontsize=10)
    
    plt.xscale("log")
    plt.xlabel(r'\bf POD index ($k$)')
    plt.ylabel(r'\bf RIC ($\%$)')
    plt.gca().set_ylim([RICd[0], RICd[-1]+3])
    plt.gca().set_xlim([1, x2*2])

    plt.savefig(dirPlot + '/content.pdf', dpi = 500, bbox_inches = 'tight')
    fig.clear(True)
    #fig, ax = plt.subplots()
    #ax.clear()

def plot(figNum, epochs, loss, valLoss, label1, label2, plotTitle, fileName):

    fig = plt.figure(figNum)
    plt.semilogy(epochs, loss, 'b', label=label1)
    plt.semilogy(epochs, valLoss, 'r', label=label2)
    plt.title(plotTitle)
    plt.legend()
    plt.savefig(fileName)
    fig.clear(True)

    fig, ax = plt.subplots()
    ax.clear()

def subplot(figNum, epochs, predData, trueData, label1, label2, label3, label4, plotTitle, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18, 9))
    fig.suptitle(plotTitle)

    i = 0
    for ax in axs.flat:
        ax.plot(epochs, trueData[:, i])
        ax.plot(epochs, predData[:, i])
        #if i%px == 0:
        #    ax.set_ylabel(label4)
        #if i%py == 1:
        #    ax.set_xlabel(label3)
        if i == 0:
            ax.legend([label1, label2], loc=0, prop={'size': 6})
        i = i + 1

    plt.savefig(fileName)
    fig.clear(True)


from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
mpl.rcParams['text.latex.preamble'] = r'\boldmath'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
mpl.rc('font', **font)

def subplotMode(epochsTrain, epochs, trueData, testData,\
                trueLabel, testLabel, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18,14))

    i = 0

    for ax in axs.flat:
        ax.plot(epochs,testData[:, i], label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
        ax.plot(epochs,trueData[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        #ax.plot(epochsTest,trueData[:, i], 'o', markerfacecolor="None", markevery = 4,\
        #        label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        #ax.plot(epochs[ind_m],w[i,:], 'o', fillstyle='none', \
        #        label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2)
        #ax.plot(epochs,ua[i,:], '--', label=r'\bf{Analysis}', linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='y', alpha=0.4, lw=0)
        #if i % 2 == 0:
        #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
        #else:
        #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=-12)
        ax.set_xlim([epochsTrain[0], epochs[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
        i = i + 1

    axs.flat[-1].set_xlabel(r'$t$',fontsize=22)
    #plt.rc('legend', fontsize = 28)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =2,fontsize=38)
    #handles, labels = axs.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,1.25),ncol =2,fontsize=38)
    #legend = axs.flat[0].legend(loc=0, ncol=1, bbox_to_anchor=(0, 0, 1, 1),
    #       prop = fontP,fancybox=True,shadow=False,title='LEGEND')

    #plt.setp(legend.get_title(),fontsize='42')
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def subplotModeAE(epochsTrain, epochsTest, trueProjct, AEdata,\
                trueLabel, AELabel, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18,14))

    i = 0
    
    for ax in axs.flat:
        ax.plot(epochsTest,AEdata[:, i], 'b', label=r'\bf{{{}}}'.format(AELabel), linewidth = 3)
        ax.plot(epochsTest,trueProjct[:, i], 'g-.', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)
        ax.set_xlim([epochsTrain[0], epochsTest[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$R_{}(t)$'.format(i+1), labelpad=5)
        i = i + 1

    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =2,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)

def subplotProbe(epochsTest, epochsTrain, trueData, testData, FOMData,\
                    trueLabel, testLabel, FOMLabel, fileName, px, py, var):

    fig, axs = plt.subplots(px, py, figsize=(18, 14))

    FOMData = FOMData.reshape(FOMData.shape[0], FOMData.shape[1]*FOMData.shape[2])
    testData = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2])
    trueData = trueData.reshape(trueData.shape[0], trueData.shape[1]*trueData.shape[2])

    i = 0
    for ax in axs.flat:
        ax.plot(epochsTest,FOMData[:, i], label=r'\bf{{{}}}'.format(FOMLabel), linewidth = 3)
        ax.plot(epochsTest,testData[:, i], label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
        ax.plot(epochsTest,trueData[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='y', alpha=0.4, lw=0)

        ax.set_xlim([epochsTrain[0], epochsTest[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$\{}$'.format(var)+r'$_{}$'.format(i+1), labelpad=5)
        i = i + 1

    
    axs.flat[-1].set_xlabel(r'$t$',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def plotMode(figNum, epochsTest, epochsTrain, trueData, testData, trainData,\
                trueLabel, testLabel, trainLabel, fileName, px, py):

    fig, ax = plt.subplots(px, py, figsize=(18,8))

    i = 0

    ax.plot(epochsTrain,trainData[:, i], label=r'\bf{{{}}}'.format(trainLabel), linewidth = 3)
    ax.plot(epochsTest,testData[:, i], label=r'\bf{{{}}}'.format(testLabel), linewidth = 3)
    ax.plot(epochsTest,trueData[:, i], ':', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
    #ax.plot(epochsTest,trueData[:, i], 'o', markerfacecolor="None", markevery = 4,\
    #        label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
    #ax.plot(epochs[ind_m],w[i,:], 'o', fillstyle='none', \
    #        label=r'\bf{Observation}', markersize = 8, markeredgewidth = 2)
    #ax.plot(epochs,ua[i,:], '--', label=r'\bf{Analysis}', linewidth = 3)
    ax.axvspan(epochsTrain[0], epochsTrain[-1], color='y', alpha=0.4, lw=0)
    #if i % 2 == 0:
    #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
    #else:
    #    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=-12)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(r'$z_{}(t)$'.format(i+1), labelpad=5)
    i = i + 1

    ax.set_xlabel(r'$t$',fontsize=22)
    ax.legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=15)
    fig.subplots_adjust(hspace=0.5)

    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def subplotModeUnc(epochsTrain, epochs, ymean, yn, yp, yAE,\
                    labelMean, labelSD, labelAE, fileName, px, py):

    fig, axs = plt.subplots(px, py, figsize=(18,14))

    i = 0

    for ax in axs.flat:

        ax.plot(epochs, ymean[:,i],\
                        'r--', label=r'\bf{{{}}}'.format(labelMean), linewidth = 3)

        ax.plot(epochs, yAE[:,i],\
                        'b-', label=r'\bf{{{}}}'.format(labelAE), linewidth = 3)

        ax.plot(epochs, yp[:,i], color='#ff8000', linewidth = 0.1)
        ax.plot(epochs, yn[:,i], color='#ff8000', linewidth = 0.1)

                        
        ax.fill_between(epochs,yn[:,i],yp[:,i],facecolor='#ff8000',\
            alpha=0.5, label=r'\bf{{{}}}'.format(labelSD))

        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)#, color='#DCDCDC'
        ax.set_xlim([epochsTrain[0], epochs[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$r_{}(t)$'.format(i+1), labelpad=5)
        i = i + 1

    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)


def subplotProbeUnc(epochsTest, epochsTrain, ymean, yn, yp, trueData, FOMData,\
                    labelMean, labelSD, trueLabel, FOMLabel, fileName, px, py, var):

    fig, axs = plt.subplots(px, py, figsize=(18, 14))

    ymean = ymean.reshape(ymean.shape[0], ymean.shape[1]*ymean.shape[2])
    yn = yn.reshape(yn.shape[0], yn.shape[1]*yn.shape[2])
    yp = yp.reshape(yp.shape[0], yp.shape[1]*yp.shape[2])
    trueData = trueData.reshape(trueData.shape[0], trueData.shape[1]*trueData.shape[2])
    FOMData = FOMData.reshape(FOMData.shape[0], FOMData.shape[1]*FOMData.shape[2])

    i = 0
    for ax in axs.flat:
        ax.plot(epochsTest,ymean[:, i], 'r--', label=r'\bf{{{}}}'.format(labelMean), linewidth = 3)
        ax.plot(epochsTest,trueData[:, i], 'g-.', label=r'\bf{{{}}}'.format(trueLabel), linewidth = 3)
        ax.plot(epochsTest,FOMData[:, i], 'k-', label=r'\bf{{{}}}'.format(FOMLabel), linewidth = 3)

        ax.plot(epochsTest, yp[:,i], color='#ff8000', linewidth = 0.1)
        ax.plot(epochsTest, yn[:,i], color='#ff8000', linewidth = 0.1)

        ax.fill_between(epochsTest,yn[:,i],yp[:,i],facecolor='#ff8000',\
            alpha=0.5, label=r'\bf{{{}}}'.format(labelSD))
        ax.axvspan(epochsTrain[0], epochsTrain[-1], color='khaki', alpha=0.4, lw=0)

        ax.set_xlim([epochsTrain[0], epochsTest[-1]])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel(r'$\{}$'.format(var)+r'$_{}$'.format(i+1), labelpad=5)
        i = i + 1

    
    axs.flat[-1].set_xlabel('Time (s)',fontsize=22)
    axs.flat[0].legend(loc="center", bbox_to_anchor=(0.5,1.25),ncol =4,fontsize=22)
    fig.subplots_adjust(hspace=0.5)

    fig.tight_layout()
    plt.savefig(fileName, dpi = 500, bbox_inches = 'tight')
    fig.clear(True)
