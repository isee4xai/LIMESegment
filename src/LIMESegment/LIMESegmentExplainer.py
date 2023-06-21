
import matplotlib.pyplot as plt
import numpy as np

from .Utils.explanations import LIMESegment


class LIMESegmentExplainer():

    def _init__():
        pass

    def explain (self, example, model, model_type='class', distance='dtw', 
                 n=100, window_size=None, cp=None, f=None):
        
        return LIMESegment( example =example,
            model = model,
            model_type = model_type,
            distance = distance,
            n = n,
            window_size = window_size,
            cp=cp,
            f=f

            )
    
    def plot_explanation (self, instance, explanation, ax=None):

        if ax is None:
            _, ax = plt.subplots()
        coeffs, segment_indexes = explanation
        i=1

        p=segment_indexes[i]
        maximp=max(np.max(coeffs),1)
        minimp=min(np.min(coeffs),0)
        scale=maximp-minimp

        while p!=-1:
            if coeffs[i-1]>=0:
                ax.axvspan(segment_indexes[i-1], segment_indexes[i], color='green', alpha=(coeffs[i-1]/scale), lw=0)
            else:
                ax.axvspan(segment_indexes[i-1], segment_indexes[i], color='red', alpha=abs(coeffs[i-1])/scale, lw=0)
            i=i+1
            p=segment_indexes[i]

        if coeffs[i-1]>=0:
            ax.axvspan(segment_indexes[i-1], len(instance.flatten())-1, color='green', alpha=(coeffs[i-1]/scale), lw=0)
        else:
            ax.axvspan(segment_indexes[i-1], len(instance.flatten())-1, color='red', alpha=abs(coeffs[i-1])/scale, lw=0)
            
        ax.plot(range(len(instance.flatten())), instance.flatten(), color='b',lw=1.5)
        ax.set_xticks(list(plt.xticks()[0]) + [len(instance.flatten())-1])

        ax.scatter(segment_indexes[1:-1], [instance.flatten()[idx] for idx in segment_indexes[1:-1]], color='blue',marker="o") 
        for cp in segment_indexes[1:-1]:
            ax.axvline(x=cp, color='black',linestyle="--",lw=1, label='axvline - full height')
            
        ax.set_title("Attributions for Class")
    
