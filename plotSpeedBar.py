import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=22) 
matplotlib.rc('ytick', labelsize=22) 
matplotlib.rc('xlabel', labelsize=22) 
matplotlib.rc('ylabel', labelsize=22) 
# Perceptron run 19.221906 sec (no_pca)
# Perceptron run 109.981965 sec (with_pca)
# StochasticGradient run 23.893772 sec (no_pca)
# StochasticGradient run 105.634998 sec (with_pca)
# PassiveAggressive run 103.339974 sec (no_pca)
# PassiveAggressive run 104.998947 sec (with_pca)
# ConfidenceWeighted run 1148.783863 sec (no_pca)
# ConfidenceWeighted run 106.770094 sec (with_pca)




f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

plt.xlabel(u"methodology", fontsize=25)
plt.ylabel(u"seconds for running 2000 instances", fontsize=20)

plt.xticks((0,1,2,3,4,5,6,7),
           ('P', 'P(PCA)',
           'LRSGD', 'LRSGD(PCA)',
           'PA', 'P(PCA)',
           'CW', 'CW(PCA)'))

ax.bar(left = (0,1,2,3,4,5,6,7),
        height = (19.221906,
                  109.981965,
                  23.893772,
                  105.634998,
                  103.339974,
                  104.998947,
                  1148.78386,
                  106.770094),
        width = 0.35, align="center")

ax2.bar(left = (0,1,2,3,4,5,6,7),
        height = (19.221906,
                  109.981965,
                  23.893772,
                  105.634998,
                  103.339974,
                  104.998947,
                  1148.78386,
                  106.770094),
        width = 0.35, align="center")

ax.set_ylim(1000, 1200)  # outliers only
ax2.set_ylim(0, 150)  # most of the data

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off')
ax2.xaxis.tick_bottom()

d = .015
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


plt.show()

