__author__ = 'jeremy'
import os
import json
import pprint

import numpy as np
import matplotlib.pyplot as plt


def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    thanks to jesshamrick http://www.jesshamrick.com/2012/09/03/saving-figures-from-pyplot/
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.

    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.

    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.

    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.

    """

    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath)

    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")


def autolabel(rects):
    # attach some text labels
    fig, ax = plt.subplots()
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height, '%d' % int(height),
                ha='center', va='bottom')


def midlabel(rect, value, text):
    # attach some text labels
    fig, ax = plt.subplots()
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., height / 2, str(text) + str(value),
            ha='center', va='bottom', weight='bold')

    results_file = 'results.txt'  # change this to include date maybe
    results_html_file = results_file + '.html'
    my_data = json.loads(open(results_file).read())
    pprint(my_data)

    # with open('classifier_results1.txt') as json_data:
    # d = json.load(json_data)
    # pprint(d)
    combined_score = [0, 0, 0]
    f = open(results_html_file, 'a')
    # write html file
    f.write('<HTML><HEAD><TITLE>classifier results</TITLE>\n')
    f.write('<BODY text=#999999 vLink=#555588 aLink=#88ff88 link=#8888ff bgColor=#000000>\n ')
    f.write('</HEAD>\n')
    f.write('<table border=\"0\" >\n ')
    fig_number = -1
    f.write('<tr>\n')
    width = 1
    offset = width + .05
    print('result:')
    # draw graph

    i = 1
    ind = np.arange(i)  # the x locations for the groups
    # fig, ax = plt.subplots()
    width = 1
    offset = 0.1
    totMatches = 1
    FalseMatches = 1
    totTargets = 1
    rects1 = ax.bar(ind, totTargets, width, color='b', alpha=0.5)
    rects2 = ax.bar(ind + offset, totMatches, width, color='g', alpha=0.5)
    rects3 = ax.bar(ind + offset * 2, FalseMatches, width, color='r', alpha=0.5)
    title = 'hi'
    ax.set_title(str(title))
    ax.set_ylabel('N')
    ax.set_xticks(ind + width)
    n_categories = len(totTargets)

    # calculate score for classifier (as though classifier were from each category)
    totTargets = 1
    for j in range(0, n_categories):
        bad_score = 0.0
    good_score = 0.0
    final_score = 0.0
    for k in range(0, n_categories):
        if k != j:
            bad_score = bad_score + float(totMatches[k]) / float(totTargets[k]) + float(FalseMatches[k]) / float(
                totTargets[k])
            print('bscore:' + str(bad_score))
        else:
            good_score = float(totMatches[k] - FalseMatches[k]) / float(totTargets[k])
            print('gscore:' + str(good_score))
    final_score = 2 + good_score - bad_score
    print('finalscore:' + str(final_score))
    midlabel(rects2[j], float(int(final_score * 100.0) / 100.0), 'score:\n')
    # ax.get_xaxis().set_ticks([])
    # plt.xlabel('c','das','dsasd')
    ax.set_xticklabels('a', 'b', 'c')
    # ax.set_xticklabels(('v', 'c', 'd') )
    ax.legend((rects1[0], rects2[0], rects3[0]), ('Ntargets', 'Nmatches', 'NFalseMatches'))
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    # plt.show()#
    # input('enter to continue')
    # my_data.close()
    plt.draw()
    # plt.show()
    fig_name = 'theFig'
    save(fig_name, ext="png", close=False, verbose=True)
    # now save in shared dir
    name = 'hello'
    fig_name = "classifier_results/" + name.split("/", 1)[1]
    print('fig name:' + fig_name)
    save(fig_name, ext="png", close=True, verbose=True)
    complete_fig_name = fig_name + '.png'

    # add HTML code
    f.write(
        '<td> <IMG style=\"WIDTH: 400px;  BORDER-RIGHT: 0px solid; BORDER-TOP: 0px solid; BORDER-LEFT: 0px solid; BORDER-BOTTOM: 0px solid;\"  src=' + complete_fig_name + '>\n')
    f.write('<br>' + str('hiThere') + ' </A> </td> \n')

    # savefig(fig_name)
    i = i + 1
    if i % 3 == 0:
        f.write('</tr>\n')

    f.write('</html>\n')
    f.close
    use_visual_output = True
    if use_visual_output:
        plt.show()

