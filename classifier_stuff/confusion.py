__author__ = 'jeremy'
#confusion matrices - input is list of classifiers
#output is conf matrix
#ideallly with n, n_falsepos,n_truepos etc written on each square
#also add here - ROC and/or precision/accuracy plot (funcs of confidence threshold)


def plot_confusion_matrix2(cm, image_dirs, classifier_names, targets_matrix, matches_matrix, extras_matrix,
                           title='Confusion matrix', cmap=plt.cm.GnBu, use_visual_output=False):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    rows, cols = np.shape(cm)
    for i in range(0, rows):
        for j in range(0, cols):
            # if type(cm[0, 0]) is float:
            txt = '{:.2}'.format(cm[i, j])
            plt.text(j - .2, i - .1, txt, fontsize=8)
            txt = '({:.0f},{:.0f},{:.0f})'.format(matches_matrix[i, j], targets_matrix[i, j], extras_matrix[i, j])
            plt.text(j - .2, i + .1, txt, fontsize=8)
            # else:
            # txt = '{:.2}'.format(cm[i, j])
            #            plt.text(j , i + .2, txt, fontsize=8)

    plt.title(title + '\nNmatches/Ntargets\n(Nmatches,Ntargets,Nextras)', fontsize=10)
    plt.colorbar()
    ylabels = classifier_names
    xlabels = image_dirs
    xtick_marks = np.arange(len(xlabels))
    ytick_marks = np.arange(len(ylabels))
    plt.xticks(xtick_marks, xlabels, rotation=90, fontsize=8)
    plt.yticks(ytick_marks, ylabels, fontsize=8)
    # plt.tight_layout()
    plt.ylabel('classifier')
    plt.xlabel('image_dir')
    date_string = strftime("%d%m%Y_%H%M%S", gmtime())
    figName = 'classifier_results/confusion' + date_string + '.png'
    plt.subplots_adjust(left=.1, right=1, bottom=0.2, top=0.9)
    savefig(figName, format="png")
    if use_visual_output:
        plt.show(block=True)
    return date_string



def test_classifiers(classifierDir='../classifiers/', imageDir='images', use_visual_output=False):
    resultsDir='classifier_results'
    results_filename=os.path.join(resultsDir,'classifier_results_'+trainDir+'.txt')
    max_files_to_try = 1000

    #go thru image directories
    imagedirs = Utils.immediate_subdirs(imageDir)
    if len(imagedirs) == 0:
        print('empty image directory:' + str(imagedirs))
        return None
    for subdir in imagedirs:
        searchStrings.append('class:'+subdir)
        print('image directory:'+str(subdir))
    n_categories = len(imagedirs)

    #go thru classifier directories
    classifiers = Utils.files_in_directory(classifierDir)
#	print(' subdirlist'+str(subdirlist))
    if len(classifiers) == 0:
        print('empty classifier directory:' + str(classifierDir))
        exit()
    n_classifiers = len(classifiers)
    classifier_names = ''
    results_list=[]
    results_matrix=np.zeros([n_classifiers,n_categories])
    matches_over_targets_matrix = np.zeros([n_classifiers, n_categories])
    targets_matrix = np.zeros([n_classifiers, n_categories])
    matches_matrix = np.zeros([n_classifiers, n_categories])
    extras_matrix = np.zeros([n_classifiers, n_categories])
    samplesize=[]
    gotSizesFlag=False
    totTargets = 0
    for i in range(0, len(classifiers)):
        classifier = classifiers[i]
        head, tail = os.path.split("/tmp/d/a.dat")
        classifier_names = classifier_names + tail + '_'
        results_row=[]
        results_matrixrow = []
        cascade_classifier = cv2.CascadeClassifier(classifier)
        check_empty = cascade_classifier.empty()
        if check_empty:
            print('classifier ' + str(classifier) + ' is empty')
            continue
        for j in range(0, len(imagedirs)):
            imagedir = imagedirs[j]
            print('classifier:' + classifier + ' image directory:' + imagedir)
            totTargets, totMatches, totExtraMatches = test_classifier(cascade_classifier, imagedir,
                                                                      use_visual_output=use_visual_output)
            print('totTargets:' + str(totTargets) + ' totMatches:' + str(totMatches) + ' tot ExtraMatches:' + str(
                totExtraMatches) + '           ')
            if totTargets:
                matches_over_targets_matrix[i, j] = float(totMatches) / totTargets
            else:
                matches_over_targets_matrix[i, j] = 0
            targets_matrix[i, j] = totTargets
            matches_matrix[i, j] = totMatches
            extras_matrix[i, j] = totExtraMatches
    results_dict = {'classifiers': classifiers, 'imagedirectories': imagedirs, 'totTargets': str(targets_matrix),
                    'totMatches': str(matches_matrix), 'ExtraMatches': str(extras_matrix)}
h