__author__ = 'jeremy'

import logging
logging.basicConfig(level=logging.INFO)

def get_netname(proto):
    logging.info('looking for netname in'+str(proto))
    with open(proto,'r') as fp:
        l1 = fp.readline()
        l2 = fp.readline()
#    print('line1 '+l1)
#    print('line2 '+l2)
    if 'name' in l1:
        netname = l1[l1.find('name:')+5:] #get string after name:
        netname = netname.replace('"','')  #remove quotes
        logging.info('line1:'+l1)
        logging.info('netname:'+netname)
        return netname
    if 'name' in l2:
        netname = l2[l2.find('name:')+5:] #get string after name:
        netname = netname.replace('"','')  #remove quotes
        logging.info('line2:'+l2)
        logging.info('netname:'+netname)
        return netname
    if 'test_net' or 'train_net' in l1: #the file is prob a solverproto and refers to test/val which may have netname
        fname = l1.split('"')[-2]
        logging.info('trying to find netname in file1 '+fname)
        return get_netname(fname)
    if 'test_net' or 'train_net' in l2:
        fname = l2.split('"')[-2]
        logging.info('trying to find netname in file2 '+fname)
        return get_netname(fname)
    else:
        netname = None
    return netname

def get_traintest_from_proto(proto):
    logging.info('looking for traintest in '+proto)
    with open(proto,'r') as fp:
        train = None
        test = None
        traintest = None
        for line in fp:
            line = line.replace(' ','')  #line with spaces removed
            logging.debug('looking at line:'+line)
            if 'train_net:' in line and line[0] is not '#':
                train = line.replace('train_net:','').replace('"','').replace('\n','')
                print('train:'+train)
            if 'test_net:' in line and line[0] is not '#':
                test = line.replace('test_net:','').replace('"','').replace('\n','')
                print('test:'+test)
            if 'net:' in line and not 'test' in line and not 'train' in line and line[0] is not '#':
                traintest = line.replace('net:','').replace('"','').replace('\n','')
                print('traintest:'+traintest)
        if train and test:
            return((train,test))
        elif train:
            print('got only train not test')
            return((train))
        elif test:
            print('got only test not train')
            return((test))
        elif traintest:
            return((traintest))
        else:
            return None

def get_labelfile_from_traintest(tfile,train_test='both'):
    print('looking for '+train_test+' in '+tfile)
    with open(tfile,'r') as fp:
        for line in fp:
            line = line.replace(' ','')  #line with spaces removed
            print('looking at line:'+line)
            if 'images_and_labels_file:' in line and line[0] is not '#':
                train = line.replace('train_net:','').replace('"','')
                print('train:'+train)
            if 'test_net:' in line and line[0] is not '#':
                test = line.replace('test_net:','').replace('"','')
                print('test:'+test)
            if 'net:' in line and not 'test' in line and not 'train' in line and line[0] is not '#':
                traintest = line.replace('net:','').replace('"','')
                print('traintest:'+traintest)
        if train and test:
            return((train,test))
        elif train:
            print('got only train not test')
            return((train))
        elif test:
            print('got only test not train')
            return((test))
        elif traintest:
            return((traintest))
        else:
            return None
