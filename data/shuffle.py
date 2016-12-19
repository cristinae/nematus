import os
import sys
import resource
import random
import mmap

import bisect
import tempfile
from subprocess import call
from random import shuffle

# random permutation
def random_permutation(N):
    l = list(range(N))
    for i, n in enumerate(l):
        r = random.randint(0, i)
        l[i] = l[r]
        l[r] = n
    return l

                                    
def main(files, temporary=False):

    # The maximum number of files allowed depends on your machine and privileges
    #resource.setrlimit(resource.RLIMIT_NOFILE, (65536,65536))
    #print resource.getrlimit(resource.RLIMIT_NOFILE)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    maxAllowedFiles = soft-200

    tf_os, tpath = tempfile.mkstemp()
    tf = open(tpath, 'w')

    fds = [open(ff) for ff in files]

    for l in fds[0]:
        lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
        print >>tf, "|||".join(lines)

    [ff.close() for ff in fds]
    tf.close()

    # cannot deal with long files, don't fit into memory
    #lines = open(tpath, 'r').readlines()
    #random.shuffle(lines)

    # Cris
    # start extra shuffling as implemented in
    # https://www.kaggle.com/c/avazu-ctr-prediction/forums/t/11018/shuffling-lines-of-big-data-files
    bigFile =  open(tpath, 'r')
    outFile = 'tmpShuff'+str(random.random())
    N=0
    for line in bigFile:
        N+=1
    bigFile.seek(0)

    
    sys.stderr.write("Shuffling {0} lines...\n".format(N))
    p = random_permutation(N)
    ridx = [0]*N
    filesTMP = [[] for i in range(maxAllowedFiles+1)]
    mx = []

    for i, n in enumerate(p):
        pos = bisect.bisect_left(mx, n) - 1
        if pos == -1:
            #files.insert(0, [n])
            filesTMP[0].append(n)
            mx.insert(0, n)
        else:
            #scales into the maximum number of allowed files
            index = pos % maxAllowedFiles 
            filesTMP[index+1].append(n)
            mx[pos] = n

    P = len(filesTMP)
    sys.stderr.write("Shuffling into {0} files...\n".format(P))
    fps = [tempfile.TemporaryFile(mode="w+") for i in range(P)]

    for file_index, line_list in enumerate(filesTMP):
        for line in line_list:
            ridx[line] = file_index

    
    #sys.stderr.write("Writing to temporals\n")
    # write to each temporal file
    for i, line in enumerate(bigFile):
        fps[ridx[i]].write(line)
    for f in fps:
        f.seek(0)

    #sys.stderr.write("Writing to the shuffled file\n")
    # write to the final shuffled file
    with open(outFile, 'w') as out:
        for i in range(N):
            out.write(fps[ridx[p[i]]].readline())

    # extra shuffling  ends
    
        
    if temporary:
        fds = []
        for ff in files:
            path, filename = os.path.split(os.path.realpath(ff))
            fds.append(tempfile.TemporaryFile(prefix=filename+'.shuf', dir=path))
    else:
        fds = [open(ff+'.shuf','w') for ff in files]

    #cris
    #for l in lines:
    #    s = l.strip().split('|||')
    #    for ii, fd in enumerate(fds):
    #        print >>fd, s[ii]
            
    with open(outFile) as myfile:
        for l in myfile:
            s = l.strip().split('|||')
            for ii, fd in enumerate(fds):
                print >>fd, s[ii]
                            
            
    if temporary:
        [ff.seek(0) for ff in fds]
    else:
        [ff.close() for ff in fds]

    os.remove(tpath)
    os.remove(outFile)

    return fds

if __name__ == '__main__':
    main(sys.argv[1:])

    


