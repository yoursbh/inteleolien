#!/usr/bin/env python
# -*- coding: utf-8 -*-

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NAME OF SCRIPT
#
#
# Authors: W. Veloz Parra, H. BAI
# Date: 13/11/2018
#
# Version:
#   - 0.0: Initial version
#   - 0.1: export all 36 angles on tower gage 1
# Comments:
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#!------------------------------------------------------------------------------
#!                                          MODULES PRÉREQUIS
#!------------------------------------------------------------------------------
#*============================ Modules Personnels ============================
from tools import utils
#*============================== Modules Communs ==============================
import sys
import itertools
import csv
import numpy
import collections
import multiprocessing
from os import listdir
from os.path import isfile, join
from itertools import islice
from pathlib import Path
import json

#!------------------------------------------------------------------------------
#!                                         CLASS
#!------------------------------------------------------------------------------
class CsvCreator(object):
    ''' Description of this class
        Attributes:
            filebase: basename of .out file (i.e. name without extension) [str]
            header: line number that contains column's name in input file [int]
            datarow: line number that contains the first row of data in input file [int]
            startline: begin writting results from this row in output file [int]
            channels: channel names that will be used for searching [list]            
        Methods:
            find(): search peak/valley for a given channel name
            run(): batch profile for finding processing
    '''
    def __init__(self, directory, wndfn, dmgfn, outputfile, gage, echo):
        self.dir = directory
        self.wndf = wndfn
        self.dmgfn = dmgfn
        self.outputfile = outputfile
        self.gage = str(gage)
        self.echo = echo
        self.csvfile = []

    def windcsv(self, echo=True):
        t = 0
        # sline = 6
        muline = 129
        mustop = 145
        sigmaustep = 73
        sigmavstep = 106
        sigmawstep = 139
        for f in self.wndf:
            seed = []
            muwind = []
            sigmawind = []
            fdata = self.file_read(t, f, extension=".sum")
            # print(fdata)
            flist = f.split("_")
            # seed.append(int(flist[-1].rstrip(".sum")))
            seed.append(int(flist[-1].rstrip(".sum")))
            # seed.append(float(fdata[sline][0]))
            for k in range(muline, mustop):
                muaux = [float(fdata[k][j])for j in range(len(fdata[k])) if j == 4]
                muwind.append(muaux)
                sigmauaux = [float(fdata[k + sigmaustep][j])for j in range(len(fdata[k + sigmaustep])) if j == 16]
                sigmavaux = [float(fdata[k + sigmavstep][j])for j in range(len(fdata[k + sigmavstep])) if j == 16]
                sigmawaux = [float(fdata[k + sigmawstep][j])for j in range(len(fdata[k + sigmawstep])) if j == 16]
                sigmawind.append(list(itertools.chain(*[sigmauaux, sigmavaux, sigmawaux])))
            csvf = []
            csvf.append(list([seed]))
            csvf.append(muwind)
            csvf.append(sigmawind)
            csvf = list(itertools.chain(*csvf))
            csvf = list(itertools.chain(*csvf))
            self.csvfile.append(csvf)
            # print(len(self.csvfile[0]))
            # print(self.csvfile)
            # break
        pass

    def damagecsv(self, echo=True):
        j = 0
        for elem in self.csvfile:
            seed = elem[0]
            for damfile in self.dmgfn:
                if str(seed) in damfile:
                    with open(self.dir[-1] + damfile + ".dam","r") as f:
                        damages = json.loads(f.read())
                    outputline = []
                    for theta in range(0, 360, 10):
                        key = "TwHt{}@{}".format(self.gage, theta)
                        outputline.append(damages[key]["Dcumulate"]*1e0) #1e9
                        # outputline = [x*10e9 for x in outputline]
                    # print(outputline)
                    if j < len(self.csvfile):
                        self.csvfile[j].extend(outputline)
                        j = j + 1
                    # print(self.csvfile[0])
        # print(self.csvfile)
        pass

    def csvwrite(self, echo= True):
        header = ["Seed"]
        temp = ["Wind speed-u (m/s)"]*16
        header.extend(temp)
        temp = ["Std. dev. u", "Std. dev. v", "Std. dev. w"] * 16
        header.extend(temp)
        temp = ["Damage@{}".format(theta) for theta in range(0, 360, 10)]
        header.extend(temp)
        self.csvfile.insert(0,header)

        with open(self.outputfile, 'w') as sfile:
            wr = csv.writer(sfile, dialect='excel', delimiter = ',')
            wr.writerows(self.csvfile)

    def file_read(self, t, filename, extension):
        fdata = []
        path = self.dir[t] + filename + extension
        with open(path, 'r') as f:
            for line in islice(f, 0, None):
                data = line.split(None)
                fdata.append(data)
        return fdata

#!------------------------------------------------------------------------------
#!                                      FUNCTIONS
#!------------------------------------------------------------------------------
def testHao():
    with utils.cd("/home/wilsonveloz/Documents/Hao_Data/Wind"):
        # data = utils.readcsv("NTM_11mps_-1029744777.sum", delimiter="\s", header=112, datarow=130, endrow=145, ncols=2)
        # print(data)
        pass

    # # 1st approch: forward
    # with utils.cd('~/Eolien/Parameters/NREL_5MW_Onshore/Wind'):
    #     with open('100seeds.json', 'r') as f:
    #         seeds = json.loads(f.read())
    # liste = [s for s in seeds if s[0] == "NTM" and s[1] == "13"]
    # seeds = liste

    # s = seeds[0]
    # fname = "{}_{}mps_{}.sum".format(s[0], s[1], s[2])
    
    
    # 2nd approch: backward
    # fname # string of filename
    # flist = fname.split("_")
    # s = flist[-1].rstrip(".sum")

    # for elem in wind_input_data:
    #     seed = elem[0]
    #     for damfile in damfilenames:
    #         if seed in damfile:
    #             with open(damfile,"r") as f:
    #                 damages = json.loads(f.read())
    #             outputline = []
    #             for i in range(10,1,-1)
    #                 key = "TwHt{}@0".format(i)
    #                 outputline.append(damages[key]["Dcumulate"])

    #     elem.extend(outputline)

def find(path, pattern, size=None):
    ''' size: minimum size in bytes (1 GB = 1024 MB = 1024^2 KB = 1024^3 Bytes) [num]
    '''
    with utils.cd(path):
        p = Path().expanduser()
        matched = sorted(p.glob(pattern))
        if size is None:
            result = [x.stem for x in matched]
        else:
            result = [x.stem for x in matched if x.stat().st_size >= size]
    return result


#!------------------------------------------------------------------------------
#!                                     MAIN FUNCTION
#!------------------------------------------------------------------------------
def main():
    # testHao()
    # return
    windpath = "./data/"
    damagepath = "./data/"
    directory = [windpath, damagepath]
    
    # windfiles = [f for f in listdir(windpath) if isfile(join(windpath, f))]
    # damagefiles = [f for f in listdir(damagepath) if isfile(join(damagepath, f))]
    windfiles = find(windpath, "*.sum")
    damagefiles = find(damagepath, "*.dam")

    case = 3
    # DmgG1 : damage from 1000 seeds for wind speed 3, 5, 7, ..., 25 m/s
    if case == 1:
        DmgG1 = CsvCreator(directory=directory, wndfn= windfiles,
                        dmgfn= damagefiles, outputfile="DataNS1000_gage1.csv", gage="1",
                        echo=True)
        DmgG1.windcsv()
        DmgG1.damagecsv()
        DmgG1.csvwrite()

    # DmgG2 : damage from 10 seeds for wind speed 3.0, 3.1, 3.2, ..., 25.0 m/s
    if case == 2:
        DmgG2 = CsvCreator(directory=directory, wndfn=windfiles,
                        dmgfn=damagefiles, outputfile="DataNS10_gage1.csv", gage="1",
                        echo=True)
        DmgG2.windcsv()
        DmgG2.damagecsv()
        DmgG2.csvwrite()

    # DmgG3 : damage from 1000 seeds for wind speed 3.0, 3.1, 3.2, ..., 25.0 m/s
    if case == 3:
        DmgG3 = CsvCreator(directory=directory, wndfn=windfiles,
                           dmgfn=damagefiles, outputfile="Data@1.0mps_1000_gage1.csv", gage="1",
                           echo=True)
        DmgG3.windcsv()
        DmgG3.damagecsv()
        DmgG3.csvwrite()


#!------------------------------------------------------------------------------
#!                                         EXÉCUTION
#!------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
