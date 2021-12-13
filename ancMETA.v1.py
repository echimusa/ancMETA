#!/usr/bin/python

from __future__ import with_statement
from __future__ import division

import os
import sys
import fileinput
import exceptions
import types
import time
import datetime
import random
import logging
import pickle
import warnings
import itertools as it

from math import sqrt, log, erf

try:
  from networkx import *
except ImportError:
    print("Install network in your python version...")
    print("Existing ...")
    sys.exit(1)
try:
    from scipy.optimize import leastsq
    import scipy as cp
    import scipy.stats as stats
    from scipy.stats import norm, chisqprob
    import scipy.interpolate
    HAS_SCIPY = True
    qnorm = norm.ppf
    pnorm = norm.cdf
except ImportError:
    HAS_SCIPY = False
    print("Install scipy in your current python version...")
    print("Existing ...")
    sys.exit(1)
try:
    import numpy as np
    from numpy.linalg import cholesky as chol
    from numpy.linalg.linalg import LinAlgError
    HAS_NP = True
except ImportError:
    HAS_NP = False
    print("Install Numpy in your current python version...")
    print("Existing ...")
    sys.exit(1)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

try:
    import rpy2.robjects as R0
    r = R0.r
    HAS_R = True
except ImportError:
    HAS_R = False


class ProcessModel:
    def __init__(self, sigma_1=10, sigma_v=10, sigma_w=1):
        self.sigma_1 = sigma_1
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w

    def sample_initial(self):
        return normalvariate(0, self.sigma_1)

    def initial_density(self, x):
        return gaussian_density(x, 0, self.sigma_1)

    def deterministic_transition(self, t, x):
        # return x/2 + 25*x/(1+x**2) + 8*cos(1.2*t)
        return x/2 + 25*x/(1+x**2)

    def sample_transition(self, t, x):
        return self.deterministic_transition(t, x) + normalvariate(0, self.sigma_v)

    def transition_density(self, t, x, x_next):
        return gaussian_density(x_next, self.deterministic_transition(t, x), self.sigma_v)

    def sample_observation(self, t, x):
        return x**2/20 + normalvariate(0, self.sigma_w)

    def observation_density(self, t, y, x_sample):
        return gaussian_density(y, x_sample**2/20, self.sigma_w)


class ancUtils:
    '''
    Useful functions
    '''

    def __init__(self):
        return None

    def logical(self, Valeur):
        '''
        Convert a string to logical variable
        Valeur: a string, NO or YES
        Return a logical variable
        '''
        if Valeur == "YES":
            return True
        else:
            return False

    def safe_open(self, filename):
        '''
        Opens a file in a safe manner or raise an exception if the file doesn't exist
        filename: a string, name of a file
        Return: file object
        '''
        if os.path.exists(filename):
            return open(filename)
        else:
            self.printHead()
            sys.stderr.write(
                "Error => No such file or directory\"%s\"\n" % format(filename))
            return False

    def get_valid_index(self, intab, rs_index):
        intab = intab[rs_index-1, :]
        return(np.where(intab[range(1, len(intab), 2)] != 'NA')[0])

    def head_rdid(self, SUB):
        HEAD = []
        for des in SUB:
            HEAD.append(des[0])
        return HEAD

    def cleanup(self, mpath):
        cmd = "rm"+" "+mpath + "tmpoutput.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpinput.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpgenenames.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpstudies.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpstudyorder.txt"
        os.system(cmd)

    def cleanup_err(self, mpath):
        cmd = "rm"+" "+mpath + "tmpoutput.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpinput.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpgenenames.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpstudies.txt"
        os.system(cmd)
        cmd = "rm"+" "+mpath + "tmpstudyorder.txt"
        os.system(cmd)

    def const_outtab(self, outtab, rs_index, total_studies, sels):

        outtab = np.array(outtab.split('\t'))

        header = outtab[0:16]
        m = 16+total_studies

        pvals = outtab[16:(16+total_studies)]
        mvals = outtab[(16+total_studies):]

        newout = np.hstack((header, pvals[sels], mvals[sels]))

        return(newout)

    def const_intab(self, intab, rs_index):
        intab = intab[rs_index-1, :]
        return(intab[np.where(intab != 'NA')[0]])

    def permute(self, data):
        """Perform a resampling WITHOUT replacement to the table in 'data'."""

        if self.adjustP == "sampling":
            D = np.array([data, data])
            flat = [x for row in D for x in row]
            permuted = random.sample(flat, len(flat))
            return [[permuted.pop() for _ in row] for row in D][0]
        else:
            flat = [x for row in data for x in row]
            permuted = random.sample(flat, len(flat))
            return [[permuted.pop() for _ in row] for row in data]

    def bootstrap(self, data):
        """Perform a resampling WITH replacement to the table in 'data'."""
        if self.adjustP == "sampling":
            D = np.array([data, data])
            flat = [x for row in D for x in row]
            return [[random.choice(flat) for _ in row] for row in D][0]
        else:
            flat = [x for row in data for x in row]
            return [[random.choice(flat) for _ in row] for row in data]

    def transpose(self, data):
        """Transpose a 2-dimensional list."""
        return zip(*data)

    def get_stats(self, data):
        """Apply the test statistic passed as 'func' to the 'data' for the pairs to compare in 'compars'."""
        pval = stats.wilcoxon(data[0], data[1])
        return pval  # [func(tdata[0], tdata[1])] # for x, y in compars]

    def comparisons(self, labels):
        """Return a list of tuples representing the samples to
          compare. If compareall is false, only comparisons against
          the first treatment are performed. If true, all comparisons
          are done."""
        return [(x, y) for x in range(len(labels)) for y in range(len(labels)) if y < x]

    def check_params(self, params):
        '''
        Check necessary parameters in a dict
        params: a dictionary, of option:value 
        '''
        for key in params:
            if params[key][0] in ['Faffy', 'Fanc', 'pathway', 'Fgeno', 'Cagwas', 'Sgwas', 'Fnetwork', 'Fsnp']:
                if os.path.exists(params[key][1]) == False:
                    sys.stderr.write(
                        "Error => No such file or directory \"%s\"\n\n" % params[key][-1])
                    sys.exit(1)

    def pathways_read(self):
        self.path_dict = {}
        P_path = set()
        for line in fileinput.input(self.pathway):
            data = line.split()
            if "ECR777" not in data:
                print"\n File of pathways %s is in a bad format" % self.pathway
                print"\nSkipping Test of overlap between Subnetworks and pathways ....."
                return self.path_dict, P_path
            else:
                idx = data.index("ECR777")
                if len(set(data[1:][:idx])) != 0:
                    self.path_dict[data[0]] = data[1:]
                    P_path = P_path | set(data[1:][idx+1:])
        return self.path_dict, P_path

    def path_enrichment(self, subnetworks, module, Label):
        """ Pathways analysis and scoring overlap between subnetworks and human pathway for enrichement"""
        rows = set()
        D_rows = []
        fin = open(self.outfolder+"NETWORK_RESULT/" +"subnetwork_pathways_score", "wt")
        Label1 = Label + ["Subnetwork_GeneHub", "Subnetwork_#Genes", "Pathway","Z_pathway", "Number_overlaps", "Overlapping_genes", "Known_genes"]
        cut = len(subnetworks)-int(self.TOPScore)
        T_rows = [];S_rows = []
        sid = 0
        for des in subnetworks:  # [:-cut]:
            if des[0] in module:
                if float(des[3]) <= 0.05 and sid <= cut:
                    tmps = set([des[0]]+list(module[des[0]]))
                    sid = sid+1
                    tmp = list(des[3:])
                    T_rows.append(list(tmps))
                    S_rows.append(tmp)
                    rows |= tmps
                    tp = [des[0]]+list(module[des[0]])
                    D_rows.append(tp)
            else:
                pass
        self.path_dict, P_path = self.pathways_read()
        T_rows = list(T_rows)
        S_rows = list(S_rows)

        if len(self.path_dict) == 0:
            fin.write("\t".join(Label+["Lists.Genes"])+"\n")
            for i in range(len(S_rows)):
                fin.write("\t".join([str(d) for d in S_rows[i]]) +
                          "\t"+" ".join([str(d) for d in T_rows[i]])+"\n")
            fin.close()
            T_rows = [];S_rows = []
            return rows, D_rows

        M = len(P_path)+1e-35  # Total number of genes in all human pathways
        for mod in T_rows:
            IDX = T_rows.index(mod)
            net = set(mod)
            net_score = {}
            for pathw in self.path_dict:
                idx = self.path_dict[pathw].index("ECR777")
                Modul = set(self.path_dict[pathw][idx+1:])  # pathway
                # intersetion of a pathway and subnetwork
                Inters = list(Modul & net)
                # intersetion of a pathway, subnetwork and the disease known
                Inters1 = list(self.diseaseGenes & net)
                # lenght of intersetion of a pathway and subnetwork
                alpha = len(Inters)
                # intersection of a subnetwork and all genes of human pathways
                beta = len(net & P_path)
                # intersection of a pathway and all genes in human pathways
                N = 1e-35+len(set(self.path_dict[pathw][idx+1:]) & P_path)
                if alpha == 0:
                    net_score[pathw] = [mod[0], str(
                        len(net)), pathw, 0.0, alpha, Inters, Inters1]
                else:
                    G = beta/float(M)
                    M_dem = G*(1-G)
                    dem_0 = M_dem/float(M)+1e-35
                    dem = np.sqrt(dem_0)+1e-35
                    num = (alpha/float(N))-(beta/float(M))
                    if dem == 0.0:
                        net_score[pathw] = [mod[0], str(len(net)), pathw, num, alpha, Inters, Inters1]
                    else:
                        net_score[pathw] = [mod[0], str(len(net)), pathw, float(num)/float(dem), alpha, Inters, Inters1]  # equation 8 of the manuscript
            if IDX == 0:
                fin.write("\t".join(Label1)+"\n")
            Z_path = net_score.values()
            path2Z_ = sorted(Z_path, key=lambda a_key: float(a_key[4]), reverse=True)[0]  # retrieve the most significant pathway
            t = self.path_dict[path2Z_[2]].index("ECR777")
            if int(path2Z_[4]) == 0:
                path2Z_[4] = '0'
            if len(path2Z_[-1]) == 0:
                path2Z_[-1] = ['-']
            fin.write("\t".join([str(de) for de in list(S_rows[IDX])])+"\t"+"\t".join(path2Z_[:2])+"\t" + " ".join(self.path_dict[path2Z_[2]][:t])+"\t" +"\t".join([str(round(path2Z_[3], 4))])+"\t" + str(int(path2Z_[4])) + "\t" + ','.join(path2Z_[-2]) + "\t" + ','.join(path2Z_[-1]) + "\n")
            net_score.clear()
        T_rows = []
        S_rows = []
        fin.close()
        return rows, D_rows

    def terminate(self):
        '''
        Terminate the process
        '''
        log_rem = os.getcwd()
        try:
            pkl_files = [fil for fil in os.listdir(
                self.outfolder) if fil.endswith('pkl')]
            os.remove(self.outfolder+"1.Rout")
            for fil in pkl_files:
                if self.clear:
                    os.remove(self.outfolder+fil)
                else:
                    pass
            os.system("cp"+" "+log_rem+'/'+self.logFile+" "+self.wrkdir)
            print'Log file generated in '+log+'\nHave a nice day!\n'
        except:
            pass        
        finally:
            print("Existing ...")
            sys.exit(1)


class ancInit(ancUtils):
    '''
    Initialize ancMETA by 
        - creating the output folder
        - Reading the parameter file
    '''

    try:
        logger.removeHandler(logger.handlers[1])
    except:
        pass
    log_rem = os.getcwd()
    logger.setLevel(logging.INFO)
    filenames = os.listdir(os.curdir)
    for filename in filenames:
        if os.path.isfile(filename) and filename.endswith('.log'):
            os.system("rm"+" "+log_rem+"/*.log")
    logFile = 'ancMETA-'+str(time.time()).split('.')[0]+'.log'
    fh = logging.FileHandler(logFile, mode='w')
    logger.addHandler(fh)

    logger.info("\n********************************************************************************************************************")
    logger.info(
        "     ancMETA: Leveraging cross-population Gene/Sub-network Meta-Analysis to Recover Signal ")
    logger.info(
        "                  Underlying Ethnic difference in Disease Risk and Drug Responses...............")
    logger.info(
        "                                  Medical Population Genetics  Group                                               ")
    logger.info(
        "                            University of Cape Town, South Africa                                       ")
    logger.info(
        "                                         Verson 1.0 Beta                                                       ")
    logger.info("********************************************************************************************************************\n")

    def __init__(self, argv, logFile=logFile):
        '''
        Initializing ancMETA by reading the parameter file
        '''
        self.argv = [argv]
        self.logFile = logFile

        if len(self.argv) == 0 or self.argv == ['']:
            logger.info('Command line usage: %s <parameter file>  ' %
                        sys.argv[0])
            logger.info('eg: python ancMETA.py parancMETA.txt\n')
            sys.exit(1)
        elif len(self.argv) == 1:
            try:
                self.paramFile = self.argv[0]
                if os.path.exists(os.getcwd()+'/'+self.paramFile):
                    inFile = open(os.getcwd()+'/'+self.paramFile)
                elif os.path.exists(self.paramFile):
                    inFile = open(self.paramFile)
                else:
                    logger.info(
                        '\nError => Failed to process the input, check the parameters!\n')
                    self.terminate()
                rows = {}
                self.opts = []
                self.opts1 = {}
                myList1 = []
                myList = ['clear_stat', 'fast_mode', 'robust']
                myList2 = ['LDcutoff', 'Pcutoff', 'SNPtoGENE', 'TOPScore']

                for line in inFile:
                    data = line.split()
                    if not line.startswith("#"):
                        if len(data) != 0:
                            rows[data[0].split(":")[0]] = data[0].split(":")[1]
                            data1 = data[0].split(":")[0]
                            data2 = data[0].split(":")[1]
                            if data1 in myList:
                                data2 = self.logical(data2)
                            self.opts.append(data2)
                            self.opts1[data1] = data2
                            myList1.append(data1)

                if len(self.opts1) == 28 or len(self.opts1) == 27:
                    pass
                else:
                    print(len(self.opts1))
                    sys.stderr.write(
                        '\nMissing parameters!!\nError => Failed to process the input, check the parameters!\n')
                    self.terminate()
                try:
                    self.outfolder = self.opts1['outfolder']
                    if os.path.exists(self.outfolder):
                        pass
                    else:
                        os.makedirs(self.outfolder)
                        os.makedirs(self.outfolder+"GENE_RESULT/")
                        os.makedirs(self.outfolder+"NETWORK_RESULT/")
                except IndexError:
                    sys.stderr.write(
                        'Error => Can not create directory. Please create a directory OUT in your working directory\n')
                    sys.exit(1)

                for par in self.opts1:
                    if par in myList2:
                        try:
                            self.opts1[par] = float(self.opts1[par])
                        except:
                            sys.stderr.write(
                                'Error => Failed to process the input, check the parameters!\n')
                            self.terminate()
                            sys.exit(1)
                        finally:
                            if (par == 'LDcutoff' or par == 'Pcutoff') and (0 >= self.opts1[par] or self.opts1[par] >= 1):
                                sys.stderr.write(
                                    'Invalid LD or Pvalue cutoff; Must range between 0 and 1!\n')
                                sys.stderr.write(
                                    'Error => Failed to process the input, check the parameters!\n\n')
                                self.terminate()
                                sys.exit(1)
                            if par == 'SNPtoGENE' and self.opts1[par] <= 0:
                                sys.stderr.write(
                                    'Invalid step path; Must be greater than 0!\n')
                                sys.stderr.write(
                                    'Error => Failed to process the input, check the parameters!\n\n')
                                self.terminate()
                                sys.edexit(1)

                if len(self.opts1) == 28:
                    allList = ['wkdir','pathscpt','Cagwas', 'Sgwas', 'pathway', 'Faffy', 'Fgeno', 'Fsnp', 'Fnetwork', 'LDcutoff', 'Pcutoff', 'SNPtoGENE', 'Path', 'Gene_pv', 'Gene_LD', 'adjustPValues',
                               'Sampling_size', 'pi0_method', 'fdr_level', 'robust', 'TOPScore', 'Disease', 'gene_disease_file', 'clear_stat', 'fast_mode', 'nb_net', 'inxed_sort', 'outfolder']
                else:
                    allList = ['wkdir','pathscpt','Cagwas', 'Sgwas', 'pathway', 'Faffy', 'Fld', 'Fnetwork', 'LDcutoff', 'Pcutoff', 'SNPtoGENE', 'Path', 'Gene_pv', 'Gene_LD', 'adjustPValues',
                               'Sampling_size', 'pi0_method', 'fdr_level', 'robust', 'TOPScore', 'Disease', 'gene_disease_file', 'clear_stat', 'fast_mode', 'nb_net', 'inxed_sort', 'outfolder']

                List1 = []
                for i in range(len(myList1)):
                    if myList1[i] != allList[i]:
                        List1.append(myList1[i])
                if len(List1) != 0:
                    sys.stderr.write('\nInvalid option: '+','.join(List1))
                    sys.stderr.write(
                        '\nError => Failed to process the input, check the parameters!\n')
                    self.terminate()
                    sys.exit()

                for param in myList:
                    rows[param] = self.logical(rows[param])
                i = 1
                self.Params = {}
                for param in myList1:
                    if rows[param] != False:
                        self.Params[i] = [param, rows[param]]
                        i += 1
                self.check_params(self.Params)
                
                self.wkdir = self.opts1['wkdir']
                self.pathscpt = self.opts1['pathscpt']
                self.assocFile = self.opts1['Cagwas']
                self.assocStudy = self.opts1['Sgwas']
                self.pathway = self.opts1['pathway']
                self.affyFile = self.opts1['Faffy']
                self.Path = self.opts1['Path']

                if len(self.opts1) == 28:
                    self.genoFile = self.opts1['Fgeno']
                    self.snpFile = self.opts1['Fsnp']
                else:
                    self.Fld = self.opts1['Fld']
                    os.system("cp"+" "+self.Fld+" "+self.outfolder +
                              "GENE_RESULT/gene2geneLD.net")
                self.networkFile = self.opts1['Fnetwork']
                try:
                    self.Pcutoff = float(self.opts1['Pcutoff'])
                    self.boundary = int(self.opts1['SNPtoGENE'])
                    self.TOPScore = int(self.opts1['TOPScore'])
                    self.LDCutoff = int(self.opts1['LDcutoff'])
                    self.nb_net = int(self.opts1['nb_net'])
                    self.inxed_sort = int(self.opts1['inxed_sort'])
                    self.Sample_size = int(self.opts1['Sampling_size'])
                    self.fdr_level = float(self.opts1['fdr_level'])
                except ValueError, TypeError:
                    sys.stderr.write(
                        'Invalid value Error in option parameter! Check all values in parameter file\n\n')
                    self.terminate()
                    sys.exit(1)
                self.disease = self.opts1['Disease']
                self.gene_disease = self.opts1['gene_disease_file']

                Gene_pvs = ['fisher', 'simes', 'gene_all', 'smallest', 'gwbon']
                self.Gene_pv = self.opts1['Gene_pv']
                self.clear = self.opts1['clear_stat']
                self.fast = self.opts1['fast_mode']

                if self.Gene_pv.lower() not in Gene_pvs:
                    sys.stderr.write('Invalid option for gene pvalue option Gene_pv:'+self.Gene_pv +
                                     '\n'+'Error => Failed to process the input, check the parameters!\n\n')
                    self.terminate()
                    sys.exit(1)
                Gene_LDs = ['closest', 'zscore', 'maxscore']
                self.Gene_LD = self.opts1['Gene_LD']
                if self.Gene_LD.lower() not in Gene_LDs:
                    sys.stderr.write('Invalid option for gene-LD option:'+self.Gene_LD +
                                     '\n'+'Error => Failed to process the input, check the parameters!\n\n')
                    self.terminate()
                    sys.exit(1)
                self.lampld = self.opts[12]
                self.adjustP = self.opts1['adjustPValues']

                self.pi0_method = self.opts1['pi0_method']
                self.robust = self.opts1['robust']
                # retrieving disease genes
                self.diseaseGenes = set(self.userDiseaseGenes()[-1])
                self.head = True
            except IndexError, TypeError:
                sys.stderr.write(
                    'Error => Failed to process the input, check the parameters!\n\n')
                self.terminate()
                sys.exit(1)
        else:
            logger.info('Command line usage: %s <parameter file>  ' %
                        sys.argv[0])
            logger.info('eg. python ancMETA.py parancMETA.txt\n')
            self.terminate()
            sys.exit(1)


class COMB_pvalues:
    '''
    extra pvalues functions
    '''

    def __init__(self):
        return None

    def multipletests(self, pvals, alpha=0.05, method='hs', returnsorted=False):

        pvals = np.asarray(pvals)
        alphaf = alpha  # Notation ?
        sortind = np.argsort(pvals)
        pvals = pvals[sortind]
        sortrevind = sortind.argsort()
        ntests = len(pvals)
        alphacSidak = 1 - np.power((1. - alphaf), 1./ntests)
        alphacBonf = alphaf / float(ntests)
        if method.lower() in ['b', 'bonf', 'bonferroni']:
            reject = pvals <= alphacBonf
            pvals_corrected = pvals * float(ntests)

        elif method.lower() in ['s', 'sidak']:
            reject = pvals <= alphacSidak
            pvals_corrected = 1 - np.power((1. - pvals), ntests)

        elif method.lower() in ['hs', 'holm-sidak']:
            alphacSidak_all = 1 - \
                np.power((1. - alphaf), 1./np.arange(ntests, 0, -1))
            notreject = pvals > alphacSidak_all
            nr_index = np.nonzero(notreject)[0]
            if nr_index.size == 0:
                                # nonreject is empty, all rejected
                notrejectmin = len(pvals)
            else:
                notrejectmin = np.min(nr_index)
            notreject[notrejectmin:] = True
            reject = ~notreject
            pvals_corrected_raw = 1 - \
                np.power((1. - pvals), np.arange(ntests, 0, -1))
            pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)

        elif method.lower() in ['h', 'holm']:
            notreject = pvals > alphaf / np.arange(ntests, 0, -1)
            nr_index = np.nonzero(notreject)[0]
            if nr_index.size == 0:
                # nonreject is empty, all rejected
                notrejectmin = len(pvals)
            else:
                notrejectmin = np.min(nr_index)
            notreject[notrejectmin:] = True
            reject = ~notreject
            pvals_corrected_raw = pvals * np.arange(ntests, 0, -1)
            pvals_corrected = np.maximum.accumulate(pvals_corrected_raw)

        elif method.lower() in ['sh', 'simes-hochberg']:
            alphash = alphaf / np.arange(ntests, 0, -1)
            reject = pvals <= alphash
            rejind = np.nonzero(reject)
            if rejind[0].size > 0:
                rejectmax = np.max(np.nonzero(reject))
                reject[:rejectmax] = True
            pvals_corrected_raw = np.arange(ntests, 0, -1) * pvals
            pvals_corrected = np.minimum.accumulate(
                pvals_corrected_raw[::-1])[::-1]

        elif method.lower() in ['ho', 'hommel']:
            a = pvals.copy()
            for m in range(ntests, 1, -1):
                cim = np.min(m * pvals[-m:] / np.arange(1, m+1.))
                a[-m:] = np.maximum(a[-m:], cim)
                a[:-m] = np.maximum(a[:-m], np.minimum(m * pvals[:-m], cim))
            pvals_corrected = a
            reject = a <= alphaf

        elif method.lower() in ['fdr_bh', 'fdr_i', 'fdr_p', 'fdri', 'fdrp']:
            # delegate, call with sorted pvals
            reject, pvals_corrected = self.fdrcorrection(
                pvals, alpha=alpha, method='indep')
        elif method.lower() in ['fdr_by', 'fdr_n', 'fdr_c', 'fdrn', 'fdrcorr']:
            # delegate, call with sorted pvals
            reject, pvals_corrected = self.fdrcorrection(
                pvals, alpha=alpha, method='n')
        elif method.lower() in ['fdr_tsbky', 'fdr_2sbky', 'fdr_twostage']:
            # delegate, call with sorted pvals
            reject, pvals_corrected = self.fdrcorrection_twostage(
                pvals, alpha=alpha, method='bky')[:2]
        elif method.lower() in ['fdr_tsbh', 'fdr_2sbh']:
            # delegate, call with sorted pvals
            reject, pvals_corrected = self.fdrcorrection_twostage(
                pvals, alpha=alpha, method='bh')[:2]

        elif method.lower() in ['fdr_gbs']:
            ii = np.arange(1, ntests + 1)
            q = (ntests + 1. - ii)/ii * pvals / (1. - pvals)
            pvals_corrected_raw = np.maximum.accumulate(q)  # up requirementd
            pvals_corrected = np.minimum.accumulate(
                pvals_corrected_raw[::-1])[::-1]
            reject = pvals_corrected <= alpha

        else:
            raise ValueError('method not recognized')

        if not pvals_corrected is None:  # not necessary anymore
            pvals_corrected[pvals_corrected > 1] = 1
        if returnsorted:
            return reject, pvals_corrected, alphacSidak, alphacBonf
        else:
            if pvals_corrected is None:
                return reject[sortrevind], pvals_corrected, alphacSidak, alphacBonf
            else:
                return reject[sortrevind], pvals_corrected[sortrevind], alphacSidak, alphacBonf

    # TODO: rename drop 0 at end
    def fdrcorrection(self, pvals, alpha=0.05, method='indep'):

        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = pvals[pvals_sortind]
        sortrevind = pvals_sortind.argsort()
        if method in ['i', 'indep', 'p', 'poscorr']:
            ecdffactor = self._ecdf(pvals_sorted)
        elif method in ['n', 'negcorr']:
            cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))  # corrected this
            ecdffactor = self._ecdf(pvals_sorted) / cm

        else:
            raise ValueError('only indep and necorr implemented')
        reject = pvals_sorted <= ecdffactor*alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True
        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(
            pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        return reject[sortrevind], pvals_corrected[sortrevind]

    def fdrcorrection_twostage(self, pvals, alpha=0.05, method='bky', iter=False):
        ntests = len(pvals)
        if method == 'bky':
            fact = (1.+alpha)
            alpha_prime = alpha / fact
        elif method == 'bh':
            fact = 1.
            alpha_prime = alpha
        else:
            raise ValueError("only 'bky' and 'bh' are available as method")
        alpha_stages = [alpha_prime]
        rej, pvalscorr = self.fdrcorrection(
            pvals, alpha=alpha_prime, method='indep')
        r1 = rej.sum()
        if (r1 == 0) or (r1 == ntests):
            return rej, pvalscorr * fact, ntests - r1, alpha_stages
        ri_old = r1
        while 1:
            ntests0 = 1.0 * ntests - ri_old
            alpha_star = alpha_prime * ntests / ntests0
            alpha_stages.append(alpha_star)

            rej, pvalscorr = self.fdrcorrection(
                pvals, alpha=alpha_star, method='indep')
            ri = rej.sum()
            if (not iter) or ri == ri_old:
                break
            elif ri < ri_old:
                # prevent cycles and endless loops
                raise RuntimeError(" oops - shouldn't be here")
            ri_old = ri

        # make adjustment to pvalscorr to reflect estimated number of Non-Null cases
        # decision is then pvalscorr < alpha  (or <=)
        pvalscorr *= ntests0 * 1.0 / ntests
        if method == 'bky':
            pvalscorr *= (1. + alpha)
        return rej, pvalscorr, ntests - ri, alpha_stages

    def _ecdf(self, x):
        '''no frills empirical cdf used in fdrcorrection
        '''
        nobs = len(x)
        return np.arange(1, nobs+1)/float(nobs)
    # Drawing Pvalues of from Z-score

    def zscore_to_pvalues(self, z):
        """Computing pvalue from a given z-score, the function retains a two-tailed pvalue """
        p = 0.5*(1.0+erf(z/np.sqrt(2)))
        pvalues = 1.0-p
        if p > 1.0:
            return pvalues-9e-16
        else:
            return pvalues

    def estimated_autocorrelation(self, x):
        n = len(x)
        x = np.array(x)
        variance = np.var(x)
        x = x-np.mean(x)
        r = np.correlate(x, x, mode='full')[-n:]
        #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
        result = r/(variance*(np.arange(n, 0, -1)))
        return result

    def fisherp(self, pvals):
        """ combined fisher probability without correction """
        s = -2 * np.sum(np.log(pvals))
        p = 0.5*(1.0+erf(s/np.sqrt(2)))
        pvalues = 2.0*(1.0-p)
        if pvalues > 1.0:
            return np.mean(pvals)+9e-16
        else:
            return pvalues

    def z_score_combine(self, pvals, sigma):
        L = len(pvals)
        pvals = np.array(pvals, dtype=np.float64)
        pvals[pvals == 1] = 1.0 - 9e-16
        z = np.mean(qnorm(1.0 - pvals, loc=0, scale=1))
        if z == 0.0:
           z = 9e-10
        sz = 1.0/L * np.sqrt(L + 2 * np.tril(sigma, k=-1).sum())
        res = {'p': norm.sf(z/sz), 'OK': True}
        if res['p'] >= 1.0:
            res['p'] = res['p']-9e-16  # 0.99
        elif res['p'] == 0.0:
            res['p'] = res['p']+9e-16
        return res['p']

    def genomic_control(self, pvals):
        """calculate genomic control factor, lambda >>> genomic_control([0.25, 0.5, 0.75]) 1.0000800684096998
        >>> genomic_control([0.025, 0.005, 0.0075]) 15.715846578113579
        """
        pvals = np.asarray(pvals)
        return np.median(stats.chi2.ppf(1 - pvals, 1)) / 0.4549

    def genome_control_adjust(self, pvals):
        """
        adjust p-values by the genomic control factor, lambda
        >>> genome_control_adjust([0.4, 0.01, 0.02])
        array([ 0.8072264 ,  0.45518836,  0.50001716])
        """
        pvals = np.asarray(pvals)
        qchi = stats.chi2.ppf(1 - pvals, 1)
        gc = np.median(qchi) / 0.4549
        return 1 - stats.chi2.cdf(qchi / gc, 1)

    def computeQValues(self, pvalues, vlambda=None, pi0_method="smoother", fdr_level=None, robust=False, smooth_df=3, smooth_log_pi0=False, pi0=None):
        """compute qvalues after the method by Storey et al. (2002)"""

        if min(pvalues) < 0 or max(pvalues) > 1:
            raise ValueError("p-values out of range")
        m = len(pvalues)
        pvalues = np.array(pvalues, dtype=np.float)
        if vlambda == None:
            vlambda = np.arange(0, 0.95, 0.05)
        if pi0 == None:
            if type(vlambda) == float:
                vlambda = (vlambda,)
            if len(vlambda) > 1 and len(vlambda) < 4:
                raise ValueError(
                    " if length of vlambda greater than 1, you need at least 4 values.")

            if len(vlambda) > 1 and (min(vlambda) < 0 or max(vlambda) >= 1):
                raise ValueError("vlambda must be within [0, 1).")
            # estimate pi0
            if len(vlambda) == 1:
                vlambda = vlambda[0]
                if vlambda < 0 or vlambda >= 1:
                    raise ValueError("vlambda must be within [0, 1).")
                pi0 = np.mean([x >= vlambda for x in pvalues]) / \
                    (1.0 - vlambda)
                pi0 = min(pi0, 1.0)
            else:
                pi0 = np.zeros(len(vlambda), np.float)
                for i in range(len(vlambda)):
                    pi0[i] = np.mean([x >= vlambda[i]
                                      for x in pvalues]) / (1.0 - vlambda[i])
                if pi0_method == "smoother" or pi0_method == "permute":
                    if smooth_log_pi0:
                        pi0 = np.log(pi0)
                    if HAS_SCIPY:
                        tck = scipy.interpolate.splrep(
                            vlambda, pi0, k=smooth_df, s=10000)
                        pi0 = scipy.interpolate.splev(max(vlambda), tck)
                    else:
                        raise ImportError("pi0_method smoother requires scipy")

                    if smooth_log_pi0:
                        pi0 = np.exp(pi0)

                elif pi0_method == "bootstrap":
                    minpi0 = min(pi0)
                    mse = np.zeros(len(vlambda), np.float)
                    pi0_boot = np.zeros(len(vlambda), np.float)
                    for i in xrange(m):
                        # sample pvalues
                        idx_boot = np.random.random_integers(0, m-1, m)
                        pvalues_boot = pvalues[idx_boot]
                        for x in xrange(len(vlambda)):
                            # compute number of pvalues larger than lambda[x]
                            pi0_boot[x] = np.mean(
                                pvalues_boot > vlambda[x]) / (1.0 - vlambda[x])
                        mse += (pi0_boot - minpi0) ** 2
                    pi0 = min(pi0[mse == min(mse)])
                else:
                    raise ValueError(
                        "'pi0_method' must be one of 'smoother' or 'bootstrap'.")
                    pi0 = min(pi0, 1.0)
        if pi0_method == "bootstrap":
            if pi0 <= 0:
                pi0 = 2*np.mean(pvalues)
                Q = self.Qvalues(pvalues, pi0, fdr_level, robust=False)
                return Q
                #raise ValueError( "The estimated pi0 <= 0 (%f). Check that you have valid p-values or use another vlambda method." %  pi0)
        else:
            if pi0 <= 0:
                pi0 = 2*np.mean(pvalues)
                Q = self.Qvalues(pvalues, pi0, fdr_level, robust=False)
                return Q
                #raise ValueError( "The estimated pi0 <= 0 (%f). Check that you have valid p-values or use another vlambda method." %  pi0)
        Q = self.Qvalues(pvalues, pi0, fdr_level, robust=False)
        return Q

    def Qvalues(self, pvalues, pi0, fdr_level, robust):
        if fdr_level != None and (fdr_level <= 0 or fdr_level > 1):
            raise ValueError("'fdr_level' must be within (0, 1].")
        # compute qvalues
        idx = np.argsort(pvalues)
        m = len(pvalues)
        # monotonically decreasing bins, so that bins[i-1] > x >=  bins[i]
        bins = np.unique(pvalues)[::-1]
        # v[i] = number of observations less than or equal to pvalue[i]
        # could this be done more elegantly?
        val2bin = len(bins) - np.digitize(pvalues, bins)
        v = np.zeros(m, dtype=np.int)
        lastbin = None
        for x in xrange(m-1, -1, -1):
            bin = val2bin[idx[x]]
            if bin != lastbin:
                c = x
            v[idx[x]] = c+1
            lastbin = bin
        qvalues = pvalues * pi0 * m / v
        if robust:
            qvalues /= (1.0 - (1.0 - pvalues)**m)
        # bound qvalues by 1 and make them monotonic
        qvalues[idx[m-1]] = min(qvalues[idx[m-1]], 1.0)
        for i in xrange(m-2, -1, -1):
            qvalues[idx[i]] = min(min(qvalues[idx[i]], qvalues[idx[i+1]]), 1.0)
        return qvalues

    ###################################################################
    # adjust P-Value
    ###################################################################
    def adjustPValues(self, pvalues, method='fdr', n=None):
        '''returns an array of adjusted pvalues p: numeric vector of p-values (possibly with 'NA's).  Any other R is coerced by 'as.numeric'. method: correction method. Valid values are: n: number of comparisons, must be at least 'length(p)'; only set.
this (to non-default) when you know what you are doing'''

        if n == None:
            n = len(pvalues)
        if method == "fdr":
            method = "BH"
        # optional, remove NA values
        p = np.array(pvalues, dtype=np.float)
        lp = len(p)
        assert n <= lp
        if n <= 1:
            return p
        if n == 2 and method == "hommel":
            method = "hochberg"
        if method == "bonferroni":
            p0 = n * p
        elif method == "holm":
            i = np.arange(lp)
            o = np.argsort(p)
            ro = np.argsort(o)
            m = np.maximum.accumulate((n - i) * p[o])
            p0 = m[ro]+1e-35
        elif method == "hommel":
            raise NotImplementedError("hommel method not fully implemented")
        elif method == "hochberg" or method == "sampling":
            i = np.arange(0, lp)[::-1]
            o = np.argsort(1-p)
            ro = np.argsort(o)
            m = np.minimum.accumulate((n - i) * p[o])
            p0 = m[ro]+1e-35
        elif method == "BH":
            i = np.arange(1, lp + 1)[::-1]
            o = np.argsort(1-p)
            ro = np.argsort(o)
            m = np.minimum.accumulate(float(n) / i * p[o])
            p0 = m[ro]+1e-35
        elif method == "BY":
            i = np.arange(1, lp + 1)[::-1]
            o = np.argsort(1-p)
            ro = np.argsort(o)
            q = np.sum(1.0 / np.arange(1, n + 1))
            m = np.minimum.accumulate(q * float(n) / i * p[o])
            p0 = m[ro]+1e-35
        elif method == "none":
            p0 = p+1e-35
        return np.minimum(p0, np.ones(len(p0)))

    def normpdf(self, x, Mean, var):
        """ Computing Gaussian probability density function"""
        pi = 3.1415926
        if var == 0.0:
            var = 1
            denom = (2*pi*var)**.5
            num = np.exp(-(float(x)-float(Mean))**2)/(2*var)
        else:
            denom = (2*pi*var)**.5
            num = np.exp(-(float(x)-float(Mean))**2)/(2*var)
        return num/denom

    def stouffer_liptak(self, pvals, sigma=None):
        """The stouffer_liptak correction. >>> stouffer_liptak([0.1, 0.2, 0.8, 0.12, 0.011])
        {'p': 0.0168..., 'C': 2.1228..., 'OK': True}
        >>> stouffer_liptak([0.5, 0.5, 0.5, 0.5, 0.5])
        {'p': 0.5, 'C': 0.0, 'OK': True}
        >>> stouffer_liptak([0.5, 0.1, 0.5, 0.5, 0.5]) {'p': 0.28..., 'C': 0.57..., 'OK': True}
        """
        if len(pvals) > 7001:
            pvals = random.sample(pvals, 7000)
        L = len(pvals)
        pvals = np.array(pvals, dtype=np.float64)
        pvals[pvals == 1] = 1.0 - 9e-16
        #qvals = qnorm(1.0 - pvals, loc=0, scale=1).reshape(L, 1)
        qvals = norm.isf(pvals, loc=0, scale=1).reshape(L, 1)
        # if any(np.isinf(qvals)):
        #raise Exception("bad values: %s" % pvals[list(np.isinf(qvals))])
        # dont do the correction unless sigma is specified.
        result = {"OK": True}
        if not sigma is None:
            try:
                C = chol(sigma)
                Cm1 = np.asmatrix(C).I  # C^-1
                # qstar
                qvals = Cm1 * qvals
            except LinAlgError, e:
                result["OK"] = False
                if False:
                    try:
                        sigma -= 0.05
                        np.fill_diagonal(sigma, 0.999)
                        sigma[sigma <= 0] = 0.005
                        C = chol(sigma)
                        Cm1 = np.asmatrix(C).I  # C^-1
                        qvals = Cm1 * qvals
                        result['OK'] = True
                    except LinAlgError, e:
                        print >>sys.stderr, e
                        # cant do the correction non-invertible
                result1 = self.z_score_combine(pvals, sigma)
                return result1
        Cp = qvals.sum() / np.sqrt(len(qvals))
        # get the right tail.
        pstar = norm.sf(Cp)
        pstar = pstar/np.sqrt(pstar)
        if np.isnan(pstar):
            print >>sys.stderr, "BAD:", pvals, sigma
            pstar = np.median(pvals)
            result["OK"] = True
        result.update({"C": Cp, "p": pstar})
        if result["p"] >= 1.0:
            result["p"] = np.mean(pvals)  # to avoid pvalues greater than 1.0
        elif result["p"] == 0.0:
            result["p"] = np.mean(pvals)
        return result["p"]

    def gen_sigma_matrix(self, group, acfs):
        if len(group) > 7001:
            group = random.sample(group, 7000)
        a = np.eye(len(group), dtype=np.float64)
        group = enumerate(group)
        for (i, ibed), (j, jbed) in it.combinations(group, 2):
            a[j, i] = a[i, j] = acfs[i]
        """# visualize the sigma matrix. from mpl_toolkits.mplot3d import Axes3D X, Y = np.mgrid[0:a.shape[0], 0:a.shape[0]]
                f = plt.figure() ax = f.add_subplot(111, projection='3d') ax.plot_wireframe(X, Y, np.log10(a + 1))
                plt.show()
                """
        return a


class qqvalues_plot:
    '''
    Useful functions
    '''

    def __init__(self):
        return None

    def calc_median(self, scores, exp_median=0.5):
        s = sp.copy(scores)
        s.sort()
        median = s[len(s) / 2]
        del s
        return (exp_median - median)

    def _estAreaBetweenCurves_(self, quantiles, expQuantiles):
        area = 0
        for i in range(0, len(quantiles) - 1):
            area += (expQuantiles[i + 1] - expQuantiles[i]) * (abs(
                quantiles[i + 1] - expQuantiles[i + 1] + quantiles[i] - expQuantiles[i])) / 2.0
        return area

    def calc_ks_stats(self, scores, exp_scores=None):
        from scipy import stats
        if exp_scores:
            (D, p_val) = stats.ks_2samp(scores, exp_scores)
        else:
            (D, p_val) = stats.kstest(scores, stats.uniform.cdf)
        return {'D': D, 'p_val': p_val}

    def _getExpectedPvalueQuantiles_(self, numQuantiles):
        quantiles = []
        for i in range(numQuantiles):
            quantiles.append(float(i) + 0.5 / (numQuantiles + 1))
        return quantiles

    def get_log_quantiles(self, scores, num_dots=1000, max_val=5):
        """
        Uses scipy
        """
        scores = cp.copy(cp.array(scores))
        scores.sort()
        indices = cp.array(10 ** ((-cp.arange(1, num_dots + 1, dtype='single') / (num_dots + 1)) * max_val)
                           * len(scores), dtype='int')
        return -cp.log10(scores[indices])

    def simple_log_qqplot(self, quantiles_list, png_file=None, pdf_file=None, quantile_labels=None, line_colors=None,
                          max_val=5, title=None, text=None, plot_label=None, ax=None, **kwargs):
        storeFig = False
        if ax is None:
            f = plt.figure(figsize=(5.4, 5))
            ax = f.add_axes([0.1, 0.09, 0.88, 0.86])
            storeFig = True
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2.0)
        num_dots = len(quantiles_list[0])
        exp_quantiles = cp.arange(
            1, num_dots + 1, dtype='single') / (num_dots + 1) * max_val
        for i, quantiles in enumerate(quantiles_list):
            if line_colors:
                c = line_colors[i]
            else:
                c = 'b'
            if quantile_labels:
                ax.plot(exp_quantiles, quantiles,
                        label=quantile_labels[i], c=c, alpha=0.5, linewidth=2.2)
            else:
                ax.plot(exp_quantiles, quantiles,
                        c=c, alpha=0.5, linewidth=2.2)
        ax.set_ylabel("Observed $-log_{10}(p$-value$)$")
        ax.set_xlabel("Expected $-log_{10}(p$-value$)$")
        if title:
            ax.title(title)
        max_x = max_val
        max_y = max(map(max, quantiles_list))
        ax.axis([-0.025 * max_x, 1.025 * max_x, -0.025 * max_y, 1.025 * max_y])
        if quantile_labels:
            fontProp = matplotlib.font_manager.FontProperties(size=10)
            # , handlelen=0.05, pad=0.018)
            ax.legend(loc=2, numpoints=2, markerscale=1, prop=fontProp)
        y_min, y_max = plt.ylim()
        if text:
            f.text(0.05 * max_val, y_max * 0.9, text)
        if plot_label:
            f.text(-0.138 * max_val, y_max * 1.01, plot_label, fontsize=14)
        if storeFig == False:
            return
        if png_file != None:
            f.savefig(png_file)
        if pdf_file != None:
            f.savefig(pdf_file, format='pdf')

    def get_quantiles(self, scores, num_dots=1000):
        """
        Uses scipy
        """
        scores = cp.copy(cp.array(scores))
        scores.sort()
        indices = [int(len(scores) * i / (num_dots + 2))
                   for i in range(1, num_dots + 1)]
        return scores[indices]

    def simple_qqplot(self, quantiles_list, png_file=None, pdf_file=None, quantile_labels=None, line_colors=None,
                      title=None, text=None, ax=None, plot_label=None, **kwargs):
        storeFig = False
        if ax is None:
            f = plt.figure(figsize=(5.4, 5))
            ax = f.add_axes([0.11, 0.09, 0.87, 0.86])
            storeFig = True
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2.0)
        num_dots = len(quantiles_list[0])
        exp_quantiles = cp.arange(
            1, num_dots + 1, dtype='single') / (num_dots + 1)
        for i, quantiles in enumerate(quantiles_list):
            if line_colors:
                c = line_colors[i]
            else:
                c = 'b'
            if quantile_labels:
                ax.plot(exp_quantiles, quantiles,
                        label=quantile_labels[i], c=c, alpha=0.5, linewidth=2.2)
            else:
                ax.plot(exp_quantiles, quantiles,
                        c=c, alpha=0.5, linewidth=2.2)
        ax.set_ylabel("Observed $p$-value")
        ax.set_xlabel("Expected $p$-value")
        if title:
            ax.title(title)
        ax.axis([-0.025, 1.025, -0.025, 1.025])
        if quantile_labels:
            fontProp = matplotlib.font_manager.FontProperties(size=10)
            # , handlelen=0.05, pad=0.018)
            ax.legend(loc=2, numpoints=2, markerscale=1, prop=fontProp)
        if text:
            f.text(0.05, 0.9, text)
        if plot_label:
            f.text(-0.151, 1.04, plot_label, fontsize=14)
        if storeFig == False:
            return
        if png_file != None:
            f.savefig(png_file)
        if pdf_file != None:
            f.savefig(pdf_file, format='pdf')

    def plot_simple_qqplots_pvals(self, png_file_prefix, pvals_list, result_labels=None, line_colors=None, num_dots=1000, title=None, max_neg_log_val=5):
        """
        Plots both log QQ-plots and normal QQ plots.
        """
        qs = []
        log_qs = []
        for pvals in pvals_list:
            qs.append(self.get_quantiles(pvals, num_dots))
            log_qs.append(self.get_log_quantiles(
                pvals, num_dots, max_neg_log_val))
        self.simple_qqplot(qs, png_file_prefix + '_qq.png', quantile_labels=result_labels,
                           line_colors=line_colors, num_dots=num_dots, title=title)
        self.simple_log_qqplot(log_qs, png_file_prefix + '_log_qq.png', quantile_labels=result_labels,
                               line_colors=line_colors, num_dots=num_dots, title=title, max_val=max_neg_log_val)


class ancmap():
    '''
    Mapping module, which maps snps to genes and computes the p-value at gene level.
    '''

    def readGWAS(self):
        '''
        Reads the GWAS file of format SNP - Pv - Ancentryy proportions if available
        '''
        self.assoc_map = {}
        self.assoc_SD = {}
        self.assoc_eff = {}
        logger.info('Reading ... %s' % self.assocFile)
        for line in fileinput.input(self.assocFile):
            data = line.split()
            if len(data) < 3:
                sys.stderr.write('\nError => Your GWA file does not match with your parameters. \"\n')
                self.terminate()
            if fileinput.lineno() == 1:
                if data[0] != 'SNP' or data[1] != 'P' or data[2] != 'Beta' or data[3] != 'SD':
                    logger.info('\nAssociation file not in valide format!')
                    logger.info('The first row should be SNP, should only contains the SNP, effect size (Beta) and Standard Error of Beta (SD), and the file should be SPACE-delimited or TAB-delimited')
                    self.terminate()
            else:
                if len(data) != 0:
                    snp = data[0]
                    for i in data[1:-2]:
                        if 0 < float(i) > 1:
                            logger.info('\nInvalid value in your GWA dataset!\nError => Failed to process the input!\n')
                            self.terminate()
                            sys.exit(1)
                    if 'nan' in data or data[1] in [".0","1"]:
                        pass
                    else:
                        self.assoc_map[snp] = [float(data[1])]
                        self.assoc_eff[snp] = [float(data[2])]
                        self.assoc_SD[snp] = [float(data[-1])]
                else:
                    logger.info('\nInvalid value in your GWA dataset!\nError => Failed to process the input!\n')
                    self.terminate()
                    sys.exit(1)
        
        dirList = os.listdir(self.assocStudy)
        self.study_names = []
        self.study_names.append(self.assocFile.split('.')[0].split("/")[1])

        for fname in dirList:
            self.study_names.append(fname.split('.')[0])
            for line in fileinput.input(self.assocStudy+fname):
                data = line.split()
                if fileinput.lineno() == 1:
                    if data[0] != 'SNP' or data[1] != 'Beta' or data[2] != 'SD':
                        logger.info('\nAssociation file from included previous study is not in valide format!')
                        logger.info('The first row should be \'SNP P\', should only contains the SNP, effect size (Beta) and Standard Error of Beta (SD), and the file should be SPACE-delimited or TAB-delimited')
                        self.terminate()
                        sys.exit(1)
                else:
                    try:
                        for i in data[1:]:
                            isinstance(float(i), (float))
                    except IndexError, TypeError:
                        sys.stderr.write('Error => EMILE:Failed to process the input, check the parameters!\n\n')
                        self.terminate()
                        sys.exit(1)

                    if data[0] in self.assoc_eff:
                        if 'nan' in data or data[1] in [".0","1"]:
                            self.assoc_map[data[0]]
                            self.assoc_eff[data[0]]
                            self.assoc_SD[data[0]]    
                        else:
                            P = self.assoc_eff[data[0]]
                            P.append(float(data[1]))
                            Q = self.assoc_SD[data[0]]
                            Q.append(float(data[-1]))
                    else:
                        sys.stderr.write('\n\nDiscarding SNP %s\n\n'%data[0])
                       
        logger.info('Including ... %s SNPs with associated Pvalue' %str(len(self.assoc_map)))
        output = open(self.outfolder+'assoc_map.pkl', 'wb')
        pickle.dump(self.assoc_map, output)
        output.close() 

    def prereadAffy(self):
        '''
        Read Affymetrix file and return the min distance of the the smallest pvalue in Assoc file
        '''
        logger.info('\nAjusting the boundary distance')
        minDist = 100000000
        for line in fileinput.input(self.affyFile):
            data1 = line.split()
            if fileinput.lineno() > 1:
                if len(data1) == 4:
                    snp = data1[1]
                    dist = data1[-2]
                elif len(data1) == 3:
                    snp = data1[0]
                    dist = data1[-2]
                if snp == self.minSNP:
                    if minDist > float(dist):
                        minDist = float(dist)
        if 100000 > minDist < 1000 * self.boundary:
            pass
        else:
            self.boundary = minDist/1000

    def readAffy(self):
        '''
        Reads the affymetrix File of format AffyID - SNP - Dist - Gene
        '''
        self.gene_map = {}
        self.affy_map1 = {}
        affy_map2 = {}
        affy_map3 = {}
        self.gene_map1 = {}
        self.Disease_gene = []
        self.gene_map2 = {}
        self.disease_name = " "
        for line in fileinput.input(self.gene_disease):
            data = line.split()
            if "ECR777" in data:
                S = data[1:]
                idx = S.index("ECR777")
                K = data[0]
                if K == self.disease:
                    self.disease_name = S[:idx]
                    self.Disease_gene = self.Disease_gene + S[idx+1:]
            else:
                logger.info('Disease-gene file is in bad format')
                self.terminate()
                raise SystemExit

        for line in fileinput.input(self.gene_disease):
            data = line.split()
            if "ECR777" in data:
                S = data[1:]
                idx = S.index("ECR777")
                K = data[0]
                if K != self.disease:
                    if len(self.disease_name) == 1:
                        if self.disease_name[0] in S:
                            self.Disease_gene = self.Disease_gene + S[idx+1:]
                    elif len(self.disease_name) >= 2:
                        if self.disease_name[0] in S and self.disease_name[1] in S:
                            self.Disease_gene = self.Disease_gene + S[idx+1:]

        for line in fileinput.input(self.affyFile):
            data1 = line.split()
            if len(data1) == 4:
                snp = data1[1]
                gene = data1[-1]
                dist = data1[-2]
                if snp in self.assoc_map:
                    if gene in self.affy_map1:
                        f2 = affy_map2[gene]
                        f2.append([snp, float(dist)])
                        if float(dist) < 1000 * self.boundary:
                            f = self.gene_map[gene]
                            g = self.gene_map2[gene]
                            h = self.gene_map1[gene]
                            f.add(self.assoc_map[snp][0])
                            g.append(assoc_SD[snp])
                            h.append(assoc_eff[snp])
                            f1 = self.affy_map1[gene]
                            f1.add(snp)
                        elif float(self.assoc_map[snp][0]) < self.Pcutoff:
                            f = self.gene_map[gene]
                            f.add(self.assoc_map[snp][0])
                        else:
                            affy_map2[gene] = [[snp, float(dist)]]
                            self.affy_map1[gene] = set([snp])
                            self.gene_map[gene] = set([self.assoc_map[snp]])
                            self.gene_map1[gene] = [self.assoc_eff[snp]]
                            self.gene_map2[gene] = [self.assoc_SD[snp]]

            elif len(data1) == 3:
                if fileinput.lineno() > 1:
                    snp = data1[0]
                    gene = data1[-1]
                    dist = data1[-2]
                    if snp in self.assoc_eff:
                        if gene in self.affy_map1:
                            f2 = affy_map2[gene]
                            f2.append([snp, float(dist)])
                            if snp in self.assoc_map:
                                if float(dist) < 1000 * self.boundary:
                                    f = self.gene_map[gene]
                                    g = self.gene_map2[gene]
                                    h = self.gene_map1[gene]
                                    f.add(self.assoc_map[snp][0])
                                    g.append(self.assoc_SD[snp])
                                    h.append(self.assoc_eff[snp])
                                    f1 = self.affy_map1[gene]
                                    f1.add(snp)
                                elif float(self.assoc_map[snp][0]) < self.Pcutoff:
                                    f = self.gene_map[gene]
                                    f.add(self.assoc_map[snp][0])
                                    f1 = self.affy_map1[gene]
                                    f1.add(snp)
                                elif gene in self.Disease_gene:
                                    f = self.gene_map[gene]
                                    g = self.gene_map2[gene]
                                    h = self.gene_map1[gene]
                                    f.add(self.assoc_map[snp][0])
                                    g.append(self.assoc_SD[snp])
                                    h.append(self.assoc_eff[snp])
                                    f1 = self.affy_map1[gene]
                                    f1.add(snp)
                            else:
                                if float(dist) < 1000 * self.boundary:
                                    g = self.gene_map2[gene]
                                    h = self.gene_map1[gene]
                                    g.append(self.assoc_SD[snp])
                                    h.append(self.assoc_eff[snp])
                                    f1 = self.affy_map1[gene]
                                    f1.add(snp)

                                elif gene in self.Disease_gene:
                                    g = self.gene_map2[gene]
                                    h = self.gene_map1[gene]
                                    g.append(self.assoc_SD[snp])
                                    h.append(self.assoc_eff[snp])
                                    f1 = self.affy_map1[gene]
                                    f1.add(snp)
                        else:
                            affy_map2[gene] = [[snp, float(dist)]]
                            self.affy_map1[gene] = set([snp])
                            self.gene_map[gene] = set([self.assoc_map[snp][0]])
                            self.gene_map1[gene] = [self.assoc_eff[snp]]
                            self.gene_map2[gene] = [self.assoc_SD[snp]]
                else:
                    if data1[0] != "SNP":
                        logger.info('Affymetrix file not in valide format!')
                        logger.info(
                            'Should include either AffyID-SNP-Dist-Gene or SNP-Dist-Gene columns')
                        self.terminate()
                        raise SystemExit

        for gene in affy_map2:
            tmp = sorted(affy_map2[gene],
                         key=lambda a_entry: float(a_entry[-1]))
            affy_map3[gene] = tmp[0]
        output = open(self.outfolder+'affy_map1.pkl', 'wb')
        pickle.dump(self.affy_map1, output)
        output.close()
        output1 = open(self.outfolder+'affy_map3.pkl', 'wb')
        pickle.dump(affy_map3, output1)
        output1.close()

        output2 = open(self.outfolder+'gene_map.pkl', 'wb')
        pickle.dump(self.gene_map, output2)
        output2.close()

        output3 = open(self.outfolder+'gene_map1.pkl', 'wb')
        pickle.dump(self.gene_map1, output3)
        output3.close()

        output4 = open(self.outfolder+'gene_map2.pkl', 'wb')
        pickle.dump(self.gene_map2, output4)
        output4.close()

        logger.info('Genes mapped ... %s Genes' % str(len(self.gene_map)))
        #self.gene_map = gene_map;self.gene_map1 = gene_map1;self.gene_map2 = gene_map2
        self.Disease_gene = []
        affy_map2.clear()
        affy_map3.clear()  # self.affy_map1 = affy_map1

    def posterior_prob(self, effect, precision):
        '''
        Computing the precision and variance for posterior prob. that the observed effect size is true.
        '''
        tmp = []  # the posterior of the effect size is true from each studies.
        # list contains the posterior of the ancestry is true from each studies.
        M_VALUE = [];V_var1 = []  # list contains all variances

        """ Computing the precision and variance for posterior prob. that the observed effect size is true"""
        for de in range(len(effect)):
            Var1 = 0;X1 = 0;Y1 = 0
            for d in range(len(effect)):
                if de != d:
                    S = float(effect[de])
                    v1 = precision[de]
                    Var1 = Var1 + v1
                    if v1 == 0.0:
                        W1 = 1  # precision
                        X1 = X1 + W1*S
                        Y1 = Y1 + W1
                    else:
                        W1 = float(Var1)  # precision
                        X1 = X1 + W1*S
                        Y1 = Y1 + W1
            M_VALUE.append(X1/Y1)
            V_var1.append(Var1)
        tmp_value = []

        # Computing the posterior prob. that the observed effect size is true from Normal distribution

        for j in range(len(effect)):
            var_x = (1.0/precision[j]) + (1.0/V_var1[j])
            V = 1.0/precision[j]
            b = self.normpdf(effect[j], 0.0, V)
            a = self.normpdf(effect[j], M_VALUE[j], var_x)
            anc_E = float(0.5*a)/float((1-0.5)*b+0.5*a)
            tmp_value.append(anc_E)
        M_VALUE = [];V_var1 = []
        return tmp_value

    def gene_all(self):
        '''
        Computes the combine p-values using different methods.
        '''
        gene2Pv_ = [];gene2Pv = {}
        file1 = open(self.outfolder+"GENE_RESULT/"+"gene2Pv.txt", "wt")

        for gene in self.gene_map:
            gene_p = [float(i) for i in list(self.gene_map[gene])]
            tmp = list(gene_p)
            tmp = sorted(gene_p)
            p = self.stouffer_liptak(tmp)
            Anto_cor = self.estimated_autocorrelation(tmp)
            sigma = self.gen_sigma_matrix(tmp, Anto_cor)
            Z_cor = float(self.z_score_combine(tmp, sigma))
            gene2Pv[gene] = [float(p), float(Z_cor)]
            gene2Pv_.append([gene, float(p), float(Z_cor)])
        gene2Pv_ = sorted(gene2Pv_, key=lambda a_key: float(a_key[-1]), reverse=False)
        tmp = [float(c[-1]) for c in gene2Pv_]
        gene2AdjP = self.adjustPValues(tmp, self.adjustP, None)
        gene2Qval = self.computeQValues(tmp, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)
        file1.writelines("Gene"+"\t"+"Liptak"+"\t"+"Q_Liptak" +"\t"+"Fisher"+"\t"+"Q_Fisher"+"\n")
        tmp = []
        tmp = [float(d[2]) for d in gene2Pv_]
        gene2AdjP1 = self.adjustPValues(tmp, self.adjustP, None)
        gene2Qval1 = self.computeQValues(tmp, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)

        for lst in gene2Pv_:
            idx = gene2Pv_.index(lst)
            file1.write("\t".join([str(lst[0])])+"\t"+"\t".join([str(gene2AdjP[idx]), str(gene2Qval[idx]), str(gene2AdjP1[idx]), str(gene2Qval1[idx])])+"\n")
        file1.close()

        gene_p = open(self.outfolder+"gene2Pv.pkl", 'wb')
        pickle.dump(gene2Pv, gene_p)
        gene_p.close()

        gene2Pv_ = []
        gene2Pv.clear()
        self.gene_map.clear()

    def sub_effect_size(self, sub_map1, sub_map2):
        '''
        Function to combine the effect size and standard error of SNPs with each gene.
        The function returns two dictionary of gene level effect size and their precisions
        '''

        self.data_eff1 = {}  # gene level effect size
        self.data_err1 = {}  # gene level standard error of the effect size
        for g in sub_map1:
            effects = np.array(sub_map1[g])  # convert list to array
            if self.fast:
                precision = np.array(sub_map2[g], dtype=float)
            else:
                precision = 1.0/(np.array(sub_map2[g], dtype=float)**2)

            if len(precision) != len(effects) or len(precision) == 0 or len(effects) == 0:
                pass
            else:
                #;precisionprec = np.array([j**2 for j in error]);precision = np.array([ 1.0/float(i) for i in prec])
                # equation 4.2.4: true effect size at gene level
                num = np.sum(effects*precision, axis=0)/np.sum(precision, axis=0)
                self.data_eff1[g] = num
                self.data_err1[g] = np.sum(precision, axis=0)
        print("PAY ATTENTION: %d"%len(self.data_eff1))
        self.sub_meta(self.data_eff1, self.data_err1)

    def gene_effect_size(self):
        '''
        Function to combine the effect size and standard error of SNPs with each gene.
        The function returns two dictionary of gene level effect size and their precisions
        '''
        self.data_eff = {}  # gene level effect size
        self.data_err = {}  # gene level standard error of the effect size
        for g in self.gene_map1:
            effects = np.array(self.gene_map1[g])  # convert list to array
            error = np.array(self.gene_map2[g])  # convert list to array
            # print"\n",error,"effect:",effects
            precision = 1.0/(error**2)
            #;precisionprec = np.array([j**2 for j in error]);precision = np.array([ 1.0/float(i) for i in prec])
            # equation 4.2.4: true effect size at gene level
            num = np.sum(effects*precision, axis=0)/np.sum(precision, axis=0)
            self.data_eff[g] = num
            # 1.0/np.sqrt(np.sum(precision, axis=0)) # equation 4.2.5: relative precision of the true effect size at gene level
            self.data_err[g] = np.sum(precision, axis=0)
        logger.info('Performing Meta-analys for: %s genes from %s studies' %(str(len(self.gene_map1)), len(self.study_names)))
        gene_ef = open(self.outfolder+"gene2Eff.pkl", 'wb')
        gene_err = open(self.outfolder+"gene2Err.pkl", 'wb')

        pickle.dump(self.data_eff, gene_ef)
        pickle.dump(self.data_err, gene_err)
        gene_ef.close()
        gene_err.close()
        self.gene_map1.clear()
        self.gene_map2.clear()
        self.meta(self.data_eff, self.data_err)

    def meta(self, genes, stderror):

        self.gene_meta = {}
        self.meta_zscore = {}
        # this peace of code is to compute sigma_j for each gene j.
        for g in genes:
            G = np.array(genes[g], dtype=np.float)
            SD = np.array(np.sqrt(1.0/stderror[g], dtype=np.float))
            z_gene = np.array(G/SD, dtype=np.float)
            p_gene = []
            for z in z_gene:
                a = 1.0 - stats.chi2.cdf(100*abs(z), 1)
                b1 = stats.t.sf(100*np.abs(z), 1)*2
                b2 = stats.t.sf(100*np.abs(z), 1)
                p_gene.append(str(b2))

            M = np.sum(genes[g]*stderror[g])/np.sum(stderror[g])
            # Cochran heterogeneity statistic is chi2 stats
            Q = np.sum(stderror[g]*((genes[g]-M)**2))

            W = np.sum(stderror[g])
            W2 = np.sum(stderror[g]**2)/W
            dof = len(genes[g])-1
            if Q > dof:
                num = Q-(len(genes[g])-1)
                dem = W-W2
                delta = num/float(dem)  # First variance
            else:
                delta = 0.0
                I_square1 = 0.0
            sigma = 1.0/(stderror[g])
            sigma1 = sigma + delta
            sif = 1.0/(sigma + delta)
            sif2 = np.sum(sif)
            delta_num = np.sum((1.0/sigma1)*genes[g])
            sum_sigma = np.sum(sigma1)
            mu = float(delta_num)/sif2  # First gene effect size
            precision_mu = np.sum(1.0/sigma1)
            sd_mu = np.sqrt(1.0/np.sum(sif2))
            A0 = np.array(genes[g]/(np.sqrt(sigma1)), dtype=np.float)
            A1 = np.sum(np.array((1.0/sigma1)/A0, dtype=np.float))
            B1 = np.sqrt(np.sum(np.array(1.0/sigma1, dtype=np.float)))
            z1 = float(A1)/B1
            p1 = stats.t.sf(100*np.abs(z1), dof)*2

            a = np.sum((1.0/sigma1)*(genes[g]-mu))
            b = np.sum((1.0/sigma1))
            c = np.sum((1.0/sigma1)**2)/float(b)
            d = np.sum(sigma1**2)/np.sum(sigma1)
            Q1 = np.sum((1.0/sigma1)*((genes[g]-mu)**2))
            # 1 - t1[0]  #np.sum(sigma1*((genes[g]-mu)**2));t1 = list(r.pchisq(100*Q1,dof))
            p_chi1 = chisqprob(np.abs(Q1), dof)
            if Q1 > dof:
                num = Q1-(len(genes[g])-1)
                dem = b-c
                # float((a -(b -c)))/(np.sum(sigma1-d)) # Second variance
                delta1 = num/float(dem)
                f = np.sum(sigma1)-d
                I_square2 = 100*((num)/float(Q1))
            else:
                delta1 = 0.0
                I_square2 = 0.0

            sigma2 = 1.0/(sigma + delta1)
            delta_num2 = np.sum(sigma2*genes[g])
            sum_sigma2 = np.sum(sigma2)
            sig = sigma + delta1
            mu2 = float(delta_num2)/sum_sigma2  # Second gene effect size

            precision_mu2 = sum_sigma2
            sd_mu2 = np.sqrt(1.0/np.sum(sigma2))
            A0 = np.array(genes[g]/(np.sqrt(sig)), dtype=np.float)
            A1 = np.sum(np.array((1.0/sig)/A0, dtype=np.float))
            B1 = np.sqrt(np.sum(np.array(1.0/sig, dtype=np.float)))

            post_prob = self.posterior_prob(genes[g], stderror[g])

            delta_num3 = np.sum(sigma2*genes[g]*post_prob)
            sum_sigma3 = np.sum(sigma2*np.sqrt(post_prob))
            mu3 = float(delta_num3)/sum_sigma3  # Third gene effect size
            z2 = float(A1)/B1
            sd_mu3 = np.sqrt(1.0/np.sum(sigma2*np.sqrt(post_prob)))
            chi_q2 = (np.sum(post_prob)/(np.sqrt(np.sum(post_prob)))) * \
                (((mu3)**2)/np.sum(sig))
            p_chib = chisqprob(100*chi_q2, dof)
            p2 = stats.t.sf(100*np.abs(z2), dof)*2
            Z_s1 = mu2/sd_mu2
            Z_s2 = mu3/sd_mu3

            p_z1 = 1.0 - stats.chi2.cdf(abs(Z_s1), 1)
            p_z2 = 1.0 - stats.chi2.cdf(abs(Z_s2), 1)

            if I_square2 >= 1.0:
                I_square2 = 1.0

            self.gene_meta[g] = [mu, sd_mu, mu3, sd_mu3]
            self.meta_zscore[g] = [g, p1, p2, p_z1, Z_s1, Z_s2, str(round(p_chib, 6)), str(round(I_square2, 6)), str(round(Q1, 6)), str(round(
                p_chi1, 6)), str(round(100*delta1, 6))]+list(p_gene)+[str(round(abs(1-(h/float(np.sqrt(np.mean(post_prob))))), 6)) for h in list(post_prob)]

        D = self.meta_zscore.values()

        if int(self.inxed_sort) in [16, 17, 18]:
            NET2Pv_1 = sorted(D, key=lambda a_key: float(
                a_key[16]), reverse=False)
            #NET2Pv_1 = sorted(sorted(Z, key = lambda x : float(x[1])), key = lambda x : float(x[self.inxed_sort+len(self.study_names)]), reverse = True)
        else:
            #NET2Pv_1 = sorted(sorted(D, key=lambda x: float(x[1])), key=lambda x: float(x[19]), reverse=True)
            NET2Pv_1 = sorted(D, key= lambda a_key: float(a_key[1]), reverse=False)

        fix1 = open(self.outfolder+"GENE_RESULT/"+"Meta_Analysis.txt", "wt")
        fix2 = open(self.outfolder+"GENE_RESULT/"+"Meta_Input.txt", "wt")
        SUP_Mstudie = ["P_"+i for i in list(self.study_names)]+["M_"+i for i in list(self.study_names)]
        fix1.write("\t".join(["Gene"] + ["#Study", "P1", "Beta1", "SD1", "P2", "Beta2", "SD2","P_chi", "Zscore1", "Zscore2", "P_be", "I2", "Q", "P_Q", "TAU_SQUARE"]+SUP_Mstudie)+"\n")
        D = []
        for i in list(self.study_names):
            D.append(i)
            D.append("SD_"+str(i))
        fix2.write("\t".join(["Gene"]+D)+"\n")

        for s in range(len(NET2Pv_1)):
            g = NET2Pv_1[s][0]
            P = [str(self.meta_zscore[g][1]), str(round(self.gene_meta[g][0], 6)), str(round(self.gene_meta[g][1], 6))]
            P1 = [str(round(self.meta_zscore[g][2], 6)), str(round(self.gene_meta[g][2], 6)), str(round(self.gene_meta[g][3], 6))]
            P2 = [str(round(self.meta_zscore[g][3], 6)), str(round(self.meta_zscore[g][4], 6)), str(round(self.meta_zscore[g][5], 6)), self.meta_zscore[g][6]]+self.meta_zscore[g][7:]
            fix1.write("\t".join([str(g), str(len(self.study_names))]+P+P1+P2)+"\n")
            D = []
            for j in range(len(genes[g])):
                D.append(str(genes[g][j]))
                D.append(str(float(np.sqrt(1/stderror[g][j]))))
            fix2.write("\t".join([str(g)]+D)+"\n")

        fix1.close()
        fix2.close()
        D = []
        self.data_eff.clear()
        self.data_err.clear()
        self.meta_zscore.clear()
        self.meta_zscore.clear()
        genes.clear()
        NET2Pv_ = []
        NET2Pv_1 = []
        D = []
        Z = {}
        """
	fil  = fix1 = open(self.outfolder+"GENE_RESULT/"+"Meta_Analysis2.txt","wt")
	for line in fileinput.input(self.outfolder+"GENE_RESULT/"+"Meta_Analysis.txt"):
		data = line.split()
		if fileinput.lineno() > 1:
			if len(data) !=L:
				print("Bad result !")
			else:
				Z[data[0]] = [data[0]] +[ float(h) for h in data[1:]]
		else:
			L = len(data)
			fil.write(line)
	Z = Z.values()
	NET2Pv_1 = sorted(sorted(Z, key = lambda x : float(x[1])), key = lambda x : float(x[self.inxed_sort+len(self.study_names)]), reverse = True) 
	for des in NET2Pv_1:
		fil.write("\t".join([ str(j) for j in des])+"\n")
	fil.close()
	os.system("mv"+ " "+self.outfolder+"GENE_RESULT/"+"Meta_Analysis2.txt"+" "+self.outfolder+"GENE_RESULT/"+"Meta_Analysis.txt")
	"""

    def sub_meta(self, genes1, stderror1):
        # genelist=genes.keys()
        self.gene_meta1 = {}
        self.meta_zscore1 = {}
        # this peace of code is to compute sigma_j for each gene j.
        for g in genes1:
            G = np.array(genes1[g], dtype=np.float)
            SD = np.sqrt(1.0/np.array(stderror1[g], dtype=np.float))
            z_gene = np.array(G/SD, dtype=np.float)
            p_gene1 = []
            for z in z_gene:
                a = 1.0 - stats.chi2.cdf(100*abs(z), 1)
                a1 = 2*(1.0 - stats.chi2.cdf(abs(z), 1))
                a2 = chisqprob(np.abs(z), 1)
                b1 = stats.t.sf(np.abs(z), 1)*2
                b2 = stats.t.sf(np.abs(z), 1)
                #print"\nSUBNETWORK ","a: ", a,"a1: ","a2: ",a2,a1,"b1: ",b1,"b2: ",b2
                #if g in ["LDOC1L", "PRKCZ", "CBX7", "MKL1", "ASTN1"]:
                #   print"\nGENE: ", "a: ", a, "a1: ", "a2: ", a2, a1, "b1: ", b1, "b2: ", b2, "MAX===", max([a, a1, a2, b1, b2]), b2
                p_gene1.append(str(b2))

            # First step, computing the between-study variance
            M = np.sum(genes1[g]*stderror1[g])/float(np.sum(stderror1[g]))
            # Cochran heterogeneity statistic is chi2 stats
            Q = np.sum(stderror1[g]*((genes1[g]-M)**2))
            # moment[g]=Q  # Cochran heterogeneity statistic is chi2 stats
            W = np.sum(stderror1[g])
            W2 = np.sum(stderror1[g]**2)/W
            dof = len(genes1[g])-1

            if Q > dof:
                num = Q-(len(genes1[g])-1)
                dem = W-W2
                delta = num/float(dem)  # First variance
            else:
                delta = 0.0
                I_square1 = 0.0

            sigma = 1.0/(stderror1[g])
            sigma1 = sigma + delta
            sif = 1.0/(sigma + delta)
            sif2 = np.sum(sif)

            delta_num = np.sum((1.0/sigma1)*genes1[g])
            sum_sigma = np.sum(sigma1)
            mu = float(delta_num)/sif2  # First gene effect size
            precision_mu = np.sum(1.0/sigma1)
            sd_mu = np.sqrt(1.0/np.sum(sif))
            A0 = np.array(genes1[g]/(np.sqrt(sigma1)), dtype=np.float)
            A1 = np.sum(np.array((1.0/sigma1)/A0, dtype=np.float))
            B1 = np.sqrt(np.sum(np.array(1.0/sigma1, dtype=np.float)))
            z1 = float(A1)/B1
            p1 = stats.t.sf(np.abs(z1), dof)*2
            a = np.sum((1.0/sigma1)*(genes1[g]-mu))
            b = np.sum((1.0/sigma1))
            c = np.sum((1.0/sigma1)**2)/float(b)
            d = np.sum(sigma1**2)/np.sum(sigma1)
            Q1 = np.sum((1.0/sigma1)*((genes1[g]-mu)**2))
            # 1 - t1[0]  #np.sum(sigma1*((genes[g]-mu)**2));t1 = list(r.pchisq(100*Q1,dof))
            p_chi1 = chisqprob(np.abs(Q1), dof)

            if Q1 > dof:
                num = Q1-dof
                dem = b-c
                # float((a -(b -c)))/(np.sum(sigma1-d)) # Second variance
                delta1 = float(num)/float(dem)
                f = np.sum(sigma1)-d
                I_square2 = 100*((num)/float(Q1))
            else:
                delta1 = 0.0
                I_square2 = 0.0

            sigma2 = 1.0/(sigma + delta1)
            delta_num2 = np.sum(sigma2*genes1[g])
            sum_sigma2 = np.sum(sigma2)
            sig = sigma + delta1

            mu2 = float(delta_num2)/sum_sigma2  # First gene effect size
            precision_mu2 = sum_sigma2
            sd_mu2 = np.sqrt(1.0/np.sum(sigma2))
            A0 = np.array(genes1[g]/(np.sqrt(sig)), dtype=np.float)
            A1 = np.sum(np.array((1.0/sig)/A0, dtype=np.float))
            B1 = np.sqrt(np.sum(np.array(1.0/sig, dtype=np.float)))
            # Z-score for meta-analysis

            post_prob = self.posterior_prob(genes1[g], stderror1[g])
            delta_num3 = np.sum(sigma2*genes1[g]*post_prob)
            sum_sigma3 = np.sum(sigma2*np.sqrt(post_prob))
            mu3 = float(delta_num3)/sum_sigma3  # Third gene effect size
            z2 = float(A1)/B1  # mu2/float(np.sqrt(np.sum(sigma + delta1)))
            sd_mu3 = np.sqrt(1.0/np.sum(sigma2*np.sqrt(post_prob)))
            chi_q2 = (np.sum(post_prob)/(np.sqrt(np.sum(post_prob)))) * \
                (((mu3)**2)/np.sum(sig))
            p_chib = 1.0 - stats.chi2.cdf(abs(chi_q2), 1)
            p_chib2 = chisqprob(chi_q2, dof)
            p2 = stats.t.sf(np.abs(z2), dof)*2

            Z_s1 = mu2/sd_mu2
            Z_s2 = mu3/sd_mu3

            p_z1 = 1.0 - stats.chi2.cdf(abs(Z_s1), 1)
            p_z2 = 1.0 - stats.chi2.cdf(abs(Z_s2), 1)
            if I_square2 > 1.0:
                I_square2 = 1.0

            self.gene_meta1[g] = [mu, sd_mu, mu3, sd_mu3]
            self.meta_zscore1[g] = [g, p1, p2, p_z1, Z_s1, Z_s2, str(round(p_chib, 6)), str(round(I_square2, 6)), str(round(Q1, 6)), str(round(p_chi1, 6)), str(round(100*delta1, 6))]+list(p_gene1)+[str(round(abs((h/float(np.sqrt(np.mean(post_prob))))), 6)) for h in list(post_prob)]

            #self.gene_meta1[g] = [mu,sd_mu,mu3,sd_mu3]; self.meta_zscore1[g] = [g,p1,p2,p_z1,Z_s1,Z_s2,str(round(p_chib,6)),str(round(I_square2,6)), str(round(Q1,6)),str(round(p_chi1,6)),str(round(100*delta1,6))]+list(p_gene1)+[str(round(abs(1-(h/float(np.sqrt(np.mean(post_prob))))),6)) for h in list(post_prob)]

        if len(self.meta_zscore1) == 0:
            pass
        else:
            D = self.meta_zscore1.values()
            #NET2Pv_1 = sorted(D, key= lambda a_key: float(a_key[1]), reverse=False)
            NET2Pv_1 = sorted(sorted(D, key=lambda x: float(x[1])), key=lambda x: float(x[-3]), reverse=True)
            fix1 = open(self.outfolder+"NETWORK_RESULT/" +
                        "Meta_Analysis.txt", "a")
            fix2 = open(self.outfolder+"NETWORK_RESULT/"+"Meta_Input.txt", "a")

            if self.head == True:
                SUP_Mstudie = ["P_"+i for i in list(self.study_names)]+["M_"+i for i in list(self.study_names)]
                #SUP_Mstudie = ["P_STUDIES","M_STUDIES"]
                fix1.write("\t".join(["Hub"] + ["#Study", "P1", "Beta1", "SD1", "P2", "Beta2", "SD2","P_chi", "Zscore1", "Zscore2", "P_be", "I2", "Q", "P_Q", "TAU_SQUARE"]+SUP_Mstudie)+"\n")
                D = []
                for i in list(self.study_names):
                    D.append(i)
                    D.append("SD_"+str(i))
                fix2.write("\t".join(["Gene"]+D)+"\n")
                self.head = False
            else:

                for s in range(len(NET2Pv_1)):
                    g = NET2Pv_1[s][0]

                    P = [str(self.meta_zscore1[g][1]), str(round(self.gene_meta1[g][0], 6)), str(round(self.gene_meta1[g][1], 6))]
                    P1 = [str(round(self.meta_zscore1[g][2], 6)), str(round(self.gene_meta1[g][2], 6)), str(round(self.gene_meta1[g][3], 6))]
                    P2 = [str(round(self.meta_zscore1[g][3], 6)), str(round(self.meta_zscore1[g][4], 6)), str(round(self.meta_zscore1[g][5], 6)), self.meta_zscore1[g][6]]+self.meta_zscore1[g][7:]
                    fix1.write("\t".join([str(g), str(len(self.study_names))]+P+P1+P2)+"\n")
                    D = []
                    for j in range(len(genes1[g])):
                        D.append(str(genes1[g][j]))
                        D.append(str(float(np.sqrt(1.0/stderror1[g][j]))))
                    fix2.write("\t".join([str(g)]+D)+"\n")
                fix1.close()
                fix2.close()
                D = []
                self.data_eff1.clear()
                self.data_err1.clear()
                self.meta_zscore1.clear()
                self.meta_zscore1.clear()
                genes1.clear()
                stderror1.clear()
                NET2Pv_ = []
                NET2Pv_1 = []
                D = []

    def simes(self):
        '''
        Computes the gene p-value using the simes method.
        '''
        gene2Pv = {}
        gene2Pv_ = []
        file1 = open(self.outfolder+"GENE_RESULT/"+"gene2Pv.txt", "wt")
        file1.writelines("## P-value for the gene using Simes Method ##\n")
        file1.writelines("Gene"+"\t"+"Pvalues"+"\t" +"Adjusted_Pvalues"+"\t"+"Qvalues"+"\n")
        for gene in self.gene_map:
            gene_p = [float(i) for i in list(self.gene_map[gene])]
            tmp = sorted(gene_p)
            pos = range(1, len(tmp)+1)
            tmp2 = (np.array(tmp)*(len(tmp)+1))/np.array(pos)
            if min(tmp2) > 1.0:
                gene2Pv[gene] = 0.90
                gene2Pv_.append([gene, 0.90])
            else:
                gene2Pv[gene] = min(tmp2)
                gene2Pv_.append([gene, min(tmp2)])
        gene2Pv_ = sorted(gene2Pv_, key=lambda a_key: float(
            a_key[1]), reverse=False)
        tmp = []
        for lst in gene2Pv_:
            tmp.append(lst[1])
        tmp.sort()
        gene2AdjP = self.adjustPValues(tmp, self.adjustP, None)
        gene2Qval = self.computeQValues(tmp, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)
        for lst in gene2Pv_:
            idx = gene2Pv_.index(lst)
            file1.writelines(lst[0]+"\t"+str(lst[1])+"\t" +str(gene2AdjP[idx])+"\t"+str(gene2Qval[idx])+"\n")
        file1.close()
        tmp = []
        output = open(self.outfolder+'gene2Pv.pkl', 'wb')
        pickle.dump(gene2Pv, output)
        gene2Pv = {}

    def fisher(self):
        '''
        Computes the gene p-value using fisher's method.
        '''
        gene2Pv = {}
        gene2Pv_ = []
        file1 = open(self.outfolder+"GENE_RESULT/"+"gene2Pv.txt", "wt")
        file1.writelines("## P-value for the nene using Fisher's Method ##\n")
        file1.writelines("Gene"+"\t"+"Pvalues"+"\t" +"Adjusted_Pvalues"+"\t"+"Qvalues"+"\n")
        for gene in self.gene_map:
            gene_p = [float(i) for i in list(self.gene_map[gene]) if float(i) != 0]
            if HAS_R:
                chi2 = -2*sum(np.log(np.array(gene_p)))
                t = list(r.pchisq(chi2, 2*len(gene_p)))
                p = 1 - t[0]
            else:
                chi2 = -2*sum(np.log(np.array(gene_p)))
                p = chisqprob(chi2, (len(gene_p)-1))
            if p >= 1.0:
                gene2Pv[gene] = 0.90
                gene2Pv_.append([gene, 0.90])
            else:
                gene2Pv[gene] = p
                gene2Pv_.append([gene, p])
        gene2Pv_ = sorted(gene2Pv_, key=lambda a_key: float(
            a_key[1]), reverse=False)
        tmp = []
        for lst in gene2Pv_:
            tmp.append(lst[1])
        gene2AdjP = self.adjustPValues(tmp, self.adjustP, None)
        gene2Qval = self.computeQValues(
            tmp, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)

        for lst in gene2Pv_:
            idx = gene2Pv_.index(lst)
            file1.writelines(lst[0]+"\t"+str(lst[1])+"\t" +
                             str(gene2AdjP[idx])+"\t"+str(gene2Qval[idx])+"\n")
        file1.close()
        tmp = []
        output = open(self.outfolder+'gene2Pv.pkl', 'wb')
        pickle.dump(gene2Pv, output)
        gene2Pv = {}

    def smallest(self):
        '''
        Computes the gene p-value using the smallest p-value of all the snps in a given gene.
        '''
        gene2Pv = {}
        gene2Pv_ = []
        file1 = open(self.outfolder+"GENE_RESULT/"+"gene2Pv.txt", "wt")
        file1.writelines("## P-value for the gene using Smallest Method ##\n")
        file1.writelines("Gene"+"\t"+"Pvalues"+"\t" +
                         "Adjusted_Pvalues"+"\t"+"Qvalues"+"\n")

        for gene in self.gene_map:
            gene_p = [float(i) for i in list(self.gene_map[gene])]
            gene2Pv[gene] = min(gene_p)
            gene2Pv_.append([gene, min(gene_p)])
        gene2Pv_ = sorted(gene2Pv_, key=lambda a_key: float(
            a_key[1]), reverse=False)
        tmp = []
        for lst in gene2Pv_:
            tmp.append(lst[1])
        gene2AdjP = cValues(tmp, self.adjustP, None)
        gene2Qval = self.computeQValues(
            tmp, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)

        for lst in gene2Pv_:
            idx = gene2Pv_.index(lst)
            file1.writelines(lst[0]+"\t"+str(lst[1])+"\t" +
                             str(gene2AdjP[idx])+"\t"+str(gene2Qval[idx])+"\n")
        file1.close()
        output = open(self.outfolder+'gene2Pv.pkl', 'wb')
        pickle.dump(gene2Pv, output)
        gene2Pv = {}

    def FDR(self, ):
        '''
        Computes the gene p-value using the gene-wise FDR method.
        '''
        gene2Pv = {}
        gene2Pv_ = []
        file1 = open(self.outfolder+"GENE_RESULT/"+"gene2Pv.txt", "wt")
        file1.writelines(
            "### Computing the p-value for the gene using gene-wise FDR value ###\n")
        file1.writelines("Gene"+"\t"+"Pvalues"+"\t" +
                         "Adjusted_Pvalues"+"\t"+"Qvalues"+"\n")
        for gene in self.gene_map:
            gene_p = [float(i) for i in list(self.gene_map[gene])]
            tmp = list(R0.r['p.adjust'](
                R0.FloatVector(gene_p), method='bonferroni'))
            gene2Pv[gene] = min(tmp)
            gene2Pv_.append([gene, min(tmp)])
        gene2Pv_ = sorted(gene2Pv_, key=lambda a_key: float(
            a_key[1]), reverse=False)
        tmp = []
        for lst in gene2Pv_:
            tmp.append(lst[1])
        gene2AdjP = self.adjustPValues(tmp, self.adjustP, None)
        gene2Qval = self.computeQValues(
            tmp, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)

        for lst in gene2Pv_:
            idx = gene2Pv_.index(lst)
            file1.writelines(lst[0]+"\t"+str(lst[1])+"\t" +str(gene2AdjP[idx])+"\t"+str(gene2Qval[idx])+"\n")
        file1.close()
        tmp = []
        output = open(self.outfolder+'gene2Pv.pkl', 'wb')
        pickle.dump(gene2Pv, output)
        gene2Pv = {}

    def mmplot(self, mplot, mpath):

        # Cleaning existing file as later the script open the file in append mode
        fix1 = open(mpath+"Meta_Analysis.txt")
        fix2 = open(mpath+"Meta_Input.txt")  # Cleaning existing file
        fin = open(mpath+"input.txt", "wt")
        fin1 = open(mpath+"output.txt", "wt")

        F_A = fix1.readlines()
        F_I = fix2.readlines()

        for line in F_I[1:6]:
            fin.write(line)
        fin.close()

        for line in F_A[1:6]:
            data = line.split()
            fin1.write("\t".join(data)+"\n")
        fin1.close()

        rsids_0 = np.loadtxt(mpath+"output.txt", dtype='str')
        intab_0 = np.loadtxt(mpath+"input.txt", dtype='str')
        List_GENE = self.head_rdid(rsids_0)
        outtab_0 = np.array(np.loadtxt(mpath+"output.txt", delimiter=' ', dtype='str'))
        studies = np.array(self.study_names, dtype='|S5')
        total_studies = len(studies)
        for i in range(len(List_GENE)):
            rsid = List_GENE[i]
            rsids = np.array(['RSID', rsids_0[i][0]])
            intab = intab_0[i]
            outtab = outtab_0[i]
            genename = List_GENE[i]
            pmplot_file = mpath+rsid+".pdf"

            rs_index_list = [1]
            if (len(intab.shape) == 1):
                intab.shape = (1, len(intab))
            rs_index = 1
            sels = self.get_valid_index(intab, rs_index)
            intab = self.const_intab(intab, rs_index)
            outtab = self.const_outtab(outtab, rs_index, total_studies, sels)
            nstudy = int(float(outtab[1]))

            height = 4 + 0.23*nstudy
            np.savetxt(mpath+"tmpoutput.txt", outtab, fmt='%s')
            np.savetxt(mpath+"tmpinput.txt", intab, fmt='%s')
            np.savetxt(mpath+"tmpgenenames.txt",np.array([genename]), fmt='%s')
            np.savetxt(mpath+"tmpstudies.txt", studies[sels], fmt='%s')
            np.savetxt(mpath+"tmpstudyorder.txt",range(1, (len(sels)+1)), fmt='%s')
            try:
                
                cmd = "R CMD BATCH --no-save --no-restore '--args"+" " + mpath+"tmpoutput.txt"+" "+mpath+"tmpinput.txt"+" "+mpath+"tmpgenenames.txt" + \
                " "+mpath+"tmpstudies.txt"+" "+mpath+"tmpstudyorder.txt " + pmplot_file + \
                " " + str(height) + " " + mplot +"' "+ self.pathscpt+"forestpmplot.R"+" "+self.wkdir+"2.Rout"
                os.system(cmd)
                PATH_CWD = self.wkdir #os.getcwd()+"/"
                if os.path.exists(PATH_CWD+"2.Rout"):
                    fp = open(PATH_CWD+"2.Rout")
                elif os.path.exists(os.getcwd()+"/"+"2.Rout"):
                    os.system("mv"+" "+os.getcwd()+"/"+"2.Rout"+" "+PATH_CWD+"2.Rout")
                lines = fp.readlines()
                if ("proc.time" in lines[-3]):
                    self.cleanup(mpath)
                    print(pmplot_file + " is successfully generated.")
                    cmd = "rm"+" "+PATH_CWD + "2.Rout"
                    os.system(cmd)
                else:
                    self.cleanup_err(mpath)
                    print("Problem occurred, while generating " + pmplot_file)
                    os.system("cat"+" "+PATH_CWD + "2.Rout")
            except (RuntimeError, TypeError, NameError, IndexError):
                sys.stderr.write('Error => in run Forest analysis ....!\n')
            finally:
                pass

class ancLD():
    '''
    Estimates the egde weight using the LD between snps in genes and constructs the LD-weighted PPI network.
    '''

    def __init__(self):
        return None

    def data_comb(self, snpFile, genoFile):
        '''
        Link genotype data to relative SNP and limit possible SNP to use for the program.
        Return: a dict, {snp:genotype}
        '''
        map_tmp = {}
        for line in fileinput.input(snpFile):
            data = line.split()
            map_tmp[fileinput.lineno()] = data
        self.tagget_dict = {}
        pos = {}
        for line in fileinput.input(genoFile):
            data = line.split()
            try:
                m_snp = map_tmp[fileinput.lineno()][0]
                self.tagget_dict[m_snp] = data[0]
                pos[m_snp] = int(map_tmp[fileinput.lineno()][3])
            except:
                self.terminate()
                raise exceptions.SystemError(
                    'Failed to process, key not found ')
        del map_tmp
        return self.tagget_dict

    def ZscoreLDGene(self, networkFile, tagget_dict, opt):
        '''
        Computes gene-gene-LD using the zscore of LDs between snps in genes
        networkFile: a string, the PPI network file
        tagget_dict: a dictionary, the tagged population genotype file
        '''
        logger.info('Start creating the network Gene-Gene-LD ...')
        rows_LD = {}
        tmp = []
        H_gene = []
        myset = set()
        keep_snp_ld = {}
        file5 = open(self.outfolder+"GENE_RESULT/gene2geneLD.net", "wt")
        for line in fileinput.input(networkFile):
            data2 = line.split()
            gene1 = data2[0]
            gene2 = data2[1]
            if gene1 in self.affy_map1 and gene2 in self.affy_map1 and gene1 != "---" and gene2 != "---":
                snp1 = self.affy_map1[gene1]
                snp2 = self.affy_map1[gene2]
                tmp = []
                for subset in it.product(list(snp1), list(snp2)):
                    SNP_list = list(subset)
                    if SNP_list[0] in tagget_dict and SNP_list[1] in tagget_dict:
                        if SNP_list[0] != SNP_list[1]:
                            if SNP_list[0]+":"+SNP_list[1] in keep_snp_ld:
                                tmp.append(
                                    keep_snp_ld[SNP_list[0]+":"+SNP_list[1]])
                            elif SNP_list[1]+":"+SNP_list[0] in keep_snp_ld:
                                tmp.append(
                                    keep_snp_ld[SNP_list[1]+":"+SNP_list[0]])
                            else:
                                LD, cor = self.calc_rsq(
                                    tagget_dict[SNP_list[0]], tagget_dict[SNP_list[1]], SNP_list[0], SNP_list[1])
                                if LD > 1.0:
                                    tmp.append(0.95)
                                    keep_snp_ld[SNP_list[0] +
                                                ":"+SNP_list[1]] = 0.95
                                else:
                                    tmp.append(LD)
                                    keep_snp_ld[SNP_list[0] +
                                                ":"+SNP_list[1]] = LD
                        else:
                            tmp.append(0.95)
                    else:
                        if gene1 in self.diseaseGenes or gene2 in self.diseaseGenes:
                            tmp.append(0.01)
                if len(tmp) != 0:
                    n = len(tmp)-tmp.count(0.0)
                    a = np.mean(tmp)
                    if a > 1.0:
                        b = 0.95
                    elif a < 0.0:
                        b = max(tmp)
                        if b < 0:
                            b = round(1-abs(b), 2)
                    else:
                        if opt == "zscore":
                            b = round(a, 2)
                        else:
                            b = round(max(tmp), 2)
                    if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                        pass
                    else:
                        file5.writelines(gene1+"\t"+gene2 +
                                         "\t"+str(abs(b))+"\n")
                        myset.add(gene1)
                        myset.add(gene2)
                        H_gene.append(gene1+":"+gene2)
                else:
                    logger.info('Missing LD ....')
            elif gene1 in self.affy_map1 and gene1 not in H_gene and gene1 != "---":
                SNP = self.affy_map1[gene1]
                temp = []
                for subset in it.product(SNP, SNP):
                    SNP_list = list(subset)
                    if SNP_list[0] in tagget_dict and SNP_list[1] in tagget_dict:
                        if SNP_list[0] != SNP_list[1]:
                            if SNP_list[0]+":"+SNP_list[1] in keep_snp_ld:
                                tmp.append(
                                    keep_snp_ld[SNP_list[0]+":"+SNP_list[1]])
                            elif SNP_list[1]+":"+SNP_list[0] in keep_snp_ld:
                                tmp.append(
                                    keep_snp_ld[SNP_list[1]+":"+SNP_list[0]])
                            else:
                                LD, cor = self.calc_rsq(
                                    tagget_dict[SNP_list[0]], tagget_dict[SNP_list[1]], SNP_list[0], SNP_list[1])
                                if LD > 1.0:
                                    temp.append(0.95)
                                    keep_snp_ld[SNP_list[0] +
                                                ":"+SNP_list[1]] = 0.95
                                else:
                                    temp.append(LD)
                                    keep_snp_ld[SNP_list[0] +
                                                ":"+SNP_list[1]] = LD
                        else:
                            temp.append(0.95)
                    elif gene1 in self.diseaseGenes:  # or gene2 in self.diseaseGenes:
                        temp.append(0.01)
                if len(temp) != 0:
                    if opt == "zscore":
                        a = np.mean(temp)
                        b = abs(a/sqrt(2))
                    else:
                        b = max(tmp)
                    try:
                        if abs(b) > 1:
                            LD1 = 0.9
                            if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                pass
                            else:
                                file5.writelines(
                                    str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                                myset.add(gene1)
                                H_gene.append(gene1+":"+gene2)
                                myset.add(gene2)
                        else:
                            if b > 0:
                                LD1 = round(abs(1-abs(b)), 7)
                                if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                    pass
                                else:
                                    file5.writelines(
                                        str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                                    H_gene.append(gene1+":"+gene2)
                                    myset.add(gene1)
                                    myset.add(gene2)
                    except (RuntimeError, TypeError, NameError):
                        pass
                else:
                    if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                        pass
                    else:
                        file5.writelines(
                            str(gene1)+"\t"+str(gene2)+"\t"+str(0.001)+"\n")
                        H_gene.append(gene1+":"+gene2)
                        myset.add(gene1)
                        myset.add(gene2)
            elif gene2 in self.affy_map1 and gene2 not in H_gene:
                SNP = self.affy_map1[gene2]
                temp = []
                try:
                    for subset in it.product(SNP, SNP):
                        SNP_list = list(subset)
                        if SNP_list[0] in tagget_dict and SNP_list[1] in tagget_dict:
                            if SNP_list[0] != SNP_list[1]:
                                if SNP_list[0]+":"+SNP_list[1] in keep_snp_ld:
                                    tmp.append(
                                        keep_snp_ld[SNP_list[0]+":"+SNP_list[1]])
                                elif SNP_list[1]+":"+SNP_list[0] in keep_snp_ld:
                                    tmp.append(
                                        keep_snp_ld[SNP_list[1]+":"+SNP_list[0]])
                                else:
                                    LD, cor = self.calc_rsq(
                                        tagget_dict[SNP_list[0]], tagget_dict[SNP_list[1]], SNP_list[0], SNP_list[1])
                                    if LD > 1.0:
                                        temp.append(0.95)
                                        keep_snp_ld[SNP_list[0] +
                                                    ":"+SNP_list[1]] = 0.95
                                    else:
                                        temp.append(LD)
                                        keep_snp_ld[SNP_list[0] +
                                                    ":"+SNP_list[1]] = LD
                            else:
                                temp.append(0.95)
                        else:
                            if gene2 in self.diseaseGenes:
                                temp.append(0.01)
                    if len(tmp) != 0:
                        if opt == "zscore":
                            a = np.mean(temp)
                            b = abs(a/sqrt(2))
                        else:
                            b = max(tmp)
                        if abs(b) > 1:
                            LD1 = 0.9
                            if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                pass
                            else:
                                file5.writelines(
                                    str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                                myset.add(gene1)
                                myset.add(gene2)
                                H_gene.append(gene1+":"+gene2)
                        else:
                            LD1 = round(abs(1-abs(b)), 7)
                            if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                pass
                            else:
                                file5.writelines(
                                    str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                                myset.add(gene1)
                                myset.add(gene2)
                                H_gene.append(gene1+":"+gene2)
                    else:
                        if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                            pass
                        else:
                            file5.writelines(
                                str(gene1)+"\t"+str(gene2)+"\t"+str(0.001)+"\n")
                            myset.add(gene1)
                            myset.add(gene2)
                            H_gene.append(gene1+":"+gene2)
                except (RuntimeError, TypeError, NameError):
                    pass
            else:
                if gene2 in self.diseaseGenes and gene1 in self.diseaseGenes:
                    if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                        pass
                    else:
                        file5.writelines(
                            str(gene1)+"\t"+str(gene2)+"\t"+str(abs(0.01))+"\n")
                        myset.add(gene1)
                        myset.add(gene2)
                        H_gene.append(gene1+":"+gene2)
        logger.info('Genes included: %s' % str(len(list(myset))))
        file5.close()
        rows_LD.clear()
        tagget_dict.clear()
        tmp = []

    def closestLDGene(self, networkFile, tagget_dict):
        '''
        Computes gene-gene-LD using the LD between the 2 closest snps in 2 given genes
        networkFile: a string, the PPI network file
        tagget_dict: a dictionary, the tagged population genotype file
        '''
        logger.info('Start creating the network Gene-Gene-LD ...')
        H_gene = []
        myset = set()
        keep_snp_ld = {}
        pkl_file = open(self.outfolder+'affy_map3.pkl', 'rb')
        self.affy_map3 = pickle.load(pkl_file)

        file5 = open(self.outfolder+"GENE_RESULT/gene2geneLD.net", "wt")
        for line in fileinput.input(networkFile):
            data2 = line.split()
            gene1 = data2[0]
            gene2 = data2[1]
            if gene1 in self.affy_map3 and gene2 in self.affy_map3:
                try:
                    snp1 = self.affy_map3[gene1][0]
                    snp2 = self.affy_map3[gene2][0]
                    if snp1 in tagget_dict and snp2 in tagget_dict:
                        if snp1 != snp2:
                            if snp1+":"+snp2 in keep_snp_ld:
                                LD1 = keep_snp_ld[snp1+":"+snp2]
                            elif snp2+":"+snp1 in keep_snp_ld:
                                LD1 = keep_snp_ld[snp2+":"+snp1]
                            else:
                                LD, cor = self.calc_rsq(
                                    tagget_dict[snp1], tagget_dict[snp2], snp1, snp2)
                                if LD > 1.0:
                                    LD1 = 0.95
                                    keep_snp_ld[snp1+":"+snp2] = 0.95
                                else:
                                    LD1 = LD
                                    keep_snp_ld[snp1+":"+snp2] = LD
                            if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                pass
                            else:
                                file5.writelines(
                                    str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                                myset.add(gene1)
                                myset.add(gene2)
                                H_gene.append(gene1+":"+gene2)
                        else:
                            if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                pass
                            else:
                                file5.writelines(
                                    str(gene1)+"\t"+str(gene2)+"\t"+str(0.95)+"\n")
                                myset.add(gene1)
                                myset.add(gene2)
                                H_gene.append(gene1+":"+gene2)
                    else:
                        if gene1 in self.diseaseGenes or gene2 in self.diseaseGenes:

                            if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                                pass
                            else:
                                file5.writelines(
                                    str(gene1)+"\t"+str(gene2)+"\t"+str(0.01)+"\n")
                                myset.add(gene1)
                                myset.add(gene2)
                                H_gene.append(gene1+":"+gene2)
                except (RuntimeError, TypeError, NameError):
                    pass
            elif gene1 in self.affy_map1 and gene1 not in H_gene:
                SNP = self.affy_map1[gene1]
                temp = []
                for subset in it.product(SNP, SNP):
                    SNP_list = list(subset)
                    if SNP_list[0] in tagget_dict and SNP_list[1] in tagget_dict:
                        if SNP_list[0] != SNP_list[1]:
                            if SNP_list[0]+":"+SNP_list[1] in keep_snp_ld:
                                temp.append(
                                    keep_snp_ld[SNP_list[0]+":"+SNP_list[1]])
                            elif SNP_list[1]+":"+SNP_list[0] in keep_snp_ld:
                                temp.append(
                                    keep_snp_ld[SNP_list[1]+":"+SNP_list[0]])
                            else:
                                LD, cor = self.calc_rsq(
                                    tagget_dict[SNP_list[0]], tagget_dict[SNP_list[1]], SNP_list[0], SNP_list[1])
                                if LD > 1.0:
                                    temp.append(0.95)
                                    keep_snp_ld[SNP_list[0] +
                                                ":"+SNP_list[1]] = 0.95
                                else:
                                    temp.append(LD)
                                    keep_snp_ld[SNP_list[0] +
                                                ":"+SNP_list[1]] = LD
                        else:
                            temp.append(0.95)
                    else:
                        if gene1 in self.diseaseGenes:
                            temp.append(0.01)
                a = np.mean(temp)
                b = abs(a/sqrt(2))
                try:
                    if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                        pass
                    else:
                        if abs(b) > 1:
                            LD1 = 0.9
                            file5.writelines(
                                str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                            myset.add(gene1)
                            myset.add(gene2)
                            H_gene.append(gene1+":"+gene2)
                        else:
                            LD1 = round(abs(1-abs(b)), 7)
                            file5.writelines(
                                str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                            H_gene.append(gene1+":"+gene2)
                            myset.add(gene1)
                            myset.add(gene2)

                except (RuntimeError, TypeError, NameError):
                    pass
            elif gene2 in self.affy_map1 and gene2 not in H_gene:
                H_gene.append(gene2)
                SNP = self.affy_map1[gene2]
                temp = []
                try:
                    for subset in it.product(SNP, SNP):
                        SNP_list = list(subset)
                        if SNP_list[0] in tagget_dict and SNP_list[1] in tagget_dict:
                            if SNP_list[0] != SNP_list[1]:
                                if SNP_list[0]+":"+SNP_list[1] in keep_snp_ld:
                                    temp.append(
                                        keep_snp_ld[SNP_list[0]+":"+SNP_list[1]])
                                elif SNP_list[1]+":"+SNP_list[0] in keep_snp_ld:
                                    temp.append(
                                        keep_snp_ld[SNP_list[1]+":"+SNP_list[0]])
                                else:
                                    LD, cor = self.calc_rsq(
                                        tagget_dict[SNP_list[0]], tagget_dict[SNP_list[1]], SNP_list[0], SNP_list[1])
                                    if LD > 1.0:
                                        temp.append(0.95)
                                        keep_snp_ld[SNP_list[0] +
                                                    ":"+SNP_list[1]] = 0.95
                                    else:
                                        temp.append(LD)
                                        keep_snp_ld[SNP_list[0] +
                                                    ":"+SNP_list[1]] = LD
                            else:
                                temp.append(0.95)
                        else:
                            if gene2 in self.diseaseGenes and gene2 not in self.diseaseGenes:
                                temp.append(0.01)
                    a = np.mean(temp)
                    b = abs(a/sqrt(2))
                    if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                        pass
                    else:
                        if abs(b) > 1:
                            LD1 = 0.9
                            file5.writelines(
                                str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                            H_gene.append(gene1+":"+gene2)
                            myset.add(gene1)
                            myset.add(gene2)
                        else:
                            LD1 = round(abs(1-abs(b)), 7)
                            file5.writelines(
                                str(gene1)+"\t"+str(gene2)+"\t"+str(abs(LD1))+"\n")
                            H_gene.append(gene1+":"+gene2)
                            myset.add(gene1)
                            myset.add(gene2)
                except (RuntimeError, TypeError, NameError):
                    pass
            else:
                if gene2 in self.diseaseGenes and gene1 in self.diseaseGenes:
                    if gene1+":"+gene2 in H_gene or gene2+":"+gene1 in H_gene:
                        pass
                    else:
                        file5.writelines(
                            str(gene1)+"\t"+str(gene2)+"\t"+str(abs(0.01))+"\n")
                        myset.add(gene1)
                        myset.add(gene2)
                        H_gene.append(gene1+":"+gene2)

        logger.info('Genes included: %s' % str(len(list(myset))))
        file5.close()
        pkl_file.close()

    def calc_rsq(self, genotypes1, genotypes2, rsid1, rsid2):
        '''
        Computes the correlation between two genotype data.
        genotypes1,genotypes2: lists, lists of gentype
        rsid1,rsid2: strings, SNPs to cpmute the LD
        '''
        gen1, gen2 = self.remove_missing(genotypes1, genotypes2, rsid1, rsid2)
        snp = []
        if len(gen1) == 0 and len(gen2) == 0:
            snp.append(rsid2)
            snp.append(0.0)
            logger.info('Too much missing data to compute LD')
            return 0.001, 0.001  # too much missing data to compute r
        elif len(gen1) != 0 and len(gen2) != 0:
            if len(gen1) == len(gen2):
                corr = self.get_r(gen1, gen2)  # get_weighted_r(gen1,gen2)
                if corr in ["NaN", "NA", "NAN"]:
                    corr = 0.001
                y = 0.5*log((1+corr)/(1-corr))
                l = sqrt(len(gen1)-3)*y
                if l == 0.0:
                    l = 0.001
                if l >= 1.0:
                    l = 0.99
                return abs(l), abs(corr)
            else:
                logger.info('Genotypes do not match (exiting)')
                raise SystemExit

    def get_r(self, x, y):
        """get_r : inputs two vectors of the same size and return the correlation between these two vectors of values. """
        if len(x) != len(y):
            logger.info('Error: lengths of vectors do not match in get_corr')
        n = len(x)
        xSum = 0.0
        xxSum = 0.0
        ySum = 0.0
        yySum = 0.0
        xySum = 0.0
        for j in range(n):
            xVal = x[j]
            yVal = y[j]
            xSum += xVal
            xxSum += xVal * xVal
            ySum += yVal
            yySum += yVal * yVal
            xySum += xVal * yVal
            cov = xySum - (xSum * ySum) / float(n)
            xVar = xxSum - (xSum * xSum) / float(n)
            yVar = yySum - (ySum * ySum) / float(n)
            den = sqrt(xVar*yVar)
        if den > 0:
            corr = cov/float(den)
            if abs(corr) >= 1.0:
                return 0.95
            elif corr == 0.0:
                return 0.01
            else:
                return abs(corr)
        else:
            return 0.01

    def remove_missing(self, genotypes1, genotypes2, rsid1, rsid2):
        '''
        Assessing missing genotypes
        '''
        geno = {"0": 0, "1": 1, "2": 2, "?": 9, "9": 9, "-": 9, "NA": 9}
        if len(genotypes1) != len(genotypes2):
            logger.info('Genotypes should be of the same length (exiting)')
            self.terminate()
            raise SystemExit
        else:
            gen1 = [geno[i] for i in genotypes1]
            gen2 = [geno[i] for i in genotypes2]
        return gen1, gen2


class ancGraph():
    '''
    Read the LD-weighted network, computes centraly measures, break down the PPI network to generate sub-networks 
    '''

    def __init__(self):
        return None

    def readGraph(self):
        '''
        Reading the LD-weighted network from the file gene2geneLD.net, generated in previous steps
        '''
        self.G = Graph()
        fp = open(self.outfolder+'GENE_RESULT/gene2geneLD.net')
        for line in fp:
            ligne = line.strip()
            if not ligne or ligne.startswith('#'):
                continue
            ligne = ligne.split()
            if ligne[-1] in ["NaN", "NAN", "nan"]:
                pass
            else:
                if float(ligne[-1]) > 1.0:
                    self.G.add_edge(ligne[0], ligne[1], weight=0.97)
                elif abs(float(ligne[-1])) < 0.05:
                    self.G.add_edge(ligne[0], ligne[1], weight=0.05)
                else:
                    self.G.add_edge(ligne[0], ligne[1],
                                    weight=abs(float(ligne[-1])))
        fp.close()
        return self.G

    def nodekLink(self):
        '''
        This function takes a network and returns protein degree distribution
        '''
        self.Degree = {}

        for node in self.G:
            self.Degree[node] = self.G.degree(node)
        NumberVertices_for_Nodes = self.Degree.values()
        Kvertices = list(set(sorted(NumberVertices_for_Nodes)))
        CountNodes = [NumberVertices_for_Nodes.count(n) for n in Kvertices]
        Proportions = [float(CountNodes[i])/sum(CountNodes) for i in xrange(len(CountNodes))]
        self.Gamma = leastsq(self.low_Distance, 0,(np.array(Kvertices), np.array(Proportions)))
        x = np.linspace(1, Kvertices[-1], 1000)
        if HAS_PLT:
            try:
                plt.figure(1, dpi=150)
                plt.plot(Kvertices, Proportions, 'ro', x, cp.power(
                x, self.Gamma[0]), 'k--', linewidth=2)
                plt.legend(("Connectivity-Distr.: "+r'$\mathcal{P}\left(k\right)$',
                          "Power-Law: "+r'$\mathcal{P}\left(k\right)\simeq k^{%.2f}$' % (self.Gamma[0])))
                plt.ylim(-0.001, max(Proportions)+0.001)
                plt.xlim(0.0, Kvertices[-1])
                plt.grid(True)
                plt.xlabel("Detected Protein Degree: "+r'$k$')
                plt.ylabel("Connections Frequency: " +
                         r'$\mathcal{P}\left(k\right)$')
                plt.savefig(self.outfolder+'NETWORK_RESULT/' +
                          'PowerLawData.png', dpi=150)
            except (RuntimeError, TypeError, NameError, IndexError):
                sys.stderr.write('Error => in protein degree Plotting ....!\n')
            finally:
                pass
        else:
            p = "plt"
            print"\nSkipping plotting due to missing %s packages..." % p

    def plotPathDistr(self):
        '''
        Compute shortest paths between all nodes in a weighted graph.
        This function takes a network path length and returns path distribution distribution and
        mean shortest paths needed for other computational purposes
        '''
        paths = {}
        LPath = set()

        for n in self.G:
            Length = single_source_dijkstra_path(self.G, n)
            for b in Length:
                if n == b:
                    continue
                LPath.add(len(Length[b])-1)
        self.ShortPathMean = np.mean(list(LPath))
        LPath = list(LPath)
        if HAS_PLT:
            try:
                plt.figure(2, dpi=150)
                n, bins, patches = plt.hist(LPath, normed=1, facecolor='0.8', alpha=1.0)
                t = np.linspace(0, max(LPath))
                y = np.exp(-(t-np.mean(LPath))**2/(2*np.std(LPath))) / \
                (np.std(LPath)*np.sqrt(2*np.pi))
                plt.plot(t, y, 'r--', linewidth=2)
                plt.grid(True)
                plt.xlabel("Path-Length: "+r'$\ell$')
                plt.ylabel("Frequency: "+r'$\mathcal{P}\left(\ell\right)$')
                plt.savefig(self.outfolder+"NETWORK_RESULT/" +
                        'PathDistr.png', dpi=150)
            except (RuntimeError, TypeError, NameError, IndexError):
                sys.stderr.write('Error => in Plotting path distribution and mean shortest paths ....!\n')
            finally:
                pass   
        else:
            p = "matlibplot"
            print"\nSkipping plotting due to missing %s packages..." % p

    def findallHubs(self):
        '''
        This returns all hubs of a given network and all connected components
        '''
        Hubs = []
        C = connected_component_subgraphs(self.G)
        for cliques in C:
            if cliques.order() <= 2:
                continue
            for gene in cliques:
                Ctemp = cliques.copy()
                Ctemp.remove_node(gene)
                if number_connected_components(Ctemp) > 1:
                    Hubs.append(gene)  # Gene under consideration is hub
                del Ctemp
        return Hubs

    def low_Distance(self, parameters, x, values):
        '''
        This function receives one parameter of power-low gamma model into 
        parameters, an array with the x and an array with the corresponding 
        values for each of the x. 
        our model is x**(-gamma)
        '''
        gamma = parameters
        errors = values - x**gamma
        return errors

    def subgraphFinding(self):
        '''
        This function computes the centraly measures, break down the PPI network to generate sub-networks.
        '''
        logger.info('Computing protein degree distribution, it may take time ...')
        logger.info('1. Now plotting Power-Law ...')
        self.nodekLink()
        logger.info('2. Computing length distribution, it may take time ...')
        self.plotPathDistr()
        logger.info("Shortest path mean "+str(self.ShortPathMean))
        logger.info('3. Searching for network structure hubs and it may take time ...')
        Hubs = self.findallHubs()
        logger.info('4. Computing Node betweenness scores...')
        self.Betw = betweenness_centrality(self.G)
        output5 = open(self.outfolder+'Betw.pkl', 'wb')
        pickle.dump(self.Betw, output5)
        logger.info('5. Computing Node closeness scores')
        self.Clos = closeness_centrality(self.G)
        output5 = open(self.outfolder+'Clos.pkl', 'wb')
        pickle.dump(self.Clos, output5)
        logger.info('6. Computing Node eigenvector scores')
        EigenScoreWorks = 0
        A = adj_matrix(self.G)
        try:
            a, b, c = svd(A)  # We will need either a[:,0] or c[0]
            if any(c[0] < 0.0):
                vecmax = -c[0]
            else:
                vecmax = c[0]
            EigenScoreWorks = 1
        except:
            logger.info('Singularity problem occurs while computing Eigen-vector scores ...')
        self.BetOf = self.ShortPathMean * self.G.order()
        if EigenScoreWorks:
            self.EigOf = np.mean(vecmax)
        self.ClosOf = 1.0/self.ShortPathMean
        self.DegOf = 2.0*self.G.size()/self.G.order()
        Node = self.G.nodes()
        self.BetweenNodes = set(
            [node for node in self.G if self.Betw[node] >= self.BetOf])
        self.CloseNodes = set(
            [node for node in self.G if self.Clos[node] >= self.ClosOf])
        if EigenScoreWorks:
            self.EigenNodes = set([Node[i] for i in xrange(len(Node)) if vecmax[i] >= self.EigOf])
        else:
            self.EigenNodes = set([node for node in self.G if len(self.G[node]) >= self.DegOf])
        logger.info('7. Running the last step: searching for subgraphs ...')
        if len(Hubs) > 1:
            if len(self.BetweenNodes & self.CloseNodes & self.EigenNodes) > 1:
                self.CenterGenes = set(Hubs) & self.BetweenNodes & self.CloseNodes & self.EigenNodes
            elif len(self.BetweenNodes & self.CloseNodes) > 1 and len(self.EigenNodes) == 0:
                self.CenterGenes = set(Hubs) & self.BetweenNodes & self.CloseNodes
            elif len(self.BetweenNodes & self.EigenNodes) > 1 and len(self.CloseNodes) == 0:
                self.CenterGenes = set(Hubs) & self.BetweenNodes & self.EigenNodes
            elif len(self.EigenNodes & self.CloseNodes) > 1 and len(self.BetweenNodes) == 0:
                self.CenterGenes = set(Hubs) & self.CloseNodes & self.EigenNodes
            else:
                self.CenterGenes = set(Hubs)
        elif len(Hubs) == 1:
            logger.info('\nFatal Error: No CenterGene found!!!\nancMETA Could not continue\n')
            self.terminate()
            sys.exit(1)
        else:
            if len(self.BetweenNodes & self.CloseNodes & self.EigenNodes) > 1:
                self.CenterGenes = self.BetweenNodes & self.CloseNodes & self.EigenNodes
            elif len(self.BetweenNodes & self.CloseNodes) > 1 and len(self.EigenNodes) == 0:
                self.CenterGenes = self.BetweenNodes & self.CloseNodes
            elif len(self.BetweenNodes & self.EigenNodes) > 1 and len(self.CloseNodes) == 0:
                self.CenterGenes = self.BetweenNodes & self.EigenNodes
            elif len(self.EigenNodes & self.CloseNodes) > 1 and len(self.BetweenNodes) == 0:
                self.CenterGenes = self.CloseNodes & self.EigenNodes
        # Clear all Centralities Measures
        self.BetweenNodes.clear()
        self.CloseNodes.clear()
        self.EigenNodes.clear()
        if self.Path:
            if int(self.ShortPathMean) > 5:
                n = int(round(float(self.ShortPathMean)))
            else:
                n = int(int(self.ShortPathMean))
        else:
            n = 1

        self.Clouds = {}
        self.Clouds1 = {}
        self.Commit = {}
        for central in self.CenterGenes:
            self.Clouds[central] = set([central])
            self.Clouds1[central] = set([central])
            Temp = set(self.G.neighbors(central))
            Temp1 = set(self.G.neighbors(central))
            i = 0
            while i < int(round(float(self.ShortPathMean), 1)):
                Current1 = Temp1-self.Clouds1[central]
                Temp1.clear()
                for gene in Current1:
                    self.Clouds1[central].add(gene)
                    Temp1 |= set(self.G.neighbors(gene))
                i += 1
            i = 0
            while i < n:
                Current = Temp-self.Clouds[central]
                Temp.clear()
                for gene in Current:
                    self.Clouds[central].add(gene)
                    Temp |= set(self.G.neighbors(gene))
                i += 1
            for gene in self.diseaseGenes:
                if gene in self.Clouds1[central] and gene not in self.Clouds[central]:
                    #T = set(self.G.neighbors(gene));T.add(gene)
                    T = set(shortest_path(self.G, gene, central))
                    if gene in self.Commit:
                        S = self.Commit[gene]
                        S.append([central]+list(T))
                    else:
                        self.Commit[gene] = [[central]+list(T)]
                    #self.Clouds[central] = self.Clouds[central] | T
        C = self.Clouds.keys()
        IDX = self.Clouds[C[0]]
        for gene in self.diseaseGenes:
            if gene in self.Commit:
                if len(self.Commit[gene]) == 1:
                    self.Clouds[self.Commit[gene][0][0]] = self.Clouds[self.Commit[gene][0][0]] | set(
                        self.Commit[gene][0][1:])
                else:
                    tmp = []
                    for de in self.Commit[gene]:
                        tmp.append(len(de))
                    n = min(tmp)
                    self.Clouds[self.Commit[gene][n][0]] = self.Clouds[self.Commit[gene][n][0]] | set(
                        self.Commit[gene][n][1:])
        for de in C[1:]:
            S = self.Clouds[de].copy()
            self.Clouds[de] = self.Clouds[de] - set(self.Clouds[de] & IDX)
            IDX = S.copy()
        output = open(self.outfolder+'CloudsStep.pkl', 'wb')
        pickle.dump(self.Clouds, output)
        self.G.clear()
        self.Clouds.clear()
        self.Clouds1.clear()
        self.Commit.clear()


class ancScoring():
    '''
    This module computes the score of each sub-netowk to detect the ones enriched in p-value and particular ancestries
    '''

    def __init__(self):
        return None

    def Scoring(self):
        '''
        Scoring sub-networks
        '''

        gene2LD = {}

        for line in fileinput.input(self.outfolder+"GENE_RESULT/gene2geneLD.net"):
            data = line.split()
            gene2LD[data[0]+":"+data[1]] = float(data[2])
        logger.info('Reading saved subnetworks ...')
        pkl_file = open(self.outfolder+"CloudsStep.pkl", 'rb')
        Gene2mod = pickle.load(pkl_file)
        if self.fast:
            gene_p = open(self.outfolder+"gene2Pv.pkl", 'rb')
            self.gene2pval = pickle.load(gene_p)
            gene_f = open(self.outfolder+"gene2Eff.pkl", "rb")
            gene_e = open(self.outfolder+"gene2Err.pkl", "rb")
            self.gene2Eff = pickle.load(gene_f)
            self.gene2Err = pickle.load(gene_e)
        else:
            gene_snp = open(self.outfolder+"gene_map.pkl", 'rb')
            self.gene_map = pickle.load(gene_snp)
            gene_snp1 = open(self.outfolder+"gene_map1.pkl", 'rb')
            self.gene_map1 = pickle.load(gene_snp1)
            gene_snp2 = open(self.outfolder+"gene_map2.pkl", 'rb')
            self.gene_map2 = pickle.load(gene_snp2)

        logger.info('Statistic model for scoring subnetworks and writing the final result...')
        # Cleaning existing file as later the script open the file in append mode
        fix1 = open(self.outfolder+"NETWORK_RESULT/"+"Meta_Analysis.txt", "wt")
        fix2 = open(self.outfolder+"NETWORK_RESULT/" +"Meta_Input.txt", "wt")  # Cleaning existing file
        suM, suM1 = self.subnetwork_scoring(Gene2mod)
        self.subnetwork_anc(suM, gene2LD)
        self.central_net_disease()

        self.write_move()
        logger.info('Plotting top Sub-network Meta-Analysis ...')
        if HAS_NP:
            try:
                self.mmplot("sub", self.outfolder+"NETWORK_RESULT/")
            except (RuntimeError, TypeError, NameError, IndexError):
                sys.stderr.write('Error => in Forest Plotting ....!\n')
            finally:
                pass

    def write_move(self):
        fi1 = open(self.outfolder+"NETWORK_RESULT/" +"Meta_subnet_tmp.txt", "wt")
        fi2 = open(self.outfolder+"NETWORK_RESULT/"+"Meta_Input_tmp.txt", "wt")
        rows = {}
        subnet_input = open(self.outfolder+"NETWORK_RESULT/"+"Meta_Analysis.txt")
        subnet = subnet_input.readlines()
        fi1.write(subnet[0])
        NUM = 16 + 2*len(self.study_names)

        rows1 = {}
        subnet_input1 = open(self.outfolder+"NETWORK_RESULT/"+"Meta_Input.txt")
        subnet1 = subnet_input1.readlines()
        fi2.write(subnet1[0])
        for l in subnet1[1:]:
            data = l.split()
            rows1[data[0]] = [data[0]]+[float(j) for j in data[1:]]

        for l in subnet[1:]:
            data = l.split()
            if len(data) < NUM:
                pass
            else:
                if int(self.inxed_sort) in [16, 17, 18]:
                    rows[data[0]] = [float(data[self.inxed_sort])]+data
                else:
                    rows[data[0]] = [float(data[16])]+data
        D_input = rows.values()
        D_input.sort()
        if int(self.inxed_sort) in [16, 17, 18]:
            D_input = sorted(sorted(D_input, key=lambda x: x[self.inxed_sort]),
                             key=lambda x: x[self.inxed_sort+len(self.study_names)], reverse=True)
        else:
            D_input = sorted(
                sorted(D_input, key=lambda x: x[16]), key=lambda x: x[19], reverse=True)
        for d in D_input:
            fi1.write("\t".join([str(h) for h in d[1:]])+"\n")
            fi2.write("\t".join([str(h) for h in rows1[d[1:][0]]])+"\n")
        fi1.close();fi2.close()

        os.system("mv"+" "+self.outfolder+"NETWORK_RESULT/"+"Meta_Input_tmp.txt" +" "+self.outfolder+"NETWORK_RESULT/"+"Meta_Input.txt")
        os.system("mv"+" "+self.outfolder+"NETWORK_RESULT/"+"Meta_subnet_tmp.txt" +" "+self.outfolder+"NETWORK_RESULT/"+"Meta_Analysis.txt")
        D_input = [];rows.clear();rows1.clear()

    def subnetwork_scoring(self, Gene2module):
        '''
        Computes the score of each module and compute the adjusted pvalues, Qvalues and FDR
        '''
        score_mod = []
        meta_mod = []
        if self.fast:
            for mod in Gene2module:
                tmps = set()
                tmps1 = set()
                self.sub_eff = {}
                self.sub_err = {}
                if mod in self.gene2pval:
                    tmps.add(self.gene2pval[mod][0])
                    tmps1.add(self.gene2pval[mod][1])  # self.gene2pval[mod]
                    if mod in self.gene2Eff:
                        #if len(Gene2module[mod]) > self.nb_net:
                        self.sub_eff[mod] = [self.gene2Eff[mod]]
                        self.sub_err[mod] = [self.gene2Err[mod]]
                        score_m = self.score_subnetworks(mod, Gene2module[mod], tmps, tmps1, self.sub_eff, self.sub_err)
                        score_mod.append(score_m)
                        #else:
                        #   print"\nSkipping subnetwork of hub %s has just %d proteines ..." % (mod, len(Gene2module[mod]))
                elif mod in self.diseaseGenes and mod not in self.gene2pval:
                    T = set(Gene2module[mod]) & set(self.gene2pval)
                    if len(T) != 0:
                        T = list(T)
                        if T[0] in self.gene2Eff:
                           self.sub_eff[T[0]] = [self.gene2Eff[T[0]]]
                           self.sub_err[T[0]] = [self.gene2Err[T[0]]]
                           score_m = self.score_subnetworks(T[0], T, tmps, tmps1, self.sub_eff, self.sub_err)
                           score_mod.append(score_m)
                        else:
                           pass
                    #else:
                    #    print"\nSkipping subnetwork of hub %s has just %d proteines ..." % (mod, len(T))
                else:
                    T = set(Gene2module[mod]) & set(self.gene2pval)
                    if len(T) > self.nb_net:
                        T = list(T)
                        if T[0] in self.gene2Eff:
                            self.sub_eff[T[0]] = [self.gene2Eff[T[0]]]
                            self.sub_err[T[0]] = [self.gene2Err[T[0]]]
                            score_m = self.score_subnetworks(T[0], T, tmps, tmps1, self.sub_eff, self.sub_err)
                            score_mod.append(score_m)
                        else:
                            pass
                    else:
                        print"\nSkipping subnetwork of hub %s has just %d proteines ..." % (mod, len(T))
        else:
            
            for mod in Gene2module:
                tmp = []
                tmp1 = set()
                self.sub_eff = {}
                self.sub_err = {}
                if mod in self.gene_map:
                    # tmps.add(self.gene2pval[mod][0]); tmps1.add(self.gene2pval[mod][1]) #self.gene2pval[mod]
                    tmp = [float(i) for i in list(self.gene_map[mod])]
                    if mod in self.gene_map1 and mod in self.gene_map2:
                        self.sub_eff[mod] = [gen for gen in list(
                            self.gene_map1[mod])]  # for h in gen]
                        self.sub_err[mod] = [fd for fd in list(
                            self.gene_map2[mod])]  # for y in fd]
                        if len(Gene2module[mod]) > self.nb_net:
                            score_m = self.score_subnetworks(mod, Gene2module[mod], tmp, tmp1, self.sub_eff, self.sub_err)
                            score_mod.append(score_m)
                        else:
                            print"\nSkipping subnetwork of hub %s has just %d proteines ..." % (mod, len(Gene2module[mod]))
                elif mod in self.diseaseGenes and mod not in self.gene_map:
                    T = set(Gene2module[mod]) & set(self.gene_map)
                    T = list(T)
                    if len(T) != 0:  # and T[0] in self.sub_eff:

                        if T[0] in self.gene_map1 and mod in self.gene_map2:
                            self.sub_eff[T[0]] = [gen for gen in list(self.gene_map1[T[0]])]  # for h in gen]
                            self.sub_err[T[0]] = [grn1 for gen1 in list(self.gene_map2[T[0]])]  # for k in gen1]
                            score_m = self.score_subnetworks(T[0], T, tmp, tmp1, self.sub_eff, self.sub_err)
                            score_mod.append(score_m)
                    else:
                        print"\nSkipping subnetwork of hub %s has just %d proteines ..." % (mod, len(T))
                else:
                    T = set(Gene2module[mod]) & set(self.gene_map)
                    T = list(T)
                    if len(T) > self.nb_net:
                        if T[0] in self.gene_map1 and mod in self.gene_map2:
                            self.sub_eff[T[0]] = [gen for gen in list(self.gene_map1[T[0]])]  # for h in gen ]
                            self.sub_err[T[0]] = [gen2 for gen2 in list(self.gene_map2[T[0]])]  # for l in gen2 ]
                            score_m = self.score_subnetworks(T[0], T, tmp, tmp1, self.sub_eff, self.sub_err)
                            score_mod.append(score_m)
                    else:
                        print"\nSkipping subnetwork of hub %s has just %d proteines ..." % (mod, len(T))
        self.gene_map.clear()
        self.gene_map1.clear()
        self.gene_map2.clear(), self.sub_eff.clear()
        self.sub_err.clear()

        if len(score_mod) == 0:
            pass
        else:
            NET2Pv_ = sorted(score_mod, key=lambda a_key: float(a_key[1]), reverse=False)
            fix = open(self.outfolder+"NETWORK_RESULT/" +"subnetwork.all_score", "wt")
            R = [float(p[1]) for p in NET2Pv_]
            pval_norm = self.adjustPValues(R, self.adjustP, None)
            gene2Qval = self.computeQValues(R, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)
            R = []
            R = [float(p[2]) for p in NET2Pv_]
            pval_norm1 = self.adjustPValues(R, self.adjustP, None)
            gene2Qval1 = self.computeQValues(R, None, self.pi0_method, self.fdr_level, self.robust, 3, False, None)
            GC = np.mean(self.genomic_control(R))
            logger.info('The Genomic Control at Subnetwork level is: %s' % str(GC))
            Label = ["Liptak", "Q_Liptak", "Fisher", "Q_Fisher"]
            # Liptak  Q_Liptak        Fisher  Q_Fisher
            fix.writelines("Liptak"+"\t"+"Q_Liptak"+"\t"+"Fisher"+"\t" +"Q_Fisher"+"\t"+"Subnetwork_Hub"+"\t"+"Lists.Genes"+"\n")
            for i in range(len(NET2Pv_)):
                NET2Pv_[i].append(pval_norm[i])
                NET2Pv_[i].append(gene2Qval[i])
                NET2Pv_[i].append(pval_norm1[i])
                NET2Pv_[i].append(gene2Qval1[i])
            rows = set()
            # Subnetwork and known pathway overlap association test
            logger.info('Association Statistical test between top Subnetworks and known biological Pathways...')
            #NET2Pv_ = sorted(score_mod, key= lambda a_key: float(a_key[1]), reverse=False)

            rows, D_rows = self.path_enrichment(NET2Pv_, Gene2module, Label)

            for des in NET2Pv_:
                if des[0] in Gene2module:
                    tmp = list(des[3:])+list([des[0]]) + list(Gene2module[des[0]])
                    tmps = list(Gene2module[des[0]])
                    rows |= set(tmps)
                    fix.write("\t".join([str(de) for de in list(tmp)])+"\n")
                else:
                    continue
            fix.close()

            return rows, D_rows

    def score_subnetworks(self, mod, Gene2mod, tmp, tmp1, sub_eff, sub_err):
        if self.fast:
            if mod in sub_eff:
                E = sub_eff[mod];S = sub_err[mod]
            for gene in Gene2mod:
                if gene in self.gene2pval:
                    tmp.add(float(self.gene2pval[gene][0]))
                    tmp1.add(float(self.gene2pval[gene][1]))
                if gene in self.gene2Eff and gene != mod:
                    #if len(E) != 0:
                    E.append(self.gene2Eff[gene])
                    S.append(self.gene2Err[gene])
                    # else:
            if 0.0 in tmp:
                tmp.remove(0.0)
            elif 1.0 in tmp:
                tmp.remove(1.0)
            if 0.0 in tmp1:
                tmp1.remove(0.0)
            elif 1.0 in tmp1:
                tmp1.remove(1.0)

            tmp = list(tmp)
            tmp.sort()
            tp = np.array(tmp)
            tmp1 = list(tmp1)
            tmp1.sort()
            tp1 = np.array(tmp1)

            if len(tp) != 0 and len(tp1) != 0:
                if len(tp) <= 20:
                    X = np.mean(tmp1)
                    W = 1.0/np.var(tmp1)
                    Z = (W*X)/np.sqrt(W)
                    p = self.zscore_to_pvalues(Z)
                    p1 = self.stouffer_liptak(tmp1)
                    results = [mod, float(p1), p]
                else:
                    p = self.stouffer_liptak(tp1)
                    Anto_cor = self.estimated_autocorrelation(tp1)
                    sigma = self.gen_sigma_matrix(tp1, Anto_cor)
                    Z_cor = self.z_score_combine(tp1, sigma)
                    results = [mod, float(p), float(Z_cor)]
            if len(sub_eff) == 0:

                return results
            else:
                self.sub_effect_size(sub_eff, sub_err)
                return results
        else:

            if mod in self.gene_map:
                # for h in gen]))
                self.sub_eff[mod] = [gen for gen in self.gene_map1[mod]]
                # for h in gen])) #####################################
                self.sub_err[mod] = [gen for gen in list(self.gene_map2[mod])]
                tmp = tmp + list(set([float(gen)
                                      for gen in list(self.gene_map[mod])]))
            else:
                pass
            for g in Gene2mod:
                if g in self.gene_map and g != mod:
                    tmp = tmp + list(set([float(gen)
                                          for gen in list(self.gene_map[g])]))
                if g in self.gene_map1 and self.gene_map2:
                    if g != mod:
                        # for h in gen])) #####################################
                        self.sub_eff[mod] = self.sub_eff[mod] + [gen for gen in list(self.gene_map1[g])]
                        # for h in gen])) #####################################
                        self.sub_err[mod] = self.sub_err[mod] + [gen for gen in list(self.gene_map2[g])]
            tmp = set(tmp)
            if 1.0 in tmp:
                tmp.remove(1.0)
            elif 0.0 in tmp:
                tmp.remove(0.0)
            tmp = list(tmp)
            tp = np.array(tmp)
            if len(tp) != 0:

                if len(tp) <= 20:
                    X = np.mean(tmp)
                    W = 1.0/np.var(tmp)
                    Z = (W*X)/np.sqrt(W)
                    p = self.zscore_to_pvalues(Z)
                    p1 = self.stouffer_liptak(tmp)
                    results = [mod, p1, p]
                else:
                    p = self.stouffer_liptak(tp)
                    Anto_cor = self.estimated_autocorrelation(tp)
                    sigma = self.gen_sigma_matrix(tp, Anto_cor)
                    Z_cor = self.z_score_combine(tp, sigma)
                    results = [mod, float(p), float(Z_cor)]

            if len(self.sub_eff) == 0:

                return results
            else:
                self.sub_effect_size(self.sub_eff, self.sub_err)
                return results

    def read_disease_genes(self):
        '''
        Read the known disease genes in pathway/disease_canidate_gene.txt and return a list of list of ['ID', 'disease name', [disease genes]]
        '''
        myList = []
        if os.path.exists(self.gene_disease):
            for line in fileinput.input(self.gene_disease):
                data = line.strip().split()
                if "ECR777" in data:
                    idx = data.index("ECR777")
                    ID = data[0]
                    disease = ' '.join(data[1:idx])
                    genes = data[idx+1:]
                    myList.append([ID, disease, list(set(genes))])
        return myList

    def userDiseaseGenes(self):
        cond = False
        for dis in self.read_disease_genes():
            if dis[0].lower() == self.disease.lower():
                cond = True
                break
        if cond == False:
            logger.info("The gene-disease file " +self.gene_disease + " is in a bad format !!")
            logger.info("Disease ID speficfied \"%s\" not found, ancMETA assumes NO disease!!" % self.disease)
            return [None, None, []]
        else:
            return dis

    def central_net_disease(self):
        gene_pvalues = {}
        gene_disease = []
        check = 0
        fin = open(self.outfolder+"NETWORK_RESULT/" +"central_subnetwork.txt", "wt")
        F = open(self.outfolder+"GENE_RESULT/gene2Pv.txt")
        G = F.readlines()
        for line in G[2:]:  # open(self.outfolder+"GENE_RESULT/gene2Pv.txt"):
            data = line.split()
            if float(data[1]) < 0.3:  # prioriotize based on the GWAS result
                gene_pvalues[data[0]] = float(data[1])
        #### List contain genes of interest, have to be included ##################
        if os.path.exists(self.gene_disease):
            gene_disease = set()
            for line in open(self.gene_disease):
                data = line.split()
                if "ECR777" not in data:
                    check = check + 1
                else:
                    K = data[0]
                    D = data[1:]
                    idx = D.index("ECR777")
                    if K == self.disease:  # or disease_name in D:
                        gene_disease = gene_disease | set(D[idx+1:])

            for line in open(self.gene_disease):
                data = line.split()
                if "ECR777" not in data:
                    check = check + 1
                else:
                    K = data[0]
                    D = data[1:]
                    idx = D.index("ECR777")
                    if len(self.disease_name) == 1:
                        if self.disease_name[0] in D:
                            gene_disease = gene_disease | set(D[idx+1:])
                    elif len(self.disease_name) >= 2:
                        if self.disease_name[0] in D and self.disease_name[1] in D:
                            gene_disease = gene_disease | set(D[idx+1:])
            if check == 0 or len(gene_disease) != 0:
                H = open(self.outfolder+"NETWORK_RESULT/network_LD.out")
                J = H.readlines()
                fin.write(J[0])
                for line in J[1:]:
                    data = line.split()
                    # if fileinput.lineno()>1:
                    if data[0] in list(gene_disease) or data[0] in gene_pvalues:
                        fin.write(line)
                    elif data[-1] in list(gene_disease) or data[-1] in gene_pvalues:
                        fin.write(line)
                fin.close()
            else:
                print("\n The gene-disease file %s is in a bad format" % self.gene_disease)
                H = open(self.outfolder+"NETWORK_RESULT/network_LD.out")
                J = H.readlines()
                fin.write(J[0])
                for line in J[1:]:
                    data = line.split()
                    # if fileinput.lineno()> 1:
                    if data[0] in gene_pvalues or data[-1] in gene_pvalues:
                        fin.write(line)
                fin.close()
        else:

            H = open(self.outfolder+"NETWORK_RESULT/network_LD.out")
            J = H.readlines()
            fin.write(J[0])
            for line in J[1:]:
                # for line in open(self.outfolder+"NETWORK_RESULT/network_LD.out"):
                data = line.split()
                # if fileinput.lineno()>1:
                if data[0] in gene_pvalues or data[-1] in gene_pvalues:
                    fin.write(line)
            fin.close()

    def subnetwork_anc(self, sub_module, LD, anc=''):
        '''
        This function take a biological network and a significant modules to create a subnetwork
        '''
        fi = open(self.outfolder+"NETWORK_RESULT/network_LD.out", "wt")
        fi.writelines("GeneA"+"\t"+"GeneB"+"\t"+"LD"+"\n")
        tmp = [];genes = [];anc_av = [];PPI = set()
        M = LD.values();M_a = np.mean(M)
        network = open(self.networkFile)
        for line in network:
            data = line.split()
            if data[0] in sub_module and data[1] in sub_module:
                if data[0]+":"+data[1] in LD:
                    GeneLD = data[0]+":"+data[1]
                    if LD[data[0]+":"+data[1]] >= float(self.LDCutoff):
                        if data[1]+":"+data[0] not in PPI:
                            PPI.add(GeneLD+":"+str(LD[data[0]+":"+data[1]]))
                    else:
                        if data[0] in self.diseaseGenes or data[1] in self.diseaseGenes:
                            GeneLD = data[0]+":"+data[1]
                            if data[0]+":"+data[1] not in PPI or data[1]+":"+data[0] not in PPI:
                                PPI.add(GeneLD+":"+str(LD[data[0]+":"+data[1]]))
                elif data[1]+":"+data[0] in LD:
                    GeneLD = data[1]+":"+data[0]
                    if LD[data[1]+":"+data[0]] >= float(self.LDCutoff):
                        if data[0]+":"+data[1] not in PPI:
                            PPI.add(GeneLD+":"+str(LD[data[1]+":"+data[0]]))
                    else:
                        if data[0] in self.diseaseGenes or data[1] in self.diseaseGenes:
                            GeneLD = data[0]+":"+data[1]
                            if data[0]+":"+data[1] not in PPI or data[1]+":"+data[0] not in PPI:
                                PPI.add(GeneLD+":"+str(LD[data[1]+":"+data[0]]))
                else:
                    if data[0] in self.diseaseGenes or data[1] in self.diseaseGenes:
                        GeneLD = data[0]+":"+data[1]
                        PPI.add(GeneLD+":"+str(M_a))
        for des in list(PPI):
            dim = des.split(":")
            fi.write("\t".join(dim)+"\n")
            genes += dim
        fi.close()


class ancMETA(ancInit, ancUtils, COMB_pvalues, qqvalues_plot, ancmap, ancLD, ancGraph, ancScoring):
    '''
    Performs the overrall steps of ancMETA as described in the ancMETA method (Chimusa et al 2013)
    '''

    def anc(self):
        '''
        Running ancMETA
        '''
        logger.info('Starting at time:%s' % str(datetime.datetime.today()))

        self.res = ancMETA(argv)

        logger.info("Loading parameters from %s ..." % os.path.abspath(argv))
        logger.info("Options in effect:")

        for param in sorted(self.res.Params):
            logger.info(' '+str(self.res.Params[param][0])+': '+str(self.res.Params[param][1]))
        print ''

        logger.info('Mapping SNPs to gene ...')
        
        self.res.readGWAS()
        self.res.readAffy()
        
        logger.info('Computing Gene pvalues ...')
        logger.info('Method of combining Pvalues at gene Level is %s.' % self.Gene_pv)
        if self.Gene_pv.capitalize() == 'Simes':
            self.res.simes()
        elif self.Gene_pv.capitalize() == 'Smallest':
            self.res.smallest()
        elif self.Gene_pv.capitalize() == 'Fisher':
            self.res.fisher()
        elif self.Gene_pv.capitalize() == 'Gwbon':
            self.res.FDR()
        else:
            self.res.gene_all()
        
        logger.info('Method of combining effect size at gene Level is %s.' % self.Gene_pv)
        self.res.gene_effect_size()
        logger.info('Plotting top genes from gene-based Mata-Annalysis ....')

        # self.res.write_move(self.outfolder+"GENE_RESULT/")
        if HAS_NP:
            try:
                self.res.mmplot("gene", self.outfolder+"GENE_RESULT/")

            except (RuntimeError, TypeError, NameError, IndexError):
                sys.stderr.write('Error => in Forest Plotting ....!\n')
            finally:
                pass
        else:
            logger.info('Skip generating Forest plotting, no RPY2 packages ....')

        if len(self.opts1) == 26:
            tagget_dict = self.res.data_comb(self.snpFile, self.genoFile)
            logger.info('Writing into a file the Gene-Gene LD ...')
            logger.info("Using the %s method for both subnetwork significance and LD ..." % self.Gene_LD)
            if self.Gene_LD.lower() == 'zscore':
                self.res.ZscoreLDGene(
                    self.networkFile, tagget_dict, self.Gene_LD.lower())
            elif self.Gene_LD.lower() == 'closest':
                self.res.closestLDGene(self.networkFile, tagget_dict)
            elif self.Gene_LD.lower() == 'maxscore':
                self.res.maxLDGene(self.networkFile, tagget_dict)
        currenttime1 = time.asctime(time.localtime())
        
        logger.info('Finish weighting the network at %s' % str(currenttime1))
        self.res.readGraph()

        currenttime = time.asctime(time.localtime())
        logger.info('Start searching at %s' % str(currenttime))
        self.res.subgraphFinding()
        currenttime1 = time.asctime(time.localtime())
        logger.info('Scoring generated subnetwork at %s' % str(currenttime1))
        
        self.res.Scoring()
        self.res.terminate()
        logger.info("Finish at time:%s" % str(datetime.datetime.today()))


if __name__ == '__main__':
    head = True
    try:
        global gpv1
        argv = sys.argv[1]
    except IndexError:
        argv = ''
    finally:
        run = ancMETA(argv)
        run.anc()
