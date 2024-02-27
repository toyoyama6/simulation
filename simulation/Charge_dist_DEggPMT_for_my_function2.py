import pickle
import csv
import numpy as np
import pandas as pd
import math
from glob import glob
from matplotlib import pyplot as plt
from scipy.special import erfcinv
import scipy.optimize
import random
from scipy import interpolate

curve_fit = scipy.optimize.curve_fit
import matplotlib.cm as cm
import copy
from sys import argv
import os

from PMTinfo import PMTinfo

plt.rcParams["figure.dpi"] = 300


class Charge_dist_DEggPMT:
    def __init__(self, PMTID=-1, isDebug=False):
        #### constructor #####################
        ## 1st argument --> PMTID in integer
        ## 2nd argument --> flag for debugging
        ## PMTID==-1 corresponds to an average
        ## of goldenPMTs
        ######################################

        # load the information of QE measured by HPK
        self.pmtinfo = PMTinfo()
        self.PMTID = PMTID
        ################################################

        self.nMPE_zen = 64
        self.nMPE_azi = 72
        self.MPE_zen = np.linspace(0, 60, self.nMPE_zen)
        self.MPE_azi = np.linspace(0, 360, self.nMPE_azi)
        Zen, Azi = np.meshgrid(self.MPE_zen, self.MPE_azi)
        self.listPMT = [283, 291, 428, 551, 657, 775, 866, 953, 967, 987, 991]
        # PMTID==-1 --> average of PMTs
        aQEs = []
        for iPMT in range(len(self.listPMT)):
            PMTID = self.listPMT[iPMT]
            aQE = self.pmtinfo.GetQE(int(PMTID), 400)
            aQEs.append(aQE)
        self.QE = np.mean(aQEs)

        def logistic_function(x, A, k, x0):
            return -1 / (1 + A * np.exp(-k * (x - x0))) + 1

        def linear_function(x, a, b):
            return a * x + b
        

        p0 = [3000, 0.16, 10]
        df = pd.read_hdf("./theta_pde.hdf5")
        df = df.to_numpy()
        df = np.tile(np.mean(df, axis = 0), (len(df), 1))
        a, b = -0.001, 1
        # func = linear_function(Zen, a, b)
        func = logistic_function(Zen, p0[0], p0[1], p0[2])
        # func[:, :50] = 1 
        # df1 = df.T * func.T
        df1 = df.T * logistic_function(Zen, p0[0], p0[1], p0[2]).T
        df1 = df1.T
        # np.save("./0.1_data", df1)
        self.MPE_val = df1

        if isDebug:
            plt.title("AE distribution", fontsize=25)
            plt.xlabel(r"$\theta$", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.plot(self.MPE_zen, df.T, label = "morii-san's data", color = "goldenrod", alpha = 0.4)
            plt.plot(self.MPE_zen, self.MPE_val.T, label = "modified", color = "royalblue", alpha = 0.2)
            plt.plot(self.MPE_zen, func[0], label = "logistic func", color = "green")
            # plt.plot(self.MPE_zen, logistic_function(self.MPE_zen, p0[0], p0[1], p0[2]), label = "logistic func", color = "green")
            # clb = plt.colorbar(cc)
            # clb.ax.tick_params(labelsize = 15)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=15)
            plt.tight_layout()
            plt.savefig(f"./uniformity_data_of_{self.PMTID}.pdf")
            plt.close()

    ################################################

   
    def GetMPE_ov_QE(self, zen, phi):
        # returns "PDE"
        # phi is defined in [-180, 360]
        idx_phi = np.int32(self.nMPE_azi * phi / 360.0) % self.nMPE_azi
        izen = np.digitize(zen, self.MPE_zen) - 1

        return np.where((zen > 60) | (izen < 0), -1, self.MPE_val[idx_phi, izen])

        try:
            return self.MPEval_interpolate[idx_phi](zen)
        except:
            return -1

    def GetChargeDist(self, zen, phi):
        # returns PDF (dP/dq) for arbitrary
        # location of zenith and phi.
        # phi is defined in [-180, 360].
        # zenith is defined in [0, 60].
        # If non sensitive region is specified, -1 is returned.
        MPEfac = self.GetMPE_ov_QE(zen, phi) / self.GetMPE_ov_QE(0, 0)

        return np.where(MPEfac > 0, MPEfac, 0)

    
    def GetChargeDistFromLocalXY(self, X, Y):
        # returns PDF (dP/dq) for arbitrary
        # location of X and Y.
        # X is defined in [-101, 101] mm.
        # Y is defined in [-101, 101] mm.
        # If non sensitive region is specified, -1 is returned.
        zen, phi = self._convertXYtoZenPhi(X, Y)
        return self.GetChargeDist(zen, phi)

    ################### private functions ####################
    ####### they are not intended to be used by users ########
    def _convertXYtoZenPhi(self, X, Y):
        phi = np.arctan2(Y, X) * 180.0 / np.pi
        rho = np.sqrt(X * X + Y * Y)
        R1 = 131.0
        R2 = 57.0894
        zen = None

        zen = np.where(rho>101, -1, \
			np.where(rho>80, np.arctan(rho/(np.sqrt(R2*R2-(rho-43.9106)**2)+59.5))*180.0/np.pi, np.arcsin(rho/R1)*180.0/np.pi))
        return zen, phi


        if rho > 101:
            return -1, -1
        elif rho > 80:
            zen = (
                np.arctan(rho / (np.sqrt(R2 * R2 - (rho - 43.9106) ** 2) + 59.5))
                * 180.0
                / np.pi
            )
        else:
            zen = np.arcsin(rho / R1) * 180.0 / np.pi
        return zen, phi


if __name__ == "__main__":
    PMTID = input("PMT ID: ")
    chgdist = Charge_dist_DEggPMT(PMTID=PMTID, isDebug=True)
    # q = np.linspace(0, 5, 100)

    # nx = 30
    # ny = 30
    # xs = np.linspace(-120, 120, nx)
    # ys = np.linspace(-120, 120, ny)
    # AE = np.empty((nx, ny))
    # for ix in range(len(xs)):
    #     for iy in range(len(ys)):
    #         x = xs[ix]
    #         y = ys[iy]
    #         PDF = chgdist.GetChargeDistFromLocalXY(x, y)
    #         AE[ix][iy] = PDF

    # # plt.savefig(f"./pdf_SQ{PMTID}.pdf")
    # # plt.close()

    # fig, ax = plt.subplots()
    # cc = ax.contourf(xs, ys, AE.T, 100)
    # plt.title("AE map for my function", fontsize=25)
    # plt.xlabel("x", fontsize=20)
    # plt.ylabel("y", fontsize=20)
    # plt.colorbar(cc)
    # plt.savefig(f"AE_map_{PMTID}.pdf")
