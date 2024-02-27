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

        # print("PMTID = SQ{0:0>4}".format(PMTID))

        ################################################
        # read information of the q0 and sigma_q0
        nbinzen = 40
        nbinazi = 4
        SQ0428dir = "./SQ0428"  # SPE data location

        npar_q0 = 4
        npar_chgres = 4

        self.par_q0 = np.zeros((nbinazi, npar_q0))
        self.par_chgres = np.zeros((nbinazi, npar_chgres))
        self.q0c = None
        self.tau = 0.25  # in pC (fixed)

        fin = open(SQ0428dir + "/profilefitpar.txt", "r")
        for id in range(nbinazi):
            line = fin.readline()
            data = line.split("\t")
            ipar = int(data[1])
            val0 = float(data[2])
            val1 = float(data[3])
            val2 = float(data[4])
            val3 = float(data[5])
            self.par_chgres[ipar][0] = val0
            self.par_chgres[ipar][1] = val1
            self.par_chgres[ipar][2] = val2
            self.par_chgres[ipar][3] = val3

        for id in range(nbinazi):
            line = fin.readline()
            data = line.split("\t")
            ipar = int(data[1])
            val0 = float(data[2])
            val1 = float(data[3])
            val2 = float(data[4])
            val3 = float(data[5])
            self.par_q0[ipar][0] = val0
            self.par_q0[ipar][1] = val1
            self.par_q0[ipar][2] = val2
            self.par_q0[ipar][3] = val3
            self.q0c = val0
        fin.close()

        ################################################
        # read information of reference PMT
        values = np.loadtxt("./ana_refPMT/fitpar.txt")
        self.tauref = float(values[5])
        self.alref = float(values[4])
        self.q0ref = float(values[7])
        del values
        ################################################

        self.nMPE_zen = 64
        self.nMPE_azi = 72
        self.MPE_zen = np.linspace(0, 60, 64)
        self.MPE_azi = np.linspace(0, 360, 72)

        # read information of MPE measurement
        if PMTID == -1:
            self.listPMT = [283, 291, 428, 551, 657, 775, 866, 953, 967, 987, 991]
            # PMTID==-1 --> average of PMTs

            self.QE_HPK = 0
            MPE_val = np.zeros((len(self.listPMT), self.nMPE_azi, self.nMPE_zen))

            aQEs = []
            for iPMT in range(len(self.listPMT)):
                PMTID = self.listPMT[iPMT]

                aQE = self.pmtinfo.GetQE(int(PMTID), 400)
                aQEs.append(aQE)
                self.QE_HPK += aQE / len(self.listPMT)
                MPEdata = "./data/SQ" + "{0:0>4}".format(PMTID) + "/pde_dict.pckl"

                with open(MPEdata, "rb") as f:
                    pde_dict = pickle.load(f)

                df = pd.json_normalize(pde_dict)
                dfnp = df.to_numpy()

                MPE_val[iPMT] = dfnp[0,].reshape((self.nMPE_azi, -1)) / aQE
            MPE_val = np.average(MPE_val, axis=0)
            self.QE = np.mean(aQEs)
            # plt.hist(aQEs)
            # plt.show()
            if isDebug:
                # fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                # ax.pcolormesh(self.MPE_azi*np.pi/180, self.MPE_zen, MPE_val.T)
                # cc = ax.contourf(self.MPE_azi*np.pi/180, self.MPE_zen, MPE_val.T)
                plt.title("AE distribution", fontsize=25)
                plt.xlabel(r"$\theta$", fontsize=20)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.plot(self.MPE_zen, MPE_val.T)
                # clb = plt.colorbar(cc)
                # clb.ax.tick_params(labelsize = 15)
                plt.tight_layout()
                plt.savefig(f"./uniformity_data_of_SQ00-1.pdf")
                plt.close()

        else:
            self.QE_HPK = self.pmtinfo.GetQE(int(PMTID), 400)
            self.QE = self.QE_HPK
            MPEdata = "./data/SQ" + "{0:0>4}".format(PMTID) + "/pde_dict.pckl"

            with open(MPEdata, "rb") as f:
                pde_dict = pickle.load(f)

            df = pd.json_normalize(pde_dict)
            dfnp = df.to_numpy()

            MPE_val = dfnp[0,].reshape((self.nMPE_azi, -1)) / self.QE_HPK
            if isDebug:
                plt.title("AE distribution", fontsize=25)
                plt.xlabel(r"$\theta$", fontsize=20)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.plot(self.MPE_zen, MPE_val.T)
                plt.tight_layout()
                plt.savefig(f"./uniformity_data_of_SQ{PMTID:0>4}.pdf")
                plt.close()
        MPE_val = MPE_val / np.max(MPE_val)
        self.MPE_val = MPE_val

        # print("QE from DB:", self.QE_HPK)
        self.MPEval_interpolate = [None for i in range(self.nMPE_azi)]
        for idx_phi in range(self.nMPE_azi):
            self.MPEval_interpolate[idx_phi] = interpolate.interp1d(
                self.MPE_zen, MPE_val[idx_phi]
            )

            ################################################

    def GetChgSigma(self, zen, phi):
        # returns sigma (not sigma/q0) in pC
        # phi is defined in [-180, 360]
        phi = np.where(phi < 0, phi + 360, phi)
        idx_phi = np.int32(phi / 90.0)
        w1 = (phi - idx_phi * 90) / 90.0
        return self._GetChgSigma_fromIdx(
            zen * np.pi / 180, idx_phi % 4
        ) * w1 + self._GetChgSigma_fromIdx(zen * np.pi / 180, (idx_phi + 1) % 4) * (
            1 - w1
        )

    def GetQ0(self, zen, phi):
        # returns q0 in pC
        # phi is defined in [-180, 360]
        phi = np.where(phi < 0, phi + 360, phi)
        idx_phi = np.int32(phi / 90.0)
        w1 = (phi - idx_phi * 90.0) / 90.0
        return self._GetQ0_fromIdx(
            zen * np.pi / 180, idx_phi % 4
        ) * w1 + self._GetQ0_fromIdx(zen * np.pi / 180, (idx_phi + 1) % 4) * (1 - w1)

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

    def GetSPETemplate(self, qSPE, zen, phi):
        # returns PDF (dP/dqspe) for arbitrary
        # location of zenith and phi, where qspe is photoelectron.
        # phi is defined in [-180, 360].
        # zenith is defined in [0, 60].
        # If non sensitive region is specified, -1 is returned.
        q = qSPE * self.q0c
        PDF = self.GetChargeDist(q, zen, phi)
        return np.where(PDF >= 0, PDF * self.q0c, -1)

    def GetSPETemplateFromLocalXY(self, qSPE, X, Y):
        # returns PDF (dP/dqspe) for arbitrary
        # location of X and Y, where qspe is photoelectron.
        # X is defined in [-101, 101] mm.
        # Y is defined in [-101, 101] mm.
        # If non sensitive region is specified, -1 is returned.
        zen, phi = self._convertXYtoZenPhi(X, Y)

        q = qSPE * self.q0c
        PDF = self.GetChargeDist(q, zen, phi)
        return np.where(PDF >= 0, PDF * self.q0c, -1)

    def GetChargeDistFromLocalXY(self, X, Y):
        # returns PDF (dP/dq) for arbitrary
        # location of X and Y.
        # X is defined in [-101, 101] mm.
        # Y is defined in [-101, 101] mm.
        # If non sensitive region is specified, -1 is returned.
        zen, phi = self._convertXYtoZenPhi(X, Y)
        return self.GetChargeDist(zen, phi)

    def GetChargeAverage(self, zen, phi):
        # returns averaged charge for arbitrary
        # location of zenith and phi.
        # phi is defined in [-180, 360].
        # zenith is defined in [0, 60].
        # If non sensitive region is specified, -1 is returned.
        MPEfac = self.GetMPE_ov_QE(zen, phi)
        q0 = self.GetQ0(zen, phi)
        bt = (
            MPEfac * self.q0c * (self.tauref * self.alref / self.q0ref + 1 - self.alref)
        )
        al = np.minimum((bt - self.q0c) / (self.tau - self.q0c), 1)
        return np.where(MPEfac > 0, al * self.tau + (1 - al) * q0, -1)

    def GetQE(self, lam):
        if self.PMTID != -1:
            return self.pmtinfo.GetQE(self.PMTID, lam)
        else:
            QEave = 0
            for aPMT in self.listPMT:
                QEave += self.pmtinfo.GetQE(aPMT, lam) / len(self.listPMT)
            return QEave

    ################### private functions ####################
    ####### they are not intended to be used by users ########
    def _convertXYtoZenPhi(self, X, Y):
        phi = np.arctan2(Y, X) * 180.0 / np.pi
        rho = np.sqrt(X * X + Y * Y)
        R1 = 131.0
        R2 = 57.0894
        zen = None

        zen = np.where(
            rho > 101,
            -1,
            np.where(
                rho > 80,
                np.arctan(rho / (np.sqrt(R2 * R2 - (rho - 43.9106) ** 2) + 59.5))
                * 180.0
                / np.pi,
                np.arcsin(rho / R1) * 180.0 / np.pi,
            ),
        )
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

    def _GetChgSigma_fromIdx(self, zen, idx_phi):
        a0 = self.par_chgres[idx_phi, 0]
        a2 = self.par_chgres[idx_phi, 1]
        a3 = self.par_chgres[idx_phi, 2]
        a4 = self.par_chgres[idx_phi, 3]
        return a0 + zen * zen * a2 + zen * zen * zen * a3 + a4 * zen**4

    def _GetQ0_fromIdx(self, zen, idx_phi):
        a0 = self.par_q0[idx_phi, 0]
        a2 = self.par_q0[idx_phi, 1]
        a3 = self.par_q0[idx_phi, 2]
        a4 = self.par_q0[idx_phi, 3]
        return a0 + zen * zen * a2 + zen * zen * zen * a3 + a4 * zen**4

    ##########################################################


if __name__ == "__main__":
    PMTID = int(input("PMT ID:"))

    chgdist = Charge_dist_DEggPMT(PMTID=PMTID, isDebug=True)
    q = np.linspace(0, 5, 100)

    nx = 30
    ny = 30
    xs = np.linspace(-120, 120, nx)
    ys = np.linspace(-120, 120, ny)
    AE = np.empty((nx, ny))
    for ix in range(len(xs)):
        for iy in range(len(ys)):
            x = xs[ix]
            y = ys[iy]
            PDF = chgdist.GetChargeDistFromLocalXY(x, y)
            AE[ix][iy] = PDF

    # plt.savefig(f"./pdf_SQ{PMTID}.pdf")
    # plt.close()

    fig, ax = plt.subplots()
    cc = ax.contourf(xs, ys, AE.T, 100)
    plt.title("AE map for SQ{0:0>4}".format(PMTID), fontsize=25)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.colorbar(cc)
    plt.savefig(f"AE_map_SQ{PMTID:0>4}.pdf")
