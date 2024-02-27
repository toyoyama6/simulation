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
    def __init__(self, isDebug=False):
        df = pd.read_hdf("./2d_data.hdf")
        self.r = df.columns.values * 3
        self.theta = df.index.values * 6 * np.pi / 180
        R, PHI = np.meshgrid(self.r, self.theta)
        self.MPE_x = R * np.cos(PHI)  # + 4
        self.MPE_y = R * np.sin(PHI)  # - 11
        self.MPE_val = df
        # plt.plot(np.ravel(self.MPE_y))
        # plt.show()

        plt.pcolormesh(self.MPE_x, self.MPE_y, df)
        plt.grid()
        plt.show()

        # for i in np.ravel(self.MPE_y):
        #     plt.plot(df[])
        # plt.show()
        # plt.close()
        # print("QE from DB:", self.QE_HPK)
        # self.MPEval_interpolate = [None for i in range(self.nMPE_azi)]
        # for idx_phi in range(self.nMPE_azi):
        #     self.MPEval_interpolate[idx_phi] = interpolate.interp1d(self.MPE_zen, MPE_val[idx_phi])

        if isDebug:
            plt.title("AE distribution", fontsize=25)
            plt.xlabel("x", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.plot(self.MPE_val.T)
            # clb = plt.colorbar(cc)
            # clb.ax.tick_params(labelsize = 15)
            plt.tight_layout()
            plt.savefig(f"./pdf_for_xyscan.pdf")
            plt.close()

    ################################################

    def GetMPE_ov_QE(self, X, Y):
        # returns "PDE"
        # phi is defined in [-180, 360]
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X) + np.pi
        # THETA = np.arctan2(Y, X) * 180 / np.pi
        r_index = np.digitize(r, self.r) - 1
        theta_index = np.digitize(theta, self.theta) - 1
        x = self.r[r_index] * np.cos(self.theta[theta_index])
        y = self.r[r_index] * np.sin(self.theta[theta_index])
        x_index = np.where(self.MPE_x == x)
        index = np.where(self.MPE_y[x_index] == y)
        indices = x_index[0][index[0][0]], x_index[1][index[0][0]]
        return self.MPE_val[indices[1]][indices[0]]
        """
        ix = np.argmin(np.abs(np.ravel(self.MPE_x) - X))
        x = np.ravel(self.MPE_x)[ix]
        iy = np.argmin(np.abs(np.ravel(self.MPE_y) - Y))
        y = np.ravel(self.MPE_y)[iy]

        x_index = np.where(self.MPE_x == x)
        y_index = np.where(self.MPE_y == y)
        """

    def GetChargeDist(self, X, Y):
        # returns PDF (dP/dq) for arbitrary
        # location of zenith and phi.
        # phi is defined in [-180, 360].
        # zenith is defined in [0, 60].
        # If non sensitive region is specified, -1 is returned.
        MPEfac = self.GetMPE_ov_QE(X, Y)  # / self.GetMPE_ov_QE(0, 0)

        return np.where(MPEfac > 0, MPEfac, 0)

    def GetChargeDistFromLocalXY(self, X, Y):
        # returns PDF (dP/dq) for arbitrary
        # location of X and Y.
        # X is defined in [-101, 101] mm.
        # Y is defined in [-101, 101] mm.
        # If non sensitive region is specified, -1 is returned.
        return self.GetChargeDist(X, Y)


if __name__ == "__main__":
    chgdist = Charge_dist_DEggPMT(isDebug=False)
    q = np.linspace(0, 5, 100)
    nx = 30
    ny = 30
    xs = np.linspace(-120, 120, nx)
    ys = np.linspace(-120, 120, ny)

    # plt.scatter(xs, ys)
    # plt.show()
    AE = np.empty((nx, ny))
    for ix in range(len(xs)):
        for iy in range(len(ys)):
            x = xs[ix]
            y = ys[iy]
            PDF = chgdist.GetChargeDistFromLocalXY(x, y)
            # print("(x, y) =", x, y)
            AE[ix][iy] = PDF
    # plt.savefig(f"./pdf_SQ{PMTID}.pdf")
    # plt.close()

    fig, ax = plt.subplots()
    cc = ax.contourf(xs, ys, AE.T, 100)
    plt.title("AE map for my function", fontsize=25)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.colorbar(cc)
    # plt.savefig(f"AE_map_xyscan.pdf")
    plt.show()
    plt.close()
