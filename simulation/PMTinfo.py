import numpy as np
import matplotlib.pyplot as plt

class PMTinfo:

    def __init__(self):


        data = np.loadtxt("PMTDB.csv", delimiter=",")
        
        listPMT = np.int32(data[0]+0.001)

        self.QEs = {}
        for idx in range(len(listPMT[1:])):
            PMTID = int(listPMT[1:][idx]+0.001)
            self.QEs[PMTID] = data[:,idx+1][1:]



    def GetQE(self, PMTID, lam):
        i1 = np.int32(lam/10)-27
        w2 = lam/10.0-27-i1
        if(i1<0 or i1>41):
            return 0
        elif(i1<=40):
            QE1 = self.QEs[PMTID][i1]
            QE2 = self.QEs[PMTID][i1+1]
            return (QE1*(1-w2)+QE2*w2)/100
        elif(i1==41):
            return self.QEs[PMTID][i1]/100


if(__name__=="__main__"):

    pmtinfo = PMTinfo()
    print("QE of SQ0725={0}".format(pmtinfo.GetQE(725, 680)))

    QE = []
    lams = np.linspace(200, 700, 100)
    for lam in lams:
        QE.append(pmtinfo.GetQE(725, lam))
    plt.plot(lams, QE)
    plt.title("QE", fontsize=20)
    plt.xlabel("Wavelength [nm]", fontsize=20)
    plt.ylabel("QE", fontsize=20)
    plt.grid()
    plt.tick_params(labelsize=15)
    plt.tight_layout()
    plt.show()
