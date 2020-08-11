
# -----------------------------------------------------
from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz

from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import maps
import SimPEG.electromagnetics.time_domain as tdem

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import interpolate

import sys
#sys.path.append("../../AEMFaultMonteCarlo")
#import MonteCarlo

from shapely.geometry import Point, Polygon

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver
import pathlib
from os import fspath
DEFAULTPATH = fspath(pathlib.Path(__file__).parent.absolute())

# ----------Create Topo Points ----------------------
def createTopo(efile=None, y0=100, ny=11):
    if efile is None:
        efile = DEFAULTPATH+"/data/PH-2018-eloc.txt"
    eloc = pd.read_csv(efile,sep='\t',header=None)
    x = eloc.values[:,1]
    z = eloc.values[:,3]
    y = np.linspace(-y0, y0, ny)
    
    yy = np.repeat(y[:,np.newaxis],x.shape[0],axis=1)
    xx = np.repeat(x[:,np.newaxis],ny,axis=1).T
    zz = np.repeat(z[:,np.newaxis],ny,axis=1).T

    topo_xyz = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

    return topo_xyz

# --------- Create Fault Polygon ------------
def createFault(topo, xpos, H, dip, dep, xtra=1000):
    
    topo2 = np.concatenate((np.array([[topo[0,0]-xtra, topo[0,1]]]), topo, np.array([[topo[topo.shape[0]-1,0]+xtra,topo[topo.shape[0]-1,1]]])))

    e_bot = np.floor(np.min(topo2[:,1]))
    zbot = np.floor(np.min(topo2[:,1])) - dep

    Z = interpolate.interp1d(topo2[:,0],topo2[:,1])
    zx = Z(xpos)
    Dz = zx-zbot

    Dx = Dz/np.tan(np.deg2rad(dip))
    xbot = xpos + Dx

    Dxbot = H/2/np.sin(np.deg2rad(dip))

    LR = (xbot+Dxbot,zbot)
    LL = (xbot-Dxbot,zbot)

    zfR = zbot + (LR[0]-topo2[:,0])*np.tan(np.deg2rad(dip))
    diffR = interpolate.interp1d(zfR-topo2[:,1],topo2[:,0])
    UR = (float(diffR(0)), float(Z(diffR(0))) )

    zfL = zbot + (LL[0]-topo2[:,0])*np.tan(np.deg2rad(dip))
    diffL = interpolate.interp1d(zfL-topo2[:,1],topo2[:,0])
    UL = (float(diffL(0)), float(Z(diffL(0))) )
    
    idxabove = (topo2[topo2[:,0] > UL[0],0]).argmin() + (topo2[topo2[:,0] < UL[0],0]).shape[0]
    idxbelow = (topo2[topo2[:,0] < UR[0],0]).argmax()

    middles = [(topo[j,0],topo[j,1]) for j in range(idxabove,idxbelow)]
    verts = [LL, UL] + middles + [UR,LR]

    return verts

# ---------- Create mesh ---------------------
def createMesh(topo_xyz, x, dh=25.0, y0=100):
    dom_width = x[x.shape[0]-1]  # domain width
    dom_len = 2*y0
    dom_vert = 1600
    nbcx = 2 ** int(np.round(np.log(dom_width / dh) / np.log(2.0)))  # num. base cells
    nbcy = 2 ** int(np.round(np.log(dom_len / dh) / np.log(2.0)))  # num. base cells
    nbcz = 2 ** int(np.round(np.log(dom_vert / dh) / np.log(2.0)))  # num. base cells

    # Define the base mesh
    mesh = TreeMesh([[(dh, nbcx)], [(dh, nbcy)], [(dh, nbcz)]], x0=[0, -y0, 2000])

    # Mesh refinement based on topography
    mesh = refine_tree_xyz(
        mesh, topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
    )

    # Mesh refinement near transmitters and receivers
    mesh = refine_tree_xyz(
        mesh, receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
    )

    # Refine core mesh region
    xz = np.repeat(np.array(fpts),mesh.vectorCCy.shape[0],axis=0)
    ys = np.concatenate(tuple([np.repeat(mesh.vectorCCy[i],len(fpts)) for i in range(mesh.vectorCCy.shape[0])]))
    xyz = np.concatenate((xz[:,0][:,np.newaxis], ys[:,np.newaxis], xz[:,1][:,np.newaxis]),axis=1)

    mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False)

    mesh.finalize()

    return mesh

# -------------- Simulate --------------------
def runSim(mesh, survey, model, model_map, t0, time_steps, outfile=None):

    simulation = tdem.simulation.Simulation3DMagneticFluxDensity(
        mesh, survey=survey, sigmaMap=model_map, Solver=Solver, t0=t0
    )
    
    simulation.time_steps = time_steps

    dpred = simulation.dpred(model)
    
    dpred_plotting = np.reshape(dpred, (survey.nSrc, int(dpred.shape[0]/survey.nSrc)))
    
    if outfile is not None:
        np.savetxt(outfile, dpred_plotting, delimiter=",")

    return dpred_plotting

class PHaem:
    def __init__(self, dip=60, H=100, xpos=250, rho_fault=30, rho_back=50, efile=None, efile2=None, sfile=None, dfile=None, tfile=None, wfile=None, dh=25.0, y0=100, ny=11, dep=1000, xtra=500, outfile=None):
        self.dep = dep
        self.dip = dip
        self.H = H
        self.xpos = xpos
        self.rho_fault = rho_fault
        self.rho_back = rho_back
        self.outfile = outfile
        self.xtra = xtra
        self.y0 = y0
        self.dh = dh
        self.ny = ny
        if efile is None:
            self.efile = DEFAULTPATH+"/data/Cross-PH.txt"
        else:
            self.efile = efile

        if sfile is None:
            self.sfile = DEFAULTPATH+"/data/PH-2018-srv.txt"
        else:
            self.sfile = sfile

        if dfile is None:
            self.dfile = DEFAULTPATH+"/data/RawAEMData.csv"
        else:
            self.dfile = dfile

        if efile2 is None:
            self.efile2 = DEFAULTPATH+"/data/PH-2018-eloc.txt"
        else:
            self.efile2 = efile2

        if tfile is None:
            self.tfile = DEFAULTPATH+"/data/time_gates.txt"
        else:
            self.tfile = tfile

        if wfile is None:
            self.wfile = DEFAULTPATH+"/data/waveform.txt"
        else:
            self.wfile = wfile

        self.rhomap = [[0, rho_back],
              [1, rho_fault],
              [2, rho_back]]
        self.topo_xyz = createTopo(efile, y0=y0, ny=ny)

    def create_fault(self):
        topo, _ = createTopo(self.efile,self.xtra,self.dep)
        return createFault(topo, self.xpos, self.dip, self.H, self.dep, xtra=self.xtra)

    def create_geom(self):
        topo, tpoly = createTopo(self.efile,self.xtra,self.dep)
        fpoly = createFault(topo, self.xpos, self.dip, self.H, self.dep, xtra=self.xtra)
        return tpoly+fpoly 

    def create_waveform(self):
        wave = pd.read_csv(self.wfile,sep=' ')
        timegates = pd.read_csv(self.tfile,sep='   ', engine='python')

        def wave_function(t):
            wfile = 'data/waveform.txt'
            wave = pd.read_csv(wfile,sep=' ')
            if t < wave.Time.values[0] or t > wave.Time.values[wave.Time.values.shape[0]-1]:
                val = 0
            else:
                w = interpolate.interp1d(wave.Time.values,wave.Amperes_Normalized)
                val = w(t)
            return val

        self.waveform = tdem.sources.RawWaveform(waveFct=wave_function, offTime=0.)
        return self.waveform


    def create_survey(self):
        dat = pd.read_csv(self.dfile)
        wave = pd.read_csv(self.wfile,sep=' ')
        ert = pd.read_csv(self.efile, sep='\t')
        ert2 = pd.read_csv(self.efile2,sep='\t',header=None)
        timegates = pd.read_csv(self.tfile,sep='   ', engine='python')

        lines = np.unique(dat.LINE)
        k=9
        dat2 = dat[dat.LINE == lines[k]]

        Xert = np.concatenate(([ert.Easting.values],[ert.Northing.values]),axis=0).T
        Yert = ert2.values[:,1]
        reg = LinearRegression().fit(Xert, Yert)
        Xaem = np.concatenate(([dat.X[dat.LINE == lines[k]].values],[dat.Y[dat.LINE == lines[k]].values]),axis=0).T
        Yaem = reg.predict(Xaem)

        xtx = Yaem[Yaem <= Yert[Yert.shape[0]-1]]
        ytx = np.zeros(xtx.shape)
        ztx = dat2.ALT.values[Yaem <= Yert[Yert.shape[0]-1]] + dat2.TOPO.values[Yaem <= Yert[Yert.shape[0]-1]]

        xrx = xtx
        yrx = ytx
        zrx = ztx
        
        self.source_locations = np.c_[mkvc(xtx), mkvc(ytx), mkvc(ztx)]
        self.receiver_locations = np.c_[mkvc(xrx), mkvc(yrx), mkvc(zrx)]
        
        self.time_channels = timegates.Gatecenter.values
        source_list = []  # Create empty list to store sources
        for ii in range(xtx.shape[0]):

            # Here we define receivers that measure the h-field in A/m
            dbzdt_receiver = tdem.receivers.PointMagneticFluxTimeDerivative(
                self.receiver_locations[ii, :], self.time_channels, "z"
            )
            receivers_list = [
                dbzdt_receiver
            ]  # Make a list containing all receivers even if just one

            # Must define the transmitter properties and associated receivers
            source_list.append(
                tdem.sources.MagDipole(
                    receivers_list,
                    location=self.source_locations[ii],
                    waveform=self.waveform,
                    moment=1.0,
                    orientation="z",
                )
            )

        self.survey = tdem.Survey(source_list)
        return self.survey

    def create_mesh(self):
        eloc = pd.read_csv(self.efile2,sep='\t',header=None)
        topo = eloc.values[:,(1,3)]

        dom_width = topo[topo.shape[0]-1,0]  # domain width
        dom_len = 2*self.y0
        dom_vert = np.min(topo[:,1]) - self.dep
        nbcx = 2 ** int(np.round(np.log(dom_width / self.dh) / np.log(2.0)))  # num. base cells
        nbcy = 2 ** int(np.round(np.log(dom_len / self.dh) / np.log(2.0)))  # num. base cells
        nbcz = 2 ** int(np.round(np.log(dom_vert / self.dh) / np.log(2.0)))  # num. base cells

        # Define the base mesh
        mesh = TreeMesh([[(self.dh, nbcx)], [(self.dh, nbcy)], [(self.dh, nbcz)]], x0=[0, -self.y0, 2000])

        # Mesh refinement based on topography
        mesh = refine_tree_xyz(
            mesh, self.topo_xyz, octree_levels=[0, 0, 0, 1], method="surface", finalize=False
        )

        # Mesh refinement near transmitters and receivers
        mesh = refine_tree_xyz(
            mesh, self.receiver_locations, octree_levels=[2, 4], method="radial", finalize=False
        )

        # Refine core mesh region
        self.fpts = createFault(topo, self.xpos, self.H, self.dip, self.dep, xtra=1000)
        xz = np.repeat(np.array(self.fpts),mesh.vectorCCy.shape[0],axis=0)
        ys = np.concatenate(tuple([np.repeat(mesh.vectorCCy[i],len(self.fpts)) for i in range(mesh.vectorCCy.shape[0])]))
        xyz = np.concatenate((xz[:,0][:,np.newaxis], ys[:,np.newaxis], xz[:,1][:,np.newaxis]),axis=1)

        mesh = refine_tree_xyz(mesh, xyz, octree_levels=[0, 2, 4], method="box", finalize=False)

        mesh.finalize()
        self.mesh = mesh

        return mesh

    def create_model(self):
        fault = Polygon(self.fpts)

        # Conductivity in S/m
        air_conductivity = 1e-8
        background_conductivity = 1/self.rho_back
        block_conductivity = 1/self.rho_fault

        # Active cells are cells below the surface.
        self.ind_active = surface2ind_topo(self.mesh, self.topo_xyz)
        self.model_map = maps.InjectActiveCells(self.mesh, self.ind_active, air_conductivity)

        # Define the model
        model = background_conductivity * np.ones(self.ind_active.sum())

        inds_fault = np.array([Point(self.mesh.gridCC[i,0],self.mesh.gridCC[i,2]).within(fault) for i in range(self.mesh.nC)])
        inds_fault = inds_fault[self.ind_active] 
        model[inds_fault] = block_conductivity
        self.model = model
        return model

    def create_timesteps(self):
        wave = pd.read_csv(self.wfile,sep=' ')
        timegates = pd.read_csv(self.tfile,sep='   ', engine='python')

        t0w = wave.Time.values[0]

        dtw = np.array([j-i for i, j in zip(wave.Time.values[:-1], wave.Time.values[1:])])

        t0g = timegates.Gatecenter.values[0]
        dtg = timegates.Gatewidth.values

        self.time_steps = np.concatenate((dtw, [t0g], dtg))
        self.t0 = t0w
        return

    def plot_slice(self):
        mpl.rcParams.update({"font.size": 12})
        fig = plt.figure(figsize=(7, 6))

        log_model = np.log10(self.model)

        plotting_map = maps.InjectActiveCells(self.mesh, self.ind_active, np.nan)

        ax1 = fig.add_axes([0.13, 0.1, 0.6, 0.85])
        self.mesh.plotSlice(
            plotting_map * log_model,
            normal="Y",
            ax=ax1,
            ind=int(self.mesh.hy.size / 2),
            grid=True,
            clim=(np.min(log_model), np.max(log_model)),
        )
        ax1.set_title("Conductivity Model at Y = 0 m")

        ax2 = fig.add_axes([0.75, 0.1, 0.05, 0.85])
        norm = mpl.colors.Normalize(vmin=np.min(log_model), vmax=np.max(log_model))
        cbar = mpl.colorbar.ColorbarBase(
            ax2, norm=norm, orientation="vertical", format="$10^{%.1f}$"
        )
        cbar.set_label("Conductivity [S/m]", rotation=270, labelpad=15, size=12)
        return

    def setup_simulation(self):
        self.create_waveform()
        self.create_survey()
        self.create_mesh()
        self.create_model()
        self.create_timesteps()
        return
       
    def simulate(self):
        self.data = runSim(self.mesh, self.survey, self.model, self.model_map, self.t0, self.time_steps, outfile=self.outfile)
        return self.data

    def plot_data(self, vmin=0, vmax=1e-10, shading='auto'):
        xv,yv = np.meshgrid(self.receiver_locations[:,0],self.time_channels)
        plt.pcolormesh(xv/1000,yv,np.abs(self.data.T),vmin=vmin,vmax=vmax,shading=shading)
        plt.gca().invert_yaxis()
        cb=plt.colorbar()
        cb.ax.set_ylabel('dB/dt')
        plt.yscale('log')
        plt.xlabel('Profile Distance (km)')
        plt.ylabel('Time (s)')
        plt.title('Simulated Data')
        return

    
