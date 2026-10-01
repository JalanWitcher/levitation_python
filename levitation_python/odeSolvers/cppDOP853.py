import numpy as np
import ode_cpp

from typing import Tuple, Union, Optional, Sequence, List
# from ...odeSolvers import generalAux
from levitation_python import generalAux
from scipy.interpolate import CubicSpline, CubicHermiteSpline

def integraCPP(tAux: Union[Sequence[float], np.ndarray], 
               Y0: Union[List[float], Tuple[float, float]],
               k: float, gEf: float, B: float, zEq: float, Lambda: float,
               Amp: Union[float, Sequence[float]] = 2.5, 
               Phi: Union[float, Sequence[float]] = 0., 
               dA: float = 0., omega: float = 0.,
               maxPeaks: int = 100, peaksTimeStart: float = 0., 
               rtol: float = 1e-7, atol: float = 1e-10, 
               oldSol: Optional[generalAux.DotDict] = None) -> generalAux.DotDict:
    """
    Executes the C++ DOP853 integrator to solve the Acoustic Levitation ODE.
      dz/dt = v
      dv/dt = A * gEf * [1 + dA * sin(omega * t)] * cos(2*k*z - Phi) - gEf - B*v

    **Driving Modes:**
    * **0 (Amplitude On/Off):** Alternates amplitude between `Amp[0]` and `Amp[1]` with a fixed phase `Phi`.
    * **1 (Phase Jumps):** Alternates phase between `Phi[0]` and `Phi[1]` with a fixed amplitude `Amp`.
    * **2 (Sine Modulation):** Continuous amplitude modulation driven by `omega` and `dA`.

    **Parameters:**
    * `tAux` (array_like): Time breakpoints for the integration sub-intervals. Its bounds [t0, tFinal] defines the integration Global time.
    * `Y0` (array_like): Initial state [z0, v0].
    * `k`, `gEf`, `B` (float): Wavenumber, effective gravity, and damping coefficient.
    * `zEq`, `Lambda` (float): Equilibrium position and acoustic wavelength (used for escape detection).
    * `Amp` (float | list): Acoustic force amplitude (multiples of gEf). A list triggers Mode 0.
    * `Phi` (float | list): Acoustic force phase (degrees). A list triggers Mode 1.
    * `dA`, `omega` (float): Amplitude modulation factor and angular frequency. Non-zero values of `omega` trigger Mode 2.
    * `maxPeaks` (int): Maximum number of minima/maxima to track. Tracks all if <= 0.
    * `peaksTimeStart` (float): Global time at which peak tracking begins.
    * `rtol`, `atol` (float): Relative and absolute tolerances for the DOP853 solver.
    * `oldSol` (Optional[DotDict]): Previous solution to concatenate with the new integration.

    **Returns:**
    * `generalAux.DotDict`: Attribute callable Dictionary containing the time arrays, state arrays, tracked peaks, and continuous spline interpolators.
    """
    
    if isinstance(Amp, (list, tuple)):
        mode = 0
        val0, val1 = Amp[0], Amp[-1]
        baseA = val0
        basePhi = Phi if not isinstance(Phi, (list, tuple)) else Phi[0]
    elif isinstance(Phi, (list, tuple)):
        mode = 1
        val0, val1 = Phi[0], Phi[-1]
        baseA = Amp
        basePhi = val0
    elif omega != 0 or dA != 0:
        mode = 2
        val0 = val1 = 0.
        baseA = Amp
        basePhi = Phi
    else:
        mode = 0
        val0 = val1 = Amp
        baseA = Amp
        basePhi = Phi
    t, z, v, tMax, zMax, tMin, zMin = ode_cpp.run_simulation(z0=Y0[0], v0=Y0[1], breakpoints=tAux,
                                                              k=k, gEf=gEf, B=B, zEq=zEq,Lambda=Lambda,
                                                              val0=val0, val1=val1, base_A=baseA, base_phi=basePhi, dA=dA, omega=omega,
                                                              max_peaks=maxPeaks, trackerStartTime=peaksTimeStart, simMode=mode, rtol=rtol, atol=atol)

    if oldSol is None:
        tEf, zEf, vEf = t, z, v
        tMaxEf, zMaxEf, tMinEf, zMinEf = tMax, zMax, tMin, zMin
    else:
        tEf = np.concatenate((oldSol.t, t))
        zEf = np.concatenate((oldSol.z, z))
        vEf = np.concatenate((oldSol.v, v))
        tMaxEf = np.concatenate((oldSol.Eventos['eventoMax'], tMax))
        tMinEf = np.concatenate((oldSol.Eventos['eventoMin'], tMin))
        zMaxEf = np.concatenate((oldSol.Eventos_z['eventoMax'], zMax))
        zMinEf = np.concatenate((oldSol.Eventos_z['eventoMin'], zMin))

    tUnique, indexUnique = np.unique(tEf, return_index=True)
    yUnique = np.vstack((zEf, vEf))[:, indexUnique]
    solCppAll = CubicSpline(tUnique, yUnique, axis=1)
    solCppZ = CubicHermiteSpline(x=tUnique, y=yUnique[0], dydx=yUnique[1])

    def solCppFull(t):
        y = solCppAll(t)
        y[0] = solCppZ(t)
        return y
    solCppFull.t_min = tUnique[0]
    solCppFull.t_max = tUnique[-1]

    resultado = generalAux.DotDict(
        t=tUnique, z=yUnique[0], v=yUnique[1], sol=solCppFull,
        Eventos=dict(eventoMax=tMaxEf, eventoMin=tMinEf),
        Eventos_z=dict(eventoMax=zMaxEf, eventoMin=zMinEf),
        Integrou=bool(abs(t[-1] - tAux[-1]) < 1e-8),
        t_min=solCppFull.t_min, t_max=solCppFull.t_max)

    return resultado