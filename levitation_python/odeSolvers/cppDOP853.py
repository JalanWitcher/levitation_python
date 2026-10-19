import numpy as np
import ode_cpp

from typing import Tuple, Union, Optional, Sequence, List
from levitation_python import generalAux
from scipy.interpolate import CubicSpline, CubicHermiteSpline

class PiecewiseCubicSpline:
    def __init__(self, t, y, breakpoints) -> None:
        self.breakpoints = breakpoints
        self.splines = []

        indices = np.atleast_1d(np.searchsorted(t, breakpoints))

        if len(indices) > 0:
            indices[-1] = len(t)

        # Create a spline for each interval
        for i in range(len(breakpoints) - 1):
            # tIni = breakpoints[i]
            # tFim = breakpoints[i+1]

            # intervalIdx = (t >= tIni) * (t <= tFim)
            # tAtu, yAtu = t[intervalIdx], y[intervalIdx]
            idxStart = indices[i]
            idxEnd = indices[i+1]
            
            tAtu, yAtu = t[idxStart:idxEnd], y[idxStart:idxEnd]
            if len(tAtu) < 2:
                self.splines.append(None)
                continue

            self.splines.append(CubicSpline(tAtu, yAtu))

    def __call__(self, t):
        t = np.atleast_1d(t)
        y = np.zeros_like(t, dtype=float)

        indices = np.searchsorted(self.breakpoints, t, side='left') - 1
        indices = np.clip(indices, 0, len(self.splines) - 1)

        for i, spline in enumerate(self.splines):
            if spline is None: continue
            mask = (indices == i)
            if np.any(mask):
                y[mask] = spline(t[mask])

        return y if y.size > 1 else y[0]

def integraCPP(tAux: Union[Sequence[float], np.ndarray], 
               Y0: Union[List[float], Tuple[float, float]],
               k: float, gEf: float, zEq: float, Lambda: float,
               Amp: Union[float, Sequence[float]] = 2.5, 
               Phi: Union[float, Sequence[float]] = 0., 
               dA: float = 0., omega: float = 0.,
               B: float = 0, fator_Re: float = 0., fator_Cd: float = 0., fator_fAr: float = 0,
               maxPeaks: int = 100, peaksTimeStart: float = 0., 
               rtol: float = 1e-7, atol: float = 1e-10, initialStep: float = 1e-4,
               oldSol: Optional[generalAux.DotDict] = None) -> generalAux.DotDict:
    """
    Executes the C++ DOP853 integrator to solve the Acoustic Levitation ODE.
      dz/dt = v
      dv/dt = A * gEf * [1 + dA * sin(omega * t)] * cos(2*k*z - Phi) - gEf - fAr

    **Driving Modes:**
    * **0 (Amplitude On/Off):** Alternates amplitude between `Amp[0]` and `Amp[1]` with a fixed phase `Phi`.
    * **1 (Phase Jumps):** Alternates phase between `Phi[0]` and `Phi[1]` with a fixed amplitude `Amp`.
    * **2 (Sine Modulation):** Continuous amplitude modulation driven by `omega` and `dA`.

    **Drag Modes:**
    * **0 (Linear Drag):** `fAr = b*v`.
    * **1 (Flemmer Drag):** `fAr = b(v)*v*|v|` with `b(v)` following Flemmer expression, based on the Reynolds number.

    **Parameters:**
    * `tAux` (array_like): Time breakpoints for the integration sub-intervals. Its bounds [t0, tFinal] defines the integration Global time.
    * `Y0` (array_like): Initial state [z0, v0].
    * `k`, `gEf`, `B` (float): Wavenumber, effective gravity, and damping coefficient.
    * `zEq`, `Lambda` (float): Equilibrium position and acoustic wavelength (used for escape detection).
    * `Amp` (float | list): Acoustic force amplitude (multiples of gEf). A list triggers Mode 0.
    * `Phi` (float | list): Acoustic force phase (degrees). A list triggers Mode 1.
    * `dA`, `omega` (float): Amplitude modulation factor and angular frequency. Non-zero values of `omega` trigger Mode 2.
    * `B`, `fator_Re`, `fator_Cd`, `fator_fAr` (float): Damping linear coefficient, Reynolds factor, Damping factor and linear coefficient aproximation.
    * `maxPeaks` (int): Maximum number of minima/maxima to track. Tracks all if <= 0.
    * `peaksTimeStart` (float): Global time at which peak tracking begins.
    * `rtol`, `atol` (float): Relative and absolute tolerances for the DOP853 solver.
    * `initialStep` (float): Initial step size for the integrator.
    * `oldSol` (Optional[DotDict]): Previous solution to concatenate with the new integration.

    **Returns:**
    * `generalAux.DotDict`: Attribute callable Dictionary containing the time arrays, state arrays, tracked peaks, and continuous spline interpolators.
    """

    # Define the simulation mode based on the provided parameters
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

    farMode = 0
    if B == 0:
        if (fator_Re != 0) and (fator_Cd != 0):
            farMode = 1

    systemParams = dict(k=k, gEf=gEf, zEq=zEq, Lambda=Lambda,
                        base_A=baseA, base_phi=basePhi, dA=dA, omega=omega, 
                        B=B, fator_Re=fator_Re, fator_Cd=fator_Cd, fator_fAr=fator_fAr)

    # Run the C++ DOP853 integrator to simulate the system dynamics
    t, z, v, a, tMax, zMax, tMin, zMin, status = ode_cpp.run_simulation(z0=Y0[0], v0=Y0[1], breakpoints=tAux,
                                                              sysParamsDict=systemParams, val0=val0, val1=val1,
                                                              max_peaks=maxPeaks, trackerStartTime=peaksTimeStart, 
                                                              simMode=mode, farMode=farMode, 
                                                              rtol=rtol, atol=atol, initialStep=initialStep)

    statusPy = dict(Integrou=status[0], simMode=status[1], farMode=status[2], nAccepted=status[3], nRejected=status[4])
    # Check if theres a previous solution to concatenate with the new integration results
    if oldSol is None:
        tEf, zEf, vEf, aEf = t, z, v, a
        tMaxEf, zMaxEf, tMinEf, zMinEf = tMax, zMax, tMin, zMin
        nSteps = statusPy["nAccepted"] + statusPy["nRejected"]
    else:
        tEf = np.concatenate((oldSol.t, t))
        zEf = np.concatenate((oldSol.z, z))
        vEf = np.concatenate((oldSol.v, v))
        aEf = np.concatenate((oldSol.a, a))
        tMaxEf = np.concatenate((oldSol.Eventos['eventoMax'], tMax))
        tMinEf = np.concatenate((oldSol.Eventos['eventoMin'], tMin))
        zMaxEf = np.concatenate((oldSol.Eventos_z['eventoMax'], zMax))
        zMinEf = np.concatenate((oldSol.Eventos_z['eventoMin'], zMin))
        nSteps = oldSol.nSteps + statusPy["nAccepted"] + statusPy["nRejected"]

    # Prepare the state arrays for interpolation
    tUnique, indexUnique = np.unique(tEf, return_index=True)
    yUnique = np.vstack((zEf, vEf))[:, indexUnique]
    aUnique = a[indexUnique]

    # Create cubic spline interpolators for the state variables
    solCppAll = CubicSpline(tUnique, yUnique, axis=1)

    # Create a cubic Hermite spline interpolator for the position variable
    solCppZ = CubicHermiteSpline(x=tUnique, y=yUnique[0], dydx=yUnique[1])
    solCppV = CubicHermiteSpline(x=tUnique, y=yUnique[1], dydx=aUnique)
    # Create a cubic spline for each sub-interval for the velocity variable
    # solCppV = PiecewiseCubicSpline(t=tUnique, y=yUnique[1], breakpoints=tAux)

    # Uses the Hermite spline for position and the cubic spline for velocity
    def solCppFull(t):
        t = np.atleast_1d(t)
        # y = solCppAll(t)
        y = np.zeros((2, len(t)))
        y[0] = solCppZ(t)
        y[1] = solCppV(t)
        return y
    
    solCppFull.t_min = tUnique[0]
    solCppFull.t_max = tUnique[-1]

    resultado = generalAux.DotDict(
        t=tUnique, z=yUnique[0], v=yUnique[1], a=aUnique, sol=solCppFull,
        Eventos=dict(eventoMax=tMaxEf, eventoMin=tMinEf),
        Eventos_z=dict(eventoMax=zMaxEf, eventoMin=zMinEf),
        Integrou=status[0], t_min=solCppFull.t_min, t_max=solCppFull.t_max,
        nSteps=nSteps, lastStatus=statusPy)

    return resultado