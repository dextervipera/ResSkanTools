import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
tqdm_notebook = tqdm

def CalcPnt(datain: pd.DataFrame, offset_correction = False, slope_correction = False, plotAll = False):
    ans = {}

    #datain = pnt

    datain.dT = datain.T_Probe - datain.T_Sample
    datain.dT.replace(to_replace=0, value=0.001)
    # datain.Alpha = datain.U_Seeb / datain.dT

    if offset_correction:
        if offset_correction == "U_Seeb":
            datain.U_Seeb -= datain.U_Seeb.mean()
    if slope_correction:
        pass

    Imax = max(datain.Current)
    Imin = min(datain.Current)
    I0 = (Imax + Imin) / 2
    Irng = Imax - Imin

    rng_PInv = max(I0 + Irng / 3, 15)
    rng_NInv = min(I0 - Irng / 3, -15)

    rng_SeebH = max(I0 + Irng / 10, 10)
    rng_SeebL = min(I0 - Irng / 10, -10)

    if Irng > 20:
        PInv = datain.loc[datain["Current"] > rng_PInv]
        NInv = datain.loc[datain["Current"] < rng_NInv]
    else:
        PInv = pd.DataFrame(columns=datain.columns)
        NInv = pd.DataFrame(columns=datain.columns)


    Seeb = datain.loc[(datain["Current"] > rng_SeebL) & (datain["Current"] < rng_SeebH)]
    ans['U_Seeb'] = Seeb.U_Seeb.mean()
    ans['U_Seeb_stdDev'] = Seeb.U_Seeb.std()

    pd.options.mode.chained_assignment = None
    PInv.loc[:, "U_Seeb"] = PInv.U_Seeb-Seeb.U_Seeb.mean()
    NInv.loc[:, "U_Seeb"] = NInv.U_Seeb-Seeb.U_Seeb.mean()

    ans['U_pRes'] = PInv.U_Seeb.mean()
    ans['U_nRes'] = NInv.U_Seeb.mean()
    ans['I_pRes'] = PInv.Current.mean()
    ans['I_nRes'] = NInv.Current.mean()

    ans['U_pRes_stdDev'] = PInv.U_Seeb.std()
    ans['U_nRes_stdDev'] = NInv.U_Seeb.std()
    ans['I_pRes_stdDev'] = PInv.Current.std()
    ans['I_nRes_stdDev'] = NInv.Current.std()

    PInv.loc[:, "Resistance"] = PInv.U_Seeb / PInv.Current
    NInv.loc[:, "Resistance"] = NInv.U_Seeb / NInv.Current

    constant_header = ['ID', 'TIME_ABSOLUTE', 'TIME_ELAPSED', 'X', 'Y', "Z"]
    for header in constant_header:
        ans[header] = Seeb[header][0]

    all_header = ['T_Probe', 'T_Sample', 'Force', 'T01', 'T02', 'T03', 'probe.heater.power', 'device.heater.power']
    for header in all_header:
        ans[header] = datain[header].mean()

    resistance_header = ['U_current', 'Current', "Resistance"]
    for header in resistance_header:
        ans[header] = PInv[header].mean() + NInv[header].mean() / 2


    pd.options.mode.chained_assignment = 'warn'

    ans['dT'] = ans["T_Probe"]-ans["T_Sample"]
    ans["Alpha"] = ans["U_Seeb"]/ans["dT"]*10e3
    ans["Alpha_stdDev"] = ans["U_Seeb_stdDev"]/ans["dT"]*10e6
    ans["Resistance_stdDev"] = (abs(PInv.Resistance.std()) + abs(NInv.Resistance.std())) / 2
    
    if plotAll:
        print(f"Imax:{Imax:.2f}, Imin:{Imin:.2f}, I0:{I0:.2f}, Irng:{Irng:.1f}]")
        print(f'Mean Seebeeck voltage:{ans["U_Seeb"]} uV')
        print(f"NInv.count:{len(NInv)}, P0Inv.count:{len(PInv)}, Seeb.Count:{len(Seeb)}")
        
        pntplt = plt.figure()
        pntSerie = pntplt.add_axes([.1,.1,.8,.4])
        curSerie = pntplt.add_axes([.1,.6,.8,.4])
        curSeebSerie = curSerie.twinx()
        
        USeebSerie = pntplt.add_axes([1.3, .1, .8, .4])
        dTSerie = pntplt.add_axes([1.3, .6, .8, .4])
        
        curSerie.plot(datain.Current, 'bo', label="current [mA]")
        curSerie.set_xticks([])
        curSerie.set_title("Current flow")
        curSerie.legend(loc="upper left")
        
        curSeebSerie.plot(datain.U_Seeb*1000, 'y+', label="Sample Voltage [mV]")
        curSeebSerie.set_ylabel("U Sample [uV]")
        curSeebSerie.legend(loc="upper right")

        pntSerie.plot(PInv.U_Seeb, 'r+', label="Not inverted")
        pntSerie.plot(NInv.U_Seeb, 'g+', label="Inverted")
        pntSerie.set_xticks([])
        pntSerie.set_title('Meas segregation')
        pntSerie.legend(loc="upper left")
        pntSerie.set_ylabel("Voltage [mV]")

        USeebSerie.set_title("U Seeb")
        USeebSerie.plot(Seeb.U_Seeb*10e3, 'yo')
        USeebSerie.set_xticks([])
        USeebSerie.set_ylabel("U Seeb [uV]")

        dTSerie.set_title("dT [K]")
        dTSerie.plot(datain.dT, 'orange')
        dTSerie.set_xticks([])
    
    return ans


def agregate_data(data: pd.DataFrame):
    means_header = ['X', 'Y', 'Z', 'U_current', 'Current',
    'U_Seeb', 'U_Seeb_StdDev', 'T_Probe', 'T_Sample', 'dT', 'Alpha',
    'Alpha_stdDev', 'Resistance', 'Resistance_stdDev', 'Force', 'T01',
    'T02', 'T03', 'U_pRes', 'U_nRes', 'I_pRes', 'I_nRes',
    'probe.heater.power', 'device.heater.power']

    #data = dout

    xticks = data.X.unique()
    yticks = data.Y.unique()

    xlen = len(xticks)
    ylen = len(yticks)

    xmap = {xval: xid for xid, xval in enumerate(xticks)}
    ymap = {yval: yid for yid, yval in enumerate(yticks)}

    coords = []
    for y in yticks:
        for x in xticks:
            coords.append({'x':x,'y':y})

    zdata = np.empty([xlen, ylen])
    zdata[:] = np.NaN

    xs = xticks[0]
    ys = yticks[0]

    agregated = pd.DataFrame(columns=data.columns)

    for coord in tqdm_notebook(coords, desc="agregating data"):
        dsel = data.loc[(data.X == coord['x']) & (data.Y == coord['y'])]
        if len(dsel):
            out = {}
            out={"ID": dsel["ID"].iloc[0], 
               "TIME_ABSOLUTE": dsel["TIME_ABSOLUTE"].iloc[0],
               "TIME_ELAPSED": dsel["TIME_ELAPSED"].iloc[0],
               }
            for mean_param in means_header:
                out[mean_param] = dsel[mean_param].quantile()
                out[f"{mean_param}_sstdDev"] = dsel[mean_param].std()
            dd = pd.DataFrame([out.values()], columns=out.keys())
            agregated = pd.concat([agregated, dd])

    return agregated

def XYMapPlot(data: pd.DataFrame, parameter: str, err_parameter = None, **kv):
    
    xticks = data.X.unique()
    yticks = data.Y.unique()
    
    xlen = len(xticks)
    ylen = len(yticks)
    
    xmap = {xval: xid for xid, xval in enumerate(xticks)}
    ymap = {yval: yid for yid, yval in enumerate(yticks)}
    
    zdata = np.empty([xlen, ylen])
    zdata[:] = np.NaN
    
    xdata = np.empty([xlen, ylen])
    xdata[:] = 0
    
    ydata = np.empty([xlen, ylen])
    ydata[:] = 0
    
    xx0 = min(xticks)
    yy0 = min(yticks)
    
    for yy in yticks:
        for xx in xticks:
            xdata[xmap[xx], ymap[yy]] = xx - xx0
            ydata[xmap[xx], ymap[yy]] = yy - yy0
    
    # offset
    param_offset = 0
    if kv.get("offset", None) == "move_to_zero":
        param_offset = data[parameter].min()
    
    for index, elm in data.iterrows():
        zdata[xmap[elm.X], ymap[elm.Y]] = elm[parameter]-param_offset
    
    # plotting
    resFig = plt.figure()
    resPlt = resFig.add_axes([0.1,.1,.8,.8])
    
    msh = resPlt.pcolormesh(xdata, ydata, zdata)
    resFig.colorbar(msh, ax=resPlt)
    resPlt.set_aspect('equal')
    resPlt.set_title(parameter)
    resPlt.set_xlabel('X [mm]')
    resPlt.set_ylabel('Y [mm]')
    
    
    
    if err_parameter:
        errPlt = resFig.add_axes([1.1,.1,.8,.8]) 
        
        zerr = np.empty([xlen, ylen])
        zerr[:] = np.NaN
        
        for index, elm in data.iterrows():
            zerr[xmap[elm.X], ymap[elm.Y]] = elm[err_parameter]
        
        msh_err = errPlt.pcolormesh(xdata, ydata, zerr)
        errPlt.set_aspect('equal')
        errPlt.set_title(err_parameter)
        errPlt.set_xlabel('X [mm]')
        errPlt.set_ylabel('Y [mm]')