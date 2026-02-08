import importlib
import matlab.engine as matlab
import io

import time

import numpy as np, pandas as pd

import SMCsampler.IM_outputs as IMout
importlib.reload(IMout)

class SimIonMongerScan:
    """
    Simulation object running IonMonger with one or several scan rates
    """
    def __init__(self, runparams, runinputs):
        # store for cloning / multi-engine use
        self.runparams = runparams
        self.runinputs = runinputs

        # parameters for the simulation model
        self.IMfolder = runparams['IMfolder']          # folder with the IonMonger functions
        self.param_option = runparams['param_option']  # option for parameter.m in master.m
        self.timeout = runparams.get('timeout', 300)   # seconds
        self.scan_rate = runinputs['scan_rate']        # array with one or several scan rate values
        self.itypes = runinputs['input_types']         # Series with "lin" / "log" for each input
        self.iimap = runinputs['input_iter_map']       # Mapping between inputs and iter list
        self.xconst = runinputs['input_const']         # Constant inputs

        # start the matlab engine and point it to the IonMonger folder
        self.eng = matlab.start_matlab()
        self.eng.cd(self.IMfolder)

        # Be a good citizen
        self.eng.maxNumCompThreads(1, nargout=0)

    def start_matlab(self):
        # optional manual restart
        self.eng = matlab.start_matlab()
        self.eng.cd(self.IMfolder)

    def quit_matlab(self):
        if getattr(self, "eng", None) is not None:
            try:
                self.eng.quit()
            except Exception as e:
                print(f"Error quitting MATLAB engine: {e}")
            finally:
                self.eng = None

    def run(self, x_input, return_detailed_results=False, suppress_matlab_output=True):
        """
        Returns aggregated results; optionally also a DataFrame with detailed results.
        Uses MATLAB Engine async to enforce per-scan timeouts.
        """

        # initialise with nan values; +1 because index 0 holds the scan rate
        params_listxi = [np.nan for _ in range(len(self.iimap) + 1)]

        # set the input types (lin/log) for each element of x_input
        if np.size(self.itypes) == 1:
            input_types = pd.Series(self.itypes, index=self.iimap.keys())
        else:
            input_types = self.itypes.reindex(self.iimap.keys())

        # constants
        for vari in self.xconst.keys():
            idx = self.iimap.loc[vari] - 1
            if input_types.loc[vari] == "log":
                params_listxi[idx] = 10 ** float(self.xconst.loc[vari])
            else:
                params_listxi[idx] = self.xconst.loc[vari]

        # varying inputs
        for vari in x_input.keys():
            idx = self.iimap.loc[vari] - 1
            if input_types.loc[vari] == "log":
                params_listxi[idx] = 10 ** float(x_input.loc[vari])
            else:
                params_listxi[idx] = x_input.loc[vari]

        # Enforce constraints: vnH == vpH, tn == tp
        # if 'vnH' in x_input.index:
        #     idx_vpH = self.iimap.loc['vpH'] - 1
        #     idx_vnH = self.iimap.loc['vnH'] - 1
        #     params_listxi[idx_vpH] = params_listxi[idx_vnH]

        # if 'vnE' in x_input.index:
        #     idx_vpE = self.iimap.loc['vpE'] - 1
        #     idx_vnE = self.iimap.loc['vnE'] - 1
        #     params_listxi[idx_vpE] = params_listxi[idx_vnE]

        # if 'tn' in x_input.index:
        #     idx_tp = self.iimap.loc['tp'] - 1
        #     idx_tn = self.iimap.loc['tn'] - 1
        #     params_listxi[idx_tp] = params_listxi[idx_tn]

        s_list, d_list, y_list = [], [], []

        for iscan in range(len(self.scan_rate)):
            # first item in list is the scan rate (no 10** conversion)
            params_listxi[0] = self.scan_rate[iscan]
            params_list = params_listxi  # copy to avoid accidental mutation

            # Call MATLAB asynchronously to enable timeout
            if suppress_matlab_output:
                out = io.StringIO()
                future = self.eng.masterpy(params_list, self.param_option, stdout=out, stderr=out, background=True)
                #future = self.eng.masterpy(params_list, self.param_option, background=True)
            else:
                future = self.eng.masterpy(params_list, self.param_option, background=True)

            try:
                soli = future.result(timeout=self.timeout)
            except matlab.TimeoutError:
                future.cancel()
                #print(f"[TIMEOUT] IonMonger scan timed out for scan rate {self.scan_rate[iscan]}")
                raise TimeoutError(f"IonMonger simulation timed out for scan rate {self.scan_rate[iscan]}")
            except Exception as e:
                # Make sure we try to cancel on unexpected errors too
                try:
                    future.cancel()
                except Exception:
                    pass
                raise RuntimeError(f"IonMonger simulation failed for scan rate {self.scan_rate[iscan]}")

            # aggregate outputs
            sumresi = IMout.calculate_outputs(soli['J'], soli['V'])
            sumresi["scan_rate"] = self.scan_rate[iscan]

            if iscan == 0:
                sumres = sumresi
            else:
                sumres = pd.concat((sumres, sumresi))

            s_list.append(self.scan_rate[iscan])
            d_list.append(soli)
            y_list.append(sumresi.set_index(["variable"]))

        sumres = sumres.set_index(["scan_rate", "variable"])
        #print(sumres)

        dr_df = pd.DataFrame({
            'scan_rate': s_list,
            'detres': d_list,
            'mainres': y_list
        })

        if return_detailed_results:
            return sumres, dr_df
        return sumres
    

class SimIonMongerImp:
    """
    Simulation object running IonMonger in impedance mode.
    """

    def __init__(self, runparams, runinputs):
        # store for cloning / multi-engine use
        self.runparams = runparams
        self.runinputs = runinputs
        
        # parameters for the simulation model
        self.IMfolder = runparams['IMfolder']           # folder with the IonMonger functions
        self.param_option = runparams['param_option']   # option for parameter.m in master.m
        self.timeout = runparams.get('timeout', 300)    # seconds
        self.itypes = runinputs['input_types']          # Series with "lin" / "log" for each input
        self.iimap = runinputs['input_iter_map']        # Mapping between inputs and iter list
        self.xconst = runinputs['input_const']          # Constant inputs

        # start the matlab engine and point it to the IonMonger folder
        self.eng = matlab.start_matlab()
        self.eng.cd(self.IMfolder)

        # Be a good citizen
        self.eng.maxNumCompThreads(1, nargout=0)

    def start_matlab(self):
        # optional manual restart
        self.eng = matlab.start_matlab()
        self.eng.cd(self.IMfolder)

    def quit_matlab(self):
        if getattr(self, "eng", None) is not None:
            try:
                self.eng.quit()
            except Exception as e:
                print(f"Error quitting MATLAB engine: {e}")
            finally:
                self.eng = None

    def run(self, x_input, return_detailed_results=True, suppress_matlab_output=True):
        """
        Run IonMonger impedance model.

        - Uses MATLAB Engine async to enforce a timeout per call.
        - Returns a Series with MultiIndex (Z, freq): Z ∈ {'R','X'}.
        - If return_detailed_results is True, also returns the raw MATLAB output dict.
        """
        # initialise with nan values; +1 because index 0 is reserved for freqs
        params_listxi = [np.nan for _ in range(len(self.iimap) + 1)]

        # set the input types (lin/log) for each element
        if np.size(self.itypes) == 1:
            input_types = pd.Series(self.itypes, index=self.iimap.keys())
        else:
            input_types = self.itypes.reindex(self.iimap.keys())

        # constants
        for vari in self.xconst.keys():
            idx = self.iimap.loc[vari] - 1
            if input_types.loc[vari] == "log":
                params_listxi[idx] = 10.0 ** float(self.xconst.loc[vari])
            else:
                params_listxi[idx] = self.xconst.loc[vari]

        # varying inputs
        for vari in x_input.keys():
            idx = self.iimap.loc[vari] - 1
            if input_types.loc[vari] == "log":
                params_listxi[idx] = 10.0 ** float(x_input.loc[vari])
            else:
                params_listxi[idx] = x_input.loc[vari]

        params_list = params_listxi  # copy to avoid mutation

        # Call MATLAB asynchronously to enable timeout
        if suppress_matlab_output:
            out = io.StringIO()
            future = self.eng.masterpy(params_list, self.param_option, stdout=out, stderr=out, background=True)
        else:
            future = self.eng.masterpy(params_list, self.param_option, background=True)

        t_imp_start = time.time()

        try:
            soli = future.result(timeout=self.timeout)
        except matlab.TimeoutError:
            t_imp_end = time.time()
            future.cancel()
            #print("[TIMEOUT] IonMonger impedance simulation timed out")
            raise TimeoutError("IonMonger impedance simulation timed out")
        except Exception as e:
            t_imp_end = time.time()
            try:
                future.cancel()
            except Exception:
                pass
            raise RuntimeError(f"IonMonger impedance simulation failed.")
        
        t_imp_end = time.time()
        #print(f"[INFO] Impedance simulation took {t_imp_end - t_imp_start:.2f} s", flush=True,)

        # impedance summary: Re(Z) and Im(Z) vs frequency
        freqs = soli["freqs"][0]
        R = np.asarray(soli["R"]).ravel()
        X = np.asarray(soli["X"]).ravel()

        yr = pd.DataFrame({"Z": "R", "freq": freqs, "value": R}).set_index(["Z", "freq"])
        yi = pd.DataFrame({"Z": "X", "freq": freqs, "value": X}).set_index(["Z", "freq"])

        # combined Series with MultiIndex (Z, freq)
        sumres = pd.concat([yr, yi])["value"]

        if return_detailed_results:
            return sumres, soli
        return sumres
