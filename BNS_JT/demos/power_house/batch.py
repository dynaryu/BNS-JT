from pathlib import Path
import numpy as np
import random
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
import typer

from BNS_JT import cpm, variable, quant

HOME = Path(__file__).parent

app = typer.Typer()

@app.command()
def get_lins( pole1, paths ):
    """
    Get link sets of a pole's end (to be connected to substations) given paths information
    [INPUT]
    - pole1: a string 
    - paths: a dictionary of substation: LIST of paths
    [OUTPUT]
    - lins: a list of link-sets

    [EXAMPLE]
    lins = get_lins( 'p1',  {'s0': [['p0', 'p1'], ['p2', 'p3', 'p5']], 's1':[['p4','p5'], ['p6', 'p7']]})
    """

    lins = []
    for s, ps in paths.items():
        for p in ps:
            if pole1 in p:
                idx = p.index(pole1)
                lin1 = [s] + p[:(idx+1)]
                lins += [lin1]
    
    return lins

def quant_cpms( haz, pfs, rep_pri, lins ):

    """
    Quantify CPMs from user inputs
    """


    cpms={}
    vars={}

    ## Hazard
    vals_h, p_h = [], []
    for s, p in haz.items():
        vals_h.append( s )
        p_h.append( p )
    vars['haz'] = variable.Variable( name='haz', values=vals_h) # values=(mild, medium, intense)
    cpms['haz'] = cpm.Cpm( variables=[vars['haz']], no_child=1, C=np.array(range(len(p_h))), p=p_h )

    ## Structures
    C_x = []
    for hs in range(len(vals_h)):
        Cs += [[0, hs], [1, hs]]
    C_x = np.array(Cs)

    for s, pf in pfs.items():
        name = 'xs' + str(i)
        vars[name] = variable.Variable( name=name, values=['fail','surv'] ) # values=(failure, survival)

        p_x = []
        for p in pf:
            p_x += [p, 1-p]

        cpms[name] = cpm.Cpm( variables = [vars[name], vars['haz']], no_child = 1,
                        C=C_x, p=p_x )
        
    ## Number of damaged structures so far (following repair priority)
    for i, s in enumerate(rep_pri):
        name = 'n' + s
        vars[name] = variable.Variable( name=name, values=list(range(i+2)) )

        if i < 1: # first element
            cpms[name] = cpm.Cpm( variables = [vars[name], vars['x'+s]], no_child = 1, C=np.array([[1,0], [0, 1]]), p=np.array([1,1]) )
        else:
            t_old_vars = vars[n_old].values

            Cx = np.empty(shape=(0,3), dtype=int)
            for v in t_old_vars:
                Cx_new = [[v, 1, v], [v+1, 0, v]]
                Cx = np.vstack( [Cx, Cx_new] )

            cpms[name] = cpm.Cpm( variables = [vars[name], vars['x'+s], vars[n_old]], no_child = 1, C=Cx, p=np.ones(shape=(2*len(t_old_vars)), dtype=np.float32) )

        n_old = copy.deepcopy(name)

    ## Closure time
    for s in rep_pri:
        name = 'c'+s
        name_n = 'n'+s
        vals = vars[name_n].values
        vars[name] = variable.Variable( name=name, values=vals )

        cst_all = vars[name_n].B.index(set(vals)) # composite state representing all states
        Cx = np.array([[0, 1, cst_all]])
        for v in vals:
            if v > 0:
                Cx = np.vstack((Cx, [v, 0, v]))

        cpms[name] = cpm.Cpm( variables = [vars[name], vars['x'+s], vars[name_n]], no_child = 1, C=Cx, p=np.ones(shape=(len(Cx),1), dtype=np.float32) )

    ## Power-cut days of houses
    for h, sets in lins.items():
        if len(sets) == 1:              
            vars_h = [vars[x] for x in sets[0]]

            cpms[h], vars[h] = quant.sys_max_val( h, vars_h )
            
        else:
            names_hs = [h+str(i) for i in range(len(sets))]
            for h_i, s_i in zip(names_hs, sets):
                vars_h_i = [vars[x] for x in s_i]

                cpms[h_i], vars[h_i] = quant.sys_max_val( h_i, vars_h_i )
                
            vars_hs = [vars[n] for n in names_hs]
            cpms[h], vars[h] = quant.sys_min_val( h, vars_hs )

    return cpms, vars

def get_pole_conn(paths):

    poles = []
    for _, ps in paths.items():
        for p1 in ps:
            poles += p1
    poles = set(poles)

    conn = {}
    for pl in poles:
        for sub, path in paths.items():
            for path1 in path:
                if pl in path1:
                    pl_idx = path1.index(pl)
                    if pl_idx < 1:
                        pl_tail = sub
                    else:
                        pl_tail_pl = path1[pl_idx-1]

                        for h, pls_h in houses.itmes():
                            if pl_tail_pl in pls_h:
                                pl_tail = h
                                break # a pole is connected to only one house
                    break # a pole appears only in only one path
            
        for h, pls_h in houses.items():
            if pl in pls_h:
                pl_head = h
                break # a pole's head is connected to exactly one house
        conn[pl] = (pl_tail, pl_head)

    return conn

def add_pole_loc( locs_sub_hou, conn_pol ):

    locs = copy.deepcopy(locs_sub_hou)

    for pl, pair in conn_pol.items():
        locs[pl] = tuple( [0.5*sum(tup) for tup in zip(locs_sub_hou[pair[0]], locs_sub_hou[pair[1]])] )

    return locs

def plot_result(locs, conn, avg_cut_days, pfs_mar):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # draw edges
    for _, pair in conn.items():
        loc0, loc1 = locs[pair[0]], locs[pair[1]]
        plt.plot([loc0[0], loc1[0]], [loc0[1], loc1[1]], color='grey')

    # draw houses
    plt.scatter([locs[h][0] for h in houses.keys()], [locs[h][0] for h in houses.keys()], [avg_cut_days[h] for h in houses.keys()], cmap='Reds', s=200, marker="X")
    cb1 = plt.colorbar()
    cb1.ax.set_xlabel('Avg. \n cut days', size = 15)

    # draw structures    
    plt.scatter( [locs[s][0] for s in pfs_mar.keys()], [locs[s][1] for s in pfs_mar.keys()], [pfs_mar[s] for s in pfs_mar.keys()], cmap='Reds' )
    cb2 = plt.colorbar()
    cb2.ax.set_xlabel('Fail. \n prob.', size = 15)

    # texts
    tx, ty = 0.10, 0.01
    for x in list(houses.keys()) + list(pfs_mar.keys()):
        ax.text(locs[x][0]+tx, locs[x][1]+ty, x)

    ax.set_xticks([])
    ax.set_yticks([])
    
    fig.savefig( 'result.png', dpi=200 )
    print('result.png is created.') 

@app.command()
def main(h_list, plot: bool=False):

    ######### USER INFORMATION ###################
    # network topology
    ## Topology is assumed radial--Define each substation and connected paths (in the order of connectivity)
    paths = {'s0': [['p0', 'p1'], ['p2', 'p3', 'p5']], 's1':[['p4','p5'], ['p6', 'p7']]}
    ## House--connected poles (NB the edges are uni-directional in the direction away from substation; all poles must be connected to a house at its head)
    houses = {'h0':['p0'], 'h1':['p1'], 'h2':['p2'], 'h3':['p3', 'p4'], 'h4':['p5'], 'h5':['p6'], 'h6':['p7']}

    # Random variables
    haz ={'mild': 0.5, 'medi': 0.2, 'inte': 0.3 } # {scenario: prob}
    pfs = {'s0': [0.001, 0.01, 0.1], 's1':[0.005, 0.05, 0.2], # {structure: failure probability for each hazard scenario}
          'p0': [0.001, 0.01, 0.1], 'p1':[0.005, 0.05, 0.2],
          'p2': [0.001, 0.01, 0.1], 'p3':[0.005, 0.05, 0.2],
          'p4': [0.001, 0.01, 0.1], 'p5':[0.005, 0.05, 0.2],
          'p6': [0.001, 0.01, 0.1], 'p7':[0.005, 0.05, 0.2]} 
    
    # Repair priority
    rep_pri = ['s0', 's1', 'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'] 
    ###############################################

    ## Compute link-sets from paths and houses
    lins = {}
    for h, pls in houses.items():
        lins_h = []
        for pl in pls:
            lins_pl = get_lins( pl, paths )
            lins_h += lins_pl
        
        lins[h] = lins_h

    # CPMs
    cpms, vars = quant_cpms( haz, pfs, rep_pri, lins )

    # Inference
    ## Variable Elimination order
    VE_ord = ['haz']
    for s in rep_pri:
        VE_ord += ['x'+s, 'n'+s, 'c'+s]
    for h, paths in lins.items(): # Actually, VE order of houses does not matter as the marginal distribution is computed for "one" house at a time
        if len(paths) < 2:
            VE_ord += [h]
        else: # there are more than 1 path
            VE_ord += [h+str(i) for i in range(len(paths))] + [h]

    ## Get P(H*) for H* in h_list
    cond_names = ['haz']

    Mhs = {}
    for h in h_list:
        st=time.time()
        Mhs[h] = cpm.cal_Msys_by_cond_VE( cpms, vars, cond_names, VE_ord, h )
        en = time.time()

        print( h + " done. " + "Took {:.2f} sec.".format(en-st) )

    avg_cut_days = {}
    for h in h_list():
        days = cpm.get_means( Mhs[h], [h] )
        avg_cut_days[h] = days[0]

    # Plot
    if plot:
        ############### USER INPUT ##############
        # Locations for mapping
        locs = {"h0": (-0.8, -1.0), "h1": (-2*0.8, -2*1.0), "h2": (0.8, -1.0), "h3": (2*0.8, -2*1.0), "h4": (0.8, -3*1.0), "h5": (5*0.8,-2*1.0), "h6": (6*0.8, -3*1.0)}
        locs["s0"], locs["s1"] = (0.0, 0.0), (3.5*0.8, 0.0)
        #########################################
        
        # Get information about paths
        conn = get_pole_conn( paths )
        locs = add_pole_loc( locs, conn )

        # failure prob. from marginal distribution
        pfs_mar = {} 
        for s in pfs.keys():
            Mx = cpm.cal_Msys_by_cond_VE( cpms, vars, cond_names, VE_ord, 'x'+s )
            pf = cpm.get_prob(Mx, ['x'+s], [0])
            pfs_mar[s] = pf

        plot_result(locs, conn, avg_cut_days, pfs_mar)  

    return Mhs, avg_cut_days



