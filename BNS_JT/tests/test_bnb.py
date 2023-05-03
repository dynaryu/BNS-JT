"""
Ji-Eun Byun (ji-eun.byun@glasgow.ac.uk)
Created: 11 Apr 2023

Generalise Branch-and-Bound (BnB) operation to build CPMs
"""
import numpy as np
import pdb

from Trans import bnb_fns
from BNS_JT.branch import get_cmat, run_bnb
from BNS_JT.cpm import variable_elim, Cpm, get_prob


def test_bnb(main_bridge):

    ODs_prob_delay, ODs_prob_disconn, _, _, _, cpms_arcs, vars_arc = main_bridge

    # cpms_arc
    cpms_arc = {}
    for k, v in cpms_arcs.items():
        cpms_arc[k] = v
        cpms_arc[k].variables = [int(i) for i in v.variables]

    vars_arc = {int(k): v for k, v in vars_arc.items()}

    ## Problem
    #odInd = 1

    info = {'path': [[2], [3, 1]],
            'time': np.array([0.0901, 0.2401]),
            'arcs': np.array([1, 2, 3, 4, 5, 6])
            }

    max_state = 2
    comp_max_states = (max_state*np.ones_like(info['arcs'])).tolist()
    branches = run_bnb(sys_fn=bnb_fns.bnb_sys,
                       next_comp_fn=bnb_fns.bnb_next_comp,
                       next_state_fn=bnb_fns.bnb_next_state,
                       info=info,
                       comp_max_states=comp_max_states)

    [C_od, varis] = get_cmat(branches, info['arcs'], vars_arc, False)

    # Check if the results are correct
    # FIXME: index issue
    od_var_id = 7 - 1
    var_elim_order = list(range(1, 7))

    M_bnb = list(cpms_arc.values())[:10]
    M_bnb[od_var_id].C = C_od
    M_bnb[od_var_id].p = np.ones(shape=(C_od.shape[0],1))
    M_bnb_VE, vars_arc = variable_elim(M_bnb, var_elim_order, vars_arc)

    # FIXME: index issue
    disconn_state = 3 # max basic state
    disconn_prob = get_prob(M_bnb_VE, [7], np.array([disconn_state]), vars_arc)
    delay_prob = get_prob(M_bnb_VE, [7], np.array([1]), vars_arc, 0 )

    # Check if the results are the same
    # FIXME: index issue
    np.testing.assert_array_almost_equal(ODs_prob_delay[0], delay_prob)
    np.testing.assert_array_almost_equal(ODs_prob_disconn[0], disconn_prob)


