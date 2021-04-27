import time_series_model as model

plot_graphs = False

def run_experiment1():
    experiment1 = model.Experiment(media='M63_Gly',
                               solute='Sucrose',
                               temperature='37',
                               date='2020-09-10',
                               folder='Data/20200910_m63gly_37c_Sucrose',
                               plot=plot_graphs)
    return experiment1


def run_experiment2():
    experiment2 = model.Experiment(media='M63_Glu_CAA',
                               solute='Sucrose',
                               temperature='30',
                               date='2020-09-29',
                               folder='Data/20200929_m63GluCAA_30c_Sucrose',
                               plot=plot_graphs)
    return experiment2

def run_experiment3():
    experiment3 = model.Experiment(media='M63_Glu_CAA',
                               solute='Sucrose',
                               temperature='37',
                               date='2020-10-02',
                               folder='Data/20201002_m63GluCAA_37C_Sucrose',
                               plot=plot_graphs)
    return experiment3


def run_experiment4():
    experiment4 = model.Experiment(media='M63_Gly_Betaine',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-10-06',
                                folder='Data/20201006_m63Gly_Betaine_37C_Sucrose',
                                plot=plot_graphs)
    return experiment4

def run_experiment5():
    experiment5 = model.Experiment(media='M63_Glu_CAA_Betaine',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-10-09',
                                folder='Data/20201009_m63GluCAA_Betaine_37C_Sucrose',
                                plot=plot_graphs)
    return experiment5

def run_experiment6():
    experiment6 = model.Experiment(media='M63_Glu_CAA',
                                solute='NaCl',
                                temperature='37',
                                date='2020-10-13',
                                folder='Data/20201013_m63GluCAA_37C_NaCl',
                                plot=plot_graphs)
    return experiment6

def run_experiment7():
    experiment7 = model.Experiment(media='M63_Glu_CAA_Betaine',
                                solute='NaCl',
                                temperature='37',
                                date='2020-10-19',
                                folder='Data/20201019_m63GluCAA_Betaine_37C_NaCl',
                                plot=plot_graphs)
    return experiment7

def run_experiment8():
    experiment8 = model.Experiment(media='M63_Glu',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-10-23',
                                folder='Data/20201023_m63Glu_37C_Sucrose',
                                plot=plot_graphs)
    return experiment8

def run_experiment9():
    experiment9 = model.Experiment(media='M63_Glu',
                               solute='Sucrose',
                               temperature='42',
                               date='2020-10-27',
                               folder='Data/20201027_m63Glu_42C_Sucrose',
                               plot=plot_graphs)
    return experiment9

def run_experiment10():
    experiment10 = model.Experiment(media='M63_Man',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-10-27',
                                folder='Data/20201030_m63Man_37C_Sucrose',
                                plot=plot_graphs)
    return experiment10

def run_experiment11():
    experiment11 = model.Experiment(media='RDM',
                                solute='Sucrose',
                                temperature='37',
                                date='2020-12-04',
                                folder='Data/20201204_RDM_37C_Sucrose',
                                plot=plot_graphs)
    return experiment11

def run_experiment12():
    experiment12 = model.Experiment(media='M63_Gly',
                                solute='NaCl',
                                temperature='37',
                                date='2021-01-19',
                                folder='Data/20210119_m63Gly_37C_NaCl',
                                plot=plot_graphs)
    return experiment12

def run_experiment13():
    experiment13 = model.Experiment(media='M63_Gly_Betaine',
                                solute='NaCl',
                                temperature='37',
                                date='2021-01-25',
                                folder='Data/20210125_m63Gly_Betaine_37C_NaCl',
                                plot=plot_graphs)
    return experiment13

def run_experiment14():
    experiment14 = model.Experiment(media='M63_Gly',
                                solute='NaCl',
                                temperature='37',
                                date='2021-02-10',
                                folder='Data/20210202_m63Gly_37C_NaCl',
                                plot=plot_graphs)
    return experiment14

def run_experiment15():
    experiment15 = model.Experiment(media='M63_Glu',
                                solute='NaCl',
                                temperature='37',
                                date='2021-02-16',
                                folder='Data/20210216_m63Glu_37C_NaCl',
                                plot=plot_graphs)
    return experiment15

def run_experiment16():
    run_experiment16 = model.Experiment(media='M63_Glu_Betaine',
                                solute='NaCl',
                                temperature='37',
                                date='2021-02-24',
                                folder='Data/20210224_m63Glu_Betaine_37C_NaCl',
                                plot=plot_graphs)
    return run_experiment16

def run_experiment17():
    run_experiment17 = model.Experiment(media='M63_Glu',
                                solute='NaCl',
                                temperature='37',
                                date='2021-03-12',
                                folder='Data/20210312_m63Glu_37C_NaCl',
                                plot=plot_graphs,
                                label = '100-600')
    return run_experiment17

def run_experiment18():
    run_experiment18 = model.Experiment(media='M63_Gly',
                                solute='NaCl',
                                temperature='37',
                                date='2021-03-29',
                                folder='Data/20210329_m63Gly_37C_NaCl',
                                plot=plot_graphs,
                                label = '100-600')
    return run_experiment18

def run_experiment19():
    run_experiment19 = model.Experiment(media='M63_Gly',
                                solute='NaCl',
                                temperature='37',
                                date='2021-04-06',
                                folder='Data/20210406_m63Gly_37_NaCl',
                                plot=plot_graphs,
                                label = '100-600')
    return run_experiment19