from pydoxtools.settings import settings
import optuna
import optuna.visualization as ov
import plotly

optuna.__version__

study_name = "hyparams_50_inf_ep2"

#storage_url = f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite"
storage_url = f"sqlite:////home/tom/Nextcloud/pydoxtools_training/study.sqlite"
#local_storage=f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite"
remote_storage="TODO: get from env variable (f"mysql+pymysql:....")"
remote_storage

try:
    optuna.copy_study(
        from_study_name=study_name,
        from_storage=remote_storage,
        to_storage=storage_url
    )
except optuna.exceptions.DuplicatedStudyError:
    pass


study = optuna.load_study(
    study_name=study_name,
    storage=remote_storage
)


# +
#study.trials
# -

try:
    study.best_trial.number, study.best_trial.params
except RuntimeError:
    is_multi_objective=True
    print("probably a multi-objective study, can no get best parameters")

if False:
    clean_trials = [t for t in study.trials if t.values]
    clean_trials = [t for t in clean_trials if t.values[0]!=-2]
    #for t in clean_trials:
    #    print(t.number, t.state, t.system_attrs.get('constraints'), t.system_attrs.get('nsga2:generation'), t.values)

    # load new study
    def constraints(trial: optuna.Trial):
        return trial.user_attrs["constraint"]


    sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
    study = optuna.create_study(
    study_name=study_name,
    storage=optuna.storages.RDBStorage(
            url=remote_storage,
            # engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
            # add heartbeat i order to automatically mark "crashed" trials
            # as "failed" so that they can be repeated
            heartbeat_interval=60 * 5,
            grace_period=60 * 21,
        ),
        sampler=sampler,
        load_if_exists=True,
        directions=["maximize", "minimize"]
    )
    # copy filtered trials into study
    study.add_trials(clean_trials)
    #study.

if True:
    for t in study.trials:
        print(t.number, t.state, t.system_attrs.get('constraints'), t.system_attrs.get('nsga2:generation'), t.values)

if not is_multi_objective: # single objective
    # import optuna.visualization.matplotlib as ov
    if ov.is_available():
        figs = {
            "intermediate.html": ov.plot_intermediate_values(study),
            # optuna.visualization.plot_pareto_front(study)
            # ov.plot_contour(study, params=study.best_params.keys()).write_html("contour.html")
            "param_importances.html": ov.plot_param_importances(study),
            "parallel_coordinate.html": ov.plot_parallel_coordinate(study, params=study.best_params.keys()),
            "optimization_history.html": ov.plot_optimization_history(study),
            "slice.html": ov.plot_slice(study),
            "edf.html": ov.plot_edf(study)
        }
        for key, fig in figs.items():
            fig.show()
            #fig.write_html()
else:
        target=lambda t: t.values[0]
        target_name="address.f1-score"
        #target_name="network size [mb]"
        figs = {
            #"intermediate.html": ov.plot_intermediate_values(study),#target=target, target_name=target_name),
            "pareto_front": optuna.visualization.plot_pareto_front(study),
            "param_importances.html": ov.plot_param_importances(study, target=target, target_name=target_name),
            "parallel_coordinate.html": ov.plot_parallel_coordinate(study, target=target, target_name=target_name),
            "optimization_history.html": ov.plot_optimization_history(study,target=target, target_name=target_name),
            "slice.html": ov.plot_slice(study,target=target, target_name=target_name),
            "edf.html": ov.plot_edf(study,target=target, target_name=target_name)
        }
        for key, fig in figs.items():
            fig.show()
            #fig.write_html()
# +
ov.plot_contour(study, target=target, target_name=target_name).write_html("/tmp/contour.html")

import subprocess
subprocess.run(f"google-chrome /tmp/contour.html",shell=True)
# +
ov.plot_contour(study, params=study.best_params.keys()).write_html("/tmp/contour.html")

import subprocess
subprocess.run(f"google-chrome /tmp/contour.html",shell=True)