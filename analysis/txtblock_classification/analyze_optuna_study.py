from pydoxtools.settings import settings
import optuna
import optuna.visualization as ov
import plotly

optuna.__version__

#storage_url = f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite"
storage_url = f"sqlite:////home/tom/Nextcloud/pydoxtools_training/study.sqlite"
#local_storage=f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite"
remote_storage="TODO: get from env variable (f"mysql+pymysql:....")"
remote_storage

study = optuna.load_study(
    study_name="find_more_data_generation_parameters",
    storage=storage_url
)


optuna.copy_study(
    from_study_name="test",
    from_storage=remote_storage,
    to_storage=storage_url,
)


study.best_params

# +

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
# +
ov.plot_contour(study, params=study.best_params.keys()).write_html("/tmp/contour.html")

import subprocess
subprocess.run(f"google-chrome /tmp/contour.html",shell=True)
# -



