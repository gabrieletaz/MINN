from colorlog import warning
import numpy as np
from sklearn.model_selection import LeaveOneOut
import pandas as pd
from src.nn_model.amn_qp import *
from src.utils.clearml_utils import connect_confiuration, setting_up_task
from src.utils.hpo import hpo
from src.utils.import_data import *
from src.utils.import_GEM import *
from src.utils.loggers import DataFrameLogger
from src.utils.training import *
from src.utils.plots import *
from src.utils.utils import *

from clearml import Task, Logger
import hydra
from omegaconf import DictConfig
import warnings



@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    #clearML init
    # task = Task.init(project_name="e-muse/FBA_ML/tests_dynamic_weighing_loss", task_name=cfg.exp_name, reuse_last_task_id=False)
    task = setting_up_task(cfg)
    
    # # Converts the Hydra DictConfig to a native Python dict
    # cfg_dict = OmegaConf.to_container(cfg, resolve=True)  
    # # This will log all configuration parameters to ClearML
    # task.connect(cfg_dict, name='hydra_configuration')
    cfg = connect_confiuration(task, cfg)
    
    # add clearml tag for KOs
    if cfg.dataset.kos_genes == True:
        task.add_tags('KOs_genes')
    
    #add clearml tag for type of scheduler
    if "loss_weight_scheduler" in cfg.keys():
        task.add_tags(cfg.loss_weight_scheduler.type)
    
    # local or remote run
    assert cfg.execution in ["local", "remote"], "execution must be either local or remote"
    if cfg.execution == "remote":
        task.execute_remotely(queue_name=f"aai-gpu-01-2080ti:{cfg.gpu_id}-standard")
    elif cfg.execution == "local":
        cfg.model.device = cfg.model.device[:-1]+str(cfg.gpu_id)
    
    # reproducible
    seed = cfg.seed
    fix_random_seed(seed=seed)

    # gem and data
    n_distribution= cfg.gem.n_total_reactions
    X, y, Vin, fit_model, reference = load_ishii(seed=seed, 
                                                 dataset=cfg.dataset.dataset_name,
                                                 fluxes_removed_from_reference=cfg.dataset.fluxes_removed_from_reference,
                                                 fluxes_to_add_input = cfg.dataset.fluxes_to_add_input,
                                                 kos_genes= cfg.dataset.kos_genes,
                                                 gem = cfg.gem.gem_name)

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()

    # Lists to store results for each LOO iteration
    Q2_loo = []
    MAE_loo = []
    RMSE_loo = []
    NE_loo = []

    # List to store single left out predictions
    Vref_true_loo = []
    Vref_pred_loo = []
    Vpred_final = []
    Vin_reservoir_final = []

    hyper_params_with_metrics_list = []
    best_params_dict = {}
    
    loo_results_logger = DataFrameLogger(task)
    loo_losses_train_logger = DataFrameLogger(task)
    loo_losses_test_logger = DataFrameLogger(task)
    
    list_SV_loss_per_sample = []
    
    total_run= 29
    count = 1
    for train_idx, val_idx in loo.split(X):
        # Extract the training and validation data
        X_train, y_train, Vin_train = X[train_idx], y[train_idx], Vin[train_idx]
        X_val, y_val, Vin_val = X[val_idx], y[val_idx], Vin[val_idx]

        if cfg.hpo.hpo_name != 'no_hpo':
            if count==1:
                task.add_tags("HPO")
            
            best_params = hpo(cfg, task=task, X_train=X_train, y_train=y_train, Vin_train=Vin_train, n_distribution=n_distribution, fit_model=fit_model)
            if 'l1_constant' not in best_params.keys():
                best_params['l1_constant'] = 1
            best_params_dict[count] = best_params
        else:
            if not best_params_dict:
                best_params_dict = get_best_hyperparameters_from_previous_task(cfg.hpo.past_hpo_task_id)
            best_params = best_params_dict[count]
            
            if 'exclusions' in cfg.hpo.keys() and cfg.hpo.exclusions is not None and cfg.hpo.exclusions != "":
                if isinstance(cfg.hpo.exclusions, str):
                    cfg.hpo.exclusions = [cfg.hpo.exclusions]
                for exclusion in cfg.hpo.exclusions:
                    assert exclusion in best_params.keys(), f"exclusion {exclusion} not in best_params"
                    best_params.pop(exclusion, None)
        
        results = train_test_evaluation(cfg, task, X_train, X_val, y_train, y_val, Vin_train, Vin_val, n_distribution, fit_model, best_params, loo_count=count)
        

        # Inverse transform targets to extract metrics
        Vref_pred_te = np.matmul(np.array(results["test"]['Vref_pred']), fit_model.Pref.T)
        Vref_true = results["test"]['Vref_true']
        #Vref_true = sc_y.inverse_transform(Vref_true)c
        #Vref_pred_te = sc_y.inverse_transform(Vref_pred_te)
       
        
        print(f'Vref_pred: {Vref_pred_te}')
        print(f'Vref_true: {Vref_true}')

        
        # save metrics in the a list (one for each left out observation)
        Q2_value = r2_metric(np.array(Vref_true), Vref_pred_te)
        Q2_loo.append(Q2_value)

        MAE_value = np.mean(np.abs(Vref_true-Vref_pred_te))
        MAE_loo.append(MAE_value)

        RMSE_value = np.sqrt(np.mean(np.square(Vref_true-Vref_pred_te)))
        RMSE_loo.append(RMSE_value)

        NE_value = np.nan_to_num(np.linalg.norm(Vref_true - Vref_pred_te) / np.linalg.norm(Vref_true), posinf=0, neginf=0, nan=0)
        NE_loo.append(NE_value)

        Vref_true_loo.append(np.array(Vref_true).flatten())
        Vref_pred_loo.append(np.array(Vref_pred_te).flatten())
        Vpred_final.append(np.array(results["test"]['Vref_pred']).flatten())
        Vin_reservoir_final.append(np.array(results["test"]['Vin_reservoir']).flatten())
        

        print(f'run {count}/{total_run}')
        print(f'Loss Train: {results["train"]["losses"]}')
        print(f'Loss Test: {results["test"]["losses"]}')
        print(f'Q²:{Q2_value}')
        print(f'MAE:{MAE_value}')
        print(f'RMSE {RMSE_value}')
        print(f'NE:{NE_value}')
        
        list_SV_loss_per_sample.append(results["test"]["losses"][1])
        
        loo_results_logger.log_results({**{"loo_count": count},**{'Q2': Q2_value, 'MAE': MAE_value, 'RMSE': RMSE_value, 'NE': NE_value}})
        loo_losses_train_logger.log_results({**{"loo_count": count}, **dict(zip(['data_driven', 'stedy_state', 'upper_bound', 'flux_positivity'], results["train"]["losses"]))})
        loo_losses_test_logger.log_results({**{"loo_count": count}, **dict(zip(['data_driven', 'stedy_state', 'upper_bound', 'flux_positivity'], results["test"]["losses"]))})

        # add metric to hyperparams to upload an artifact on clearml and update each run
        best_params['MAE'] = MAE_value
        best_params['RMSE'] = RMSE_value
        best_params['NE'] = NE_value
        hyper_params_with_metrics_list.append(best_params)
        hyper_params_pandas = pd.DataFrame(hyper_params_with_metrics_list)
        task.upload_artifact('Pandas', artifact_object=hyper_params_pandas )
        count += 1
    
    # sade hyperparams in a pandas dataframe to ClearML
    task.upload_artifact('best_hyperparameters', best_params_dict)
    
    hyper_params_pandas = pd.DataFrame(hyper_params_with_metrics_list)
    # add and upload pandas.DataFrame (onetime snapshot of the object)
    task.upload_artifact('Pandas', artifact_object=hyper_params_pandas)
    
    loo_results_logger.report_as_table('results LOO samples')
    loo_results_logger.report_as_artifact('loo_results')
    
    loo_losses_train_logger.report_as_table('losses train LOO samples')
    loo_losses_train_logger.report_as_artifact('loo_losses train')
    
    loo_losses_test_logger.report_as_table('losses test LOO samples')
    loo_losses_test_logger.report_as_artifact('loo_losses test')

    # Calculate and store the mean Q2 for all LOO iterations
    #print(f'Mean Q² of LOOCV:{np.mean(Q2_loo)} ± {np.std(Q2_loo)}')
    #print(f'Mean MAE of LOOCV:{np.mean(MAE_loo)} ± {np.std(MAE_loo)}')
    #print(f'Mean RMSE of LOOCV:{np.mean(RMSE_loo)} ± {np.std(RMSE_loo)}')
    #print(f'Mean NE of LOOCV:{np.mean(NE_loo)} ± {np.std(NE_loo)}')

    # Report table - DataFrame with index
    if cfg.dataset.dataset_name=='ref_47_fluxes':
        df_true = pd.read_csv('data/ishii_data/fluxomics_iAF1260_reduced_split.csv')[cfg.dataset.metric_fluxes]
        
    elif cfg.dataset.dataset_name=='ref_47_fluxes_fit':
        df_true = pd.read_csv('data/ishii_data/fluxomics_iAF1260_reduced_split_fit.csv')[cfg.dataset.metric_fluxes]
    
    elif cfg.dataset.dataset_name=='fluxomics_iAF1260_reduced_split_filtered_iNF517': #TODO
        df_true = pd.read_csv('data/ishii_data/fluxomics_iAF1260_reduced_split_filtered_iNF517.csv')[cfg.dataset.metric_fluxes] #TODO
    
    elif cfg.dataset.dataset_name=='fluxomics_iNF517_reduced_split': #TODO
        df_true = pd.read_csv('data/ishii_data/fluxomics_iNF517_reduced_split.csv')[cfg.dataset.metric_fluxes] #TODO
        
    else:
        df_true = pd.read_csv('data/ishii_data/fluxomics_ecore_correct.csv')[cfg.dataset.metric_fluxes]

    Logger.current_logger().report_table(
        "df_true", 
        "df_true", 
        iteration=0, 
        table_plot=df_true
    )

    # Report table - DataFrame with index
    df_V = pd.DataFrame(Vpred_final, columns=fit_model.reactions)[cfg.dataset.metric_fluxes]
    Logger.current_logger().report_table(
        "df_pred", 
        "df_pred", 
        iteration=0, 
        table_plot=df_V
    )
    
    
    if cfg.model.model_reservoir == True:
        # Report table - DataFrame with index
        df_input_pFBA = pd.DataFrame(Vin_reservoir_final, columns=['R_EX_glc__D_e_rev', 'R_EX_o2_e_rev', 'R_EX_co2_e_fwd', 'R_EX_etoh_e', 'R_EX_ac_e'])
        Logger.current_logger().report_table(
            "df_input_pFBA", 
            "df_input_pFBA", 
            iteration=0, 
            table_plot=df_input_pFBA
        )


    metrics_df = metrics_table(df_true, df_V)
    #add row with SV avg and std
    avg_SV = np.array(list_SV_loss_per_sample).mean()
    std_SV = np.array(list_SV_loss_per_sample).std()
    metrics_df.loc["SV loss"] = [avg_SV, std_SV]
    
    Logger.current_logger().report_table(
        "metrics", 
        "Table", 
        iteration=0, 
        table_plot=metrics_df
    )

    plt = histogram_rmse_fluxes(df_true, df_V)
    task.logger.report_matplotlib_figure(title="RMSE_fluxes", 
                                        series= "RMSE_fluxes", 
                                        iteration=0, 
                                        figure=plt)

    task.close()



if __name__ == "__main__":
    main()