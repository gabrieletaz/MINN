import matplotlib.pyplot as plt
import numpy as np
import torch
from clearml import Task, Logger
from scipy.stats import linregress
import pandas as pd




def R_squared(observed, predicted, mode='Q', plot = False, task=None, rep=None):
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Calculate the R-squared or Q-squared value
    residuals = observed - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    r_squared = 1 - (ss_res / ss_tot)

    plt.figure()
    if plot:
        if mode == 'Q':
            # Create a scatter plot of observed vs. predicted values
            plt.scatter(predicted, observed, label=f'Q² = {r_squared:.2f}', s=10, c='r')
        else:
            # Create a scatter plot of observed vs. predicted values
            plt.scatter(predicted, observed, label=f'R² = {r_squared:.2f}', s=10, c='b')

        # Add a 45-degree line to visualize perfect predictions
        plt.plot([min(observed), max(observed)], [min(observed), max(observed)], linestyle='--', color='gray')

        # Labels and title
        plt.xlabel('Predicted Values')
        plt.ylabel('Observed Values')
        if mode == 'Q':
            plt.title('Q² Plot')
        else:
            plt.title('R² Plot')

        # Show legend
        plt.legend()
        # Show the plot

        if task:
            task.logger.report_matplotlib_figure(title="Q squared", series="Plots", 
                                                 iteration=rep, figure=plt, report_image=True)
        else:
            plt.show()

    return r_squared


def plot_barplot(data, title, x_labels=None):
    #print(data.shape)
    n = data.shape[0]  # Number of bars
    x = np.arange(n)  # X-axis values

    plt.bar(x, data, align='center')
    #plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title(title)
    # Set custom x-axis labels if provided
    if x_labels:
        plt.xticks(x, x_labels, rotation=45, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def losses_distribution(Vpreds, S, Pin, Vins, labels):

    SV_for_flux = []
    relu_Vin_for_flux = []
    relu_V_for_flux = []


    for Vpred, Vin in zip(Vpreds, Vins):
        #2nd element of the loss (steady-state costraint)
        S = np.float32(S)
        SV = np.matmul(Vpred, S.T)
        SV_for_flux.append(SV)

        #3rd element of the loss(upper bounds)
        Pin = torch.from_numpy(np.float32(Pin))
        Vin = torch.from_numpy(np.float32(Vin))
        Vin_pred = torch.matmul(torch.from_numpy(np.float32(Vpred)), Pin.T)
        
        relu_Vin= torch.relu(Vin_pred - Vin).numpy()
        relu_Vin_for_flux.append(relu_Vin)

        #4th element of the loss(positive fluxes)
        relu_V = torch.relu(torch.from_numpy(np.float32(-Vpred)))
        relu_V_for_flux.append(relu_V.numpy())
    
    avg_SV_for_flux = np.mean(np.array(SV_for_flux), axis=0)
    avg_relu_Vin_for_flux = np.mean(np.array(relu_Vin_for_flux), axis=0)
    #print(np.array(relu_Vin_for_flux).shape)
    #print(avg_relu_Vin_for_flux.shape)
    avg_relu_V_for_flux = np.mean(np.array(relu_V_for_flux), axis=0)

    plot_barplot(avg_SV_for_flux, 'SV violation for each metabolite', labels[0])
    plot_barplot(avg_relu_Vin_for_flux, 'upper bound violation for bounded flux', labels[1])
    plot_barplot(avg_relu_V_for_flux, 'positivity violation for each flux', labels[2])


def r2_metric(true,pred):
    m, n = true.shape
    r_squared_scores = np.zeros((m,))

    for i in range(m):
        y_true = true[i, :]
        y_pred = pred[i, :]
        try:
            slope, intercept, r, p, se = linregress(y_true, y_pred)
        except:
            r_squared_scores[i] = 0
            continue
        r_squared_scores[i] = r**2

    return r_squared_scores


def metrics_table(df_true, df_pred):

    def relu(x):
        return max(0, x)
    df_pred = df_pred.applymap(relu)

    mae_per_row = np.mean((np.abs(df_true - df_pred)), axis=1)
    # Calculate the average of MSEs across all rows
    average_mae = np.mean(mae_per_row)
    # Calculate the standard deviation of MSE values
    mae_std = np.std(mae_per_row)

    rmse_per_row = np.sqrt(np.mean(np.square(df_true-df_pred), axis=1))
    # Calculate the average of MSEs across all rows
    average_rmse = np.mean(rmse_per_row)
    # Calculate the standard deviation of MSE values
    rmse_std = np.std(rmse_per_row)

    NE_per_row = np.nan_to_num(np.linalg.norm((df_true - df_pred), axis=1) / np.linalg.norm(df_true, axis=1), posinf=0, neginf=0, nan=0)
    # Calculate the average of MSEs across all rows
    average_NE = np.mean(NE_per_row)
    # Calculate the standard deviation of MSE values
    NE_std = np.std(NE_per_row)

    q2_per_row = r2_metric(df_true.values, df_pred.values)
    # Calculate the average of MSEs across all rows
    average_q2 = np.mean(q2_per_row)
    # Calculate the standard deviation of MSE values
    q2_std = np.std(q2_per_row)


    df_metrics= pd.DataFrame({'avg': [average_q2, average_mae, average_rmse, average_NE], 
                              'std': [q2_std, mae_std, rmse_std, NE_std ]}, 
                              index= ['Q2', 'MAE', 'RMSE', 'NE'])

    return df_metrics


    
def histogram_rmse_fluxes(df_true, df_pred):
    # Calculate RMSE between columns
    rmse_values = {}
    for col in df_true.columns:
        rmse_values[col] = np.sqrt(((df_true[col] - df_pred[col]) ** 2).mean())
        

    # Create bar plot
    plt.figure(figsize=(16, 7))
    plt.bar(rmse_values.keys(), rmse_values.values())
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees
    plt.ylabel('RMSE')
    plt.ylim((0, 1.8))
    plt.title('RMSE for each flux', fontweight='bold', fontsize=20)

    plt.tight_layout()
    plt.grid(False)

    return plt


if __name__ == "__main__":
    pass