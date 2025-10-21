import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from loki._utils import lokiError


def kernel_density_estimate(datasets, cols=None, n_cols=3, figsize=(16, 10)):
    """
    Plot kernel density estimation plots comparing numeric feature distributions 
    between real and synthetic datasets.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real/original dataset.
    synthetic_data : pd.DataFrame
        Synthetic dataset.
    cols : list, optional
        List of column names to plot. If None, all numeric columns are used.
    n_cols : int, default=3
        Number of plots per row.
    figsize : tuple, default=(16,10)
        Overall figure size.
    """
    if type(datasets) != dict:
        raise lokiError("datasets must be a dictionary", '0002')
    
    datasets_list = [dataset.assign(Source=dataset_name) 
                     for dataset_name, dataset in datasets.items()] 

    combined = pd.concat(datasets_list)

    if cols is None:
        cols = datasets_list[0].select_dtypes(include=['float64', 'int64']).columns

    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.kdeplot(data=combined, 
                    x=col,  
                    ax=axes[i],
                    hue='Source', 
                    fill=True, 
                    common_norm=False, 
                    alpha=0.5)
        
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Kernel Density Estimate: Real vs Synthetic Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return


def categorical_proportions(datasets, cols=None, n_cols=3, figsize=(16, 10)):
    """
    Plot side-by-side count plots comparing category distributions
    between real and synthetic datasets.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real/original dataset.
    synthetic_data : pd.DataFrame
        Synthetic dataset.
    cols : list, optional
        Categorical columns to plot. If None, automatically detect them.
    n_cols : int, default=3
        Number of subplots per row.
    figsize : tuple, default=(16,10)
        Overall figure size.
    """


    if type(datasets) != dict:
        raise lokiError("datasets must be a dictionary", '0002')
    
    datasets_list = [dataset.assign(Source=dataset_name) 
                     for dataset_name, dataset in datasets.items()] 

    combined = pd.concat(datasets_list)

    if cols is None:
        cols = datasets_list[0].select_dtypes(include=['object', 'category']).columns
    
    cols = cols.delete(-1)

    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        combined[col] = combined[col].fillna('Missing')

        sns.countplot(data=combined, 
                      y=col, 
                      ax=axes[i],
                      hue='Source',
                      alpha=0.5)
        
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis='x', rotation=45)


        
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])


    plt.suptitle("Categorical Proportions: Real vs Synthetic Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def correlation_heatmaps(real_data, synthetic_data):
    """
    Plot side-by-side correlation heatmaps for real and synthetic datasets, 
    along with a heatmap showing their correlation differences.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real/original dataset.
    synthetic_data : pd.DataFrame
        Synthetic dataset.
    """
    num_cols = real_data.select_dtypes(include=np.number).columns

    real_num = real_data[num_cols]
    syn_num = synthetic_data[num_cols]

    real_corr = real_num.corr()
    syn_corr = syn_num.corr()
    
    
    diff = real_data[num_cols].corr() - synthetic_data[num_cols].corr()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.heatmap(real_corr, ax=axes[0], vmin=-1, vmax=1, cmap='coolwarm', square=True)
    axes[0].set_title("Real Data Correlation", fontsize=14)

    sns.heatmap(syn_corr, ax=axes[1], vmin=-1, vmax=1, cmap='coolwarm', square=True)
    axes[1].set_title("Synthetic Data Correlation", fontsize=14)

    sns.heatmap(diff, ax=axes[2], vmin=-1, vmax=1, cmap='coolwarm', square=True)
    axes[2].set_title("Correlation Difference (Real − Synthetic)", fontsize=14)

    plt.tight_layout()
    plt.show()

def violin(datasets, cols=None, n_cols=3, figsize=(16, 10)):
    """
    Plot violin plots comparing numeric feature distributions 
    between real and synthetic datasets.

    Parameters
    ----------
    real_data : pd.DataFrame
        Real/original dataset.
    synthetic_data : pd.DataFrame
        Synthetic dataset.
    cols : list, optional
        List of column names to plot. If None, all numeric columns are used.
    n_cols : int, default=3
        Number of plots per row.
    figsize : tuple, default=(16,10)
        Overall figure size.
    """
        
    if type(datasets) != dict:
        raise lokiError("datasets must be a dictionary", '0002')
    
    datasets_list = [dataset.assign(Source=dataset_name) 
                     for dataset_name, dataset in datasets.items()] 

    combined = pd.concat(datasets_list)


    if cols is None:
        cols = datasets_list[0].select_dtypes(include=['float64', 'int64']).columns

    n_rows = int(np.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    

    for i, col in enumerate(cols):
        sns.violinplot(data=combined,
                        x="income",
                        y=col,
                        ax=axes[i],
                        inner="quart",
                        hue='Source',
                        dodge=False,
                        fill=True,
                        alpha=0.5
            )

        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])


    plt.suptitle("Violin Plots: Real vs Synthetic Distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()






