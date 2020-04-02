import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import linearcorex as lc

from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu

from rdkit.Chem import Draw, MolFromSmiles
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


class WrappedVAE(VAEUtils):
    def __init__(self, directory):
        super().__init__(directory=directory)
        self.hidden = []

    def __encode(self, smiles, smiles_col='smiles'):
        if isinstance(smiles, pd.DataFrame):
            smiles = smiles[smiles_col].values

        X = self.smiles_to_hot(smiles)
        self.hidden = self.encode(X)

    def decode_to_smiles(self, smiles, noise=5, decode_attempts=100, concat=True):
        molecules = []
        self.__encode(smiles)

        for i, hidden_i in enumerate(self.hidden):
            df = self.z_to_smiles(hidden_i, noise_norm=noise, decode_attempts=decode_attempts)
            df['parent'] = smiles[i]
            molecules.append(df)

        if concat:
            return pd.concat(molecules).reset_index(drop=True)

        return molecules

    def predict(self, smiles, smiles_col='smiles'):
        if isinstance(smiles, pd.DataFrame):
            smiles = smiles[smiles_col].values

        self.__encode(smiles)

        y_p = self.predict_prop_Z(self.hidden)
        y_p = pd.DataFrame(columns=['qed', 'SAS', 'logP'], data=y_p, index=smiles) \
            .reset_index() \
            .rename(columns={'index': 'smiles'})

        y_p['best'] = 5 * y_p['qed'] - y_p['SAS']
        return y_p

    def decompose_2d(self, smiles, method='PCA'):
        self.__encode(smiles)

        if method == 'PCA':
            y = self.__perform_pca()
        elif method == 'Corex':
            y = self.__perform_corex()
        else:
            return 'Unknown method! Use PCA or Corex'

        return y

    def __perform_pca(self):
        y = PCA(n_components=2).fit_transform(self.hidden)
        y = MinMaxScaler().fit_transform(y).transpose()

        return y

    def __perform_corex(self):
        y = lc.Corex(n_hidden=2).fit_transform(self.hidden)
        y = MinMaxScaler().fit_transform(y).transpose()

        return y

    @staticmethod
    def draw(smiles, parents=None, subImgSize=(400, 200), molsPerRow=3, smiles_col='smiles'):
        if isinstance(smiles, pd.DataFrame):
            smiles = smiles[smiles_col].values

        if isinstance(smiles, np.ndarray):
            smiles = smiles.tolist()

        if parents is not None:
            legends = [f'P: {parents[i]} D: {name}' for i, name in enumerate(smiles)]
        else:
            legends = smiles

        ms_rd = [MolFromSmiles(mol) for mol in smiles]
        img = Draw.MolsToGridImage(ms_rd, subImgSize=subImgSize, legends=legends, molsPerRow=molsPerRow)

        return img

    @staticmethod
    def plot_decomposed(y, c='blue'):
        y_x = y[0]
        y_y = y[1]

        g = plt.scatter(y_x, y_y, c=c)
        cbar = plt.colorbar(g)

        plt.show()
