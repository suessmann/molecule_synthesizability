![](https://github.com/suessmann/jb_test_synthesizability/blob/master/img/given_mols.png?raw=true)
# Генерация и предсказание свойств молекул лекарственных препаратов с помощью вариационного автонкодера
---
В этом репозитории находится мой эксперимент по поиску лекарственных молекул. При помощи VAE я декодирую новые молекулы из существующих и предсказываю их химические свойства.

# Contents
---
```molecule_synthesizability.ipynb``` - тетрадка с экспериментами  
```chemvae_wrapper.py``` - обертка над модулем [chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae) для более удобного взаимодействия

# References
---
Gómez-Bombarelli, R., Wei, J. N., Duvenaud, D., Hernández-Lobato, J. M., Sánchez-Lengeling, B., Sheberla, D., ... & Aspuru-Guzik, A. (2018). Automatic chemical design using a data-driven continuous representation of molecules. ACS central science, 4(2), 268-276.

