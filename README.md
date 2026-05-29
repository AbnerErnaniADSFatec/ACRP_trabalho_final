# Sample Extractor: um plugin QGIS para extrair amostras de uso e cobertura da terra a partir de imagens de satélite

A metodologia comumente usada para a geração de mapas de uso e cobertura da terra envolve a análise de séries temporais de imagens de satélite aplicadas em algoritmos de aprendizado de máquina. Contudo, para extrair estas séries temporais, são necessárias coordenadas ou amostras coletadas a partir de um processo multidisciplinar de interpretação de imagens. Este processo exige uma equipe de especialistas em biologia, física, química e geografia, além disso são necessários sistemas de informação geográfica para a visualização e processamentos das imagens. O processo muitas vezes exige um esforço manual, por essa razão existem metodologias para a seleção automática de amostras. Este trabalho apresenta uma extensão de \textit{software} no QGIS para a extração de amostras usando imagens de alta resolução aplicadas no agrupamento SOM somado à técnicas de estimativas para vetorizar a imagens e selecionar amostras automaticamente.

Ambiente de testes.

```
conda create --name acrp python=3.11

pip install tqdm rioxarray requests aiohttp

pip install fsspec s3fs aiohttp zarr numpy==1.26.4 scipy==1.15.3 pandas==2.2.3 pyarrow==17.0.0 scikit-learn==1.6.1

python3 -m pip install -r requirements.txt

ipython kernel install --user --name acrp
```
