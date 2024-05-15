# cluwords_rep
 Implementation of the CluWords Module Representation

## Instalation
Follow the instructions to intall Poetry - https://python-poetry.org/docs/

To install dependencies, use the following commands:
```
poetry shell
poetry install
```

## Usage

The folder `flows` contains a few samples of CluWords instantiations. For instance, the flow -- `topic_modeling_config.yaml` is the CluWords instantiation for Topic Modeling. Thus, to use this instantiation follow the instructions:

* Copy the content of `topic_modeling_config.yaml` and paste in the file `flow_config.yaml` in the root path.
* As you may see in the YAML file, the instantiation may require a few data inputs. First the dataset, I left a sample data set `dropPre.csv` in the `data` folder, so please check it out. Second, it require a embedding representation file, you can download the fasttext embedding represetation through the [link](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip). Finally, it also may require a vocabulary datafile. I left a notebook `get_vocabulary.ipynb` in the notebooks folder so you can generate your vocabulary file.
* After setting up the dependency files, you may run the script `run_cluwords_pipeline.py`.

## References
You may find more detail about our work here:

```
title={CluWords: Exploiting SemanticWord Clustering Representation for Enhanced Topic Modeling},
author={Viegas, Felipe and Canuto, Sérgio and Gomes, Christian and Luiz, Washington and Rosa, 
Thierson and Ribas, Sabir and Rocha, Leonardo and Gonçalves, Marcos André},
booktitle={The Twelfth ACM International Conference on Web Search and Data Mining (WSDM ’19)},
year={2019},
organization={ACM}
```

```
title={CluHTM - Semantic Hierarchical Topic Modeling based on CluWords},
author={Viegas, Felipe and Cunha, Washington and Gomes, Christian and  Pereira Antonio and Rocha, Leonardo and Gonçalves, Marcos André},
booktitle={The 58th Annual Meeting of the Association for Computational Linguistics (ACL ’20)},
year={2020},
organization={ACL}
```
