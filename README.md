# UNet++ con mecanismos de atención

Este repositorio contiene una implementación de UNet++ usando mecanismos de atención para proyecto de tesis.

## Requerimentos

- PyTorch 1.x or 0.41

## Instalación

1. Crear un entorno de python (Se recomienda Conda)

```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```

2. Instalar PyTorch.

```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

3. Instalar paquetes necesarios.

```sh
pip install -r requirements.txt
```

## Datos de entrenamiento

1. Descarga el dataset desde la página de LIDC-IDRI y agreguelos de la siguiente manera

```
.
└── input/
    └── LIDC-IDRI/
        ├── stage1_train/
        │   └── images/
        │       ├── LIDC_IDRI-0001/
        │       │   ├── 1-001.dcm
        │       │   ├── 1-002.dcm
        │       │   └── ...
        │       └── ...
        └── stage2_test/
            ├── LIDC-IDRI-0001/
            │   ├── 1-001.dcm
            │   ├── 1-002.dcm
            │   └── ...
            └── ...
```

2. Preprocesamiento.

```sh
python preprocess_LIDC-IDRI.pyy
```

3. Entrenamiento

El archivo de entrenamiento se encuentra en train.py

4. Evaluate.

```sh
python test/original.py
python test/groundtruth.py
python test/nested.py
python test/evaluate.py
```
