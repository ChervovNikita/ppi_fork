## Подготовка эксперимента

Для запуска эксперимента необоходимо поменять модель в файле `train.py`, варианты моделей, с которыми проводились эксперименты:

### model = GCNN_with_descriptors(num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80, transformer_dim=31, nhead=4, num_layers=2, dim_feedforward=128)
Score: 86.40548

### model = GCNN_with_descriptors(num_features_pro=1024, output_dim=128, dropout=0.2, descriptor_dim=80, transformer_dim=31, nhead=4, num_layers=1, dim_feedforward=128)
Score: 87.20135774970679

### model = GCNN_with_descriptors(num_features_pro=1024, output_dim=128, dropout=0.1, descriptor_dim=80, transformer_dim=31, nhead=4, num_layers=1, dim_feedforward=128)
Score: 86.98706810190387

### model = GCNN_with_descriptors(num_features_pro=1024, output_dim=128, dropout=0.3, descriptor_dim=80, transformer_dim=31, nhead=4, num_layers=1, dim_feedforward=128)
Score: 87.14017230813934

### model = GCNN_mutual_attention(num_layers=1)
Score: 87.38506639358573

### model = GCNN_mutual_attention(num_layers=1, dropout=0.1)
Score: 87.00234688700611

### model = GCNN_mutual_attention(num_layers=2)
Score: 86.71149927765559

### model = GCNN_mutual_attention(num_layers=3)
Score: 87.35442680536303

### model = GCNN_mutual_attention(num_layers=1, dropout=0.3)
Score: 87.27789227753492

### model = GCNN()
Score: 87.1554510932416

### model = GCNN_desc_pool()
Score: 86.84921

## Запуск эксперимента

сначала необходимо подготовить данные для обучения. А именно, в папке на уровне `ppi_fork` должна быть папка `masif_features`, в ней должна быть папка `processed` в нее необходимо положить файлы с признаками для всех белков `XXXX_L.pt`, `XXXX_R.pt`, а также папка со всеми дескрипторами (`masif_features/processed/masif_descriptors`) из `MaSIF` (внутри файлы `desc_flipped.npy `, `desc_straight.npy`).
И также нужно создать файл с тем какие белки взаимодействуют и какие нет - `.npy` файл (пример строки `array(['', '', '8fbe_L', '', '', '8fbe_R', '1'], dtype='<U12')`)

Чтобы получить признаки для белков, нужно положить их `.pdb` файлы в папку `masif_features/raw` и запустить скрипт
```
python proteins_to_graphs.py
```

Для запуска эксперимента необходимо выполнить команду:
```
bash run_cv.sh <model_name> <path_to_dataset> <cv_fold_count>
```

Будет запущено обучение с 5 различными фолдами для cross validation.

Результаты обучений будут в папке `log_dir/<model_name>`