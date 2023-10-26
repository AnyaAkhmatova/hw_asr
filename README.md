# ASR project 

Устанавливать библиотеки нужно с помощью requirements.txt. Dockerfile невалидный.

Guide по установке:
```
git clone https://github.com/AnyaAkhmatova/hw_asr.git
sudo apt-get install rubberband-cli
pip install -r ./hw_asr/requirements.txt
pip install ./hw_asr/.
```

Запуск train из hw_asr:
```
python3 ./train.py -c <path to your config>
```

Запуск test из hw_asr:
```
python3 ./test.py --resume <path to your checkpoint> -c <path to your test config>
```
____

Скачать финальную модель в папку ./final_model:
```
pip install gdown
gdown https://drive.google.com/uc?id=14p2h3wzOQfOC-una8g1ITvVOIhd-yX5L
unzip final_model.zip
rm -rf final_model.zip
```

Запуск test.py для модели:
```
!python3 ./test.py --resume ./final_model/final_all/model_best.pth -c ./final_model/final_all/config_test.json
```

Выдача test.py для модели находится в папке results (output-ы и ноутбук с запуском).
Результаты:

test-clean: wer (argmax) = 0.3384724309781591, wer (ctc_beam_search) = 0.32759457361201527, cer (argmax) = 0.11087391707763049, cer (ctc_beam_search) = 0.10782303549931747

test-other: wer (argmax) = 0.5609493959133318, wer (ctc_beam_search) = 0.5548816673067045, cer (argmax) = 0.23319069026993902, cer (ctc_beam_search) = 0.23160599191116957

Модель - DeepSpeech2, с 3 свертками, 5 GRU (hidden_size=800). Обучалась на всех train выборках Librispeech, с аугментациями, использовались оптимизатор Adam с lr=1e-3, расписание шага MultiStepLR с gamma=0.5, подробнее в ./hw_asr/configs/train_config_librispeech_all.json. Получилось обучить 4 эпохи, каждая занимала от 3.5 до 4.5 часов. Обучалась в два подхода, логи лежат в ./final_model/checkpoint_all/info.log и ./final_model/final_all/info.log, время и прогресс обучения можно посмотреть в ноутбуке ./final_model/final_all/notebookd9387f2c59.ipynb (final_model нужно скачать, инструкции см. выше).

____

W&B Report: ... .






