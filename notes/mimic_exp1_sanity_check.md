# mimic-video, Exp1: проверка абляций (пункт 2 научницы) — подробное объяснение

*Сохранено 2026-10-29, чтобы не потерялось в чате. Все ссылки ведут на код в
репозитории mimic-video на GitHub — в ту версию, которая реально работала.*

## 0. Два вопроса

**Что научница имеет в виду под «masks».** Фразу можно понять двояко:
1. Как пункт проверочного списка для каждой абляции: «проверьте формы, маски,
   внимание, значения активаций и что ветка действий работает». Тогда «маски» =
   убедиться, что вмешательство не открыло и не закрыло то, на что модель смотрит.
   Мы читали так, потому что все слова стоят в одном ряду после «check».
2. Как новое вмешательство: закрывать маской часть прогноза (например, половину
   токенов) и смотреть, как падает успех — частичная абляция, близка к её пункту 1
   (градуированная подмена).

Лучше уточнить у неё. Сделать можно оба варианта.

**Маски проверены тестом, не только чтением кода** — см. раздел 6.

## 1. Термины

- **Тензор** — многомерный массив чисел. Форма `[1, N, D]` = 1 пример, N строк, по D чисел.
- **Токен** — одна строка такого массива (вектор из D чисел); модель обрабатывает вход как последовательность токенов.
- **Активации** — промежуточные числа внутри сети (выходы слоёв).
- **Слой, блок** — одна ступень обработки; блок трансформера состоит из нескольких слоёв.
- **LayerNorm** — для каждого токена вычитает его среднее, делит на его стандартное отклонение, затем умножает на выученный вектор весов γ и прибавляет выученный вектор сдвига β. Для нулевого входа первая часть даёт 0, и на выходе ровно β.
- **Внимание (attention)** — механизм, которым токены собирают информацию из других токенов. Из «спрашивающего» токена получается вектор запроса q, из каждого токена-источника — ключ k и значение v. Сходство q·k для всех источников превращается функцией softmax в веса (неотрицательные, сумма 1); результат — взвешенная сумма значений v.
- **Cross-attention** — запросы из одной последовательности (токены действий), ключи и значения из другой (токены прогноза).
- **Маска внимания** — правило, запрещающее некоторым запросам смотреть на некоторые ключи: перед softmax туда пишется −∞, вес становится ровно 0. Если запретить всё, softmax даёт 0/0 = NaN.
- **NaN** — «не число», признак сломанного вычисления.
- **Энтропия внимания** — насколько веса размазаны; мы делим её на максимум, поэтому 1.0 = все веса одинаковые.
- **Косинус между токенами** — похожесть направлений векторов: 1.0 = одинаковые, около 0 = несвязанные.
- **Пакет действий** — 15 будущих шагов × 10 чисел: 3 перемещение, 6 поворот, 1 захват.
- **6D-поворот** — поворот, записанный двумя столбцами матрицы поворота по 3 числа; у настоящего поворота длина каждого столбца ровно 1.
- **Сид** — число, задающее генератор случайных чисел; одинаковый сид → одинаковый «случайный» шум.
- **Замкнутый цикл** — робот реально выполняет действия в симуляторе, следующий кадр зависит от его действий.
- **«В распределении»** — входы, похожие на те, что модель видела при обучении. Вне распределения поведение модели может быть произвольным.

## 2. Как mimic-video выбирает действие

1. Кадры приводятся к 480×640 и масштабу [−1, 1] в [`_process_image`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py#L113) и складываются в очередь из 5 кадров в [`_add_image_to_history`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py#L122); перед вызовом модели очередь склеивается в видео ([строка 174](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py#L174)).
2. Видеомодель Cosmos-Predict2 делает один шаг расшумления (`stop=1`); из её слоя [`xattn_layer_idx`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/pipelines/video2world2action.py#L60) берутся активации и раскладываются в `crossattn_emb` формы `[1, N, D]` ([строка 63](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/pipelines/video2world2action.py#L63)). Это и есть «прогноз».
3. Он передаётся в декодер действий ([строка 65](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/pipelines/video2world2action.py#L65)) вместе с состоянием робота и числом `sigma`.
4. В декодере прогноз сначала проходит LayerNorm [`ctx_norm`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L796) ([применение](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L891)), затем в каждом блоке — модуль [`cross_attn`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L588) ([вызов](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L712)): q, k, v в [`compute_qkv`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L293), внимание в [`compute_attention`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L326) быстрым ядром с [`attn_mask_type="no_mask"`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L263).
5. Выход — пакет действий в [`action_buffer`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py#L200).

## 3. Что делал старый Exp1

- Нули: [`ZeroWorldModelPipeline`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/zero_world_model_pipeline.py#L78) подаёт тензор нулей той же формы.
- Shuffle: [`VAMShuffledInference._add_image_to_history`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/vam_shuffled_inference.py#L102) на каждом шаге кладёт в очередь кадр случайного чужого эпизода ([строка 106](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/vam_shuffled_inference.py#L106)).

## 4. Диагностика

[`SanityPipeline`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L76) встаёт на место модели. На каждом перепланировании:

1. Считает настоящий прогноз так же, как модель ([строка 155](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L155)).
2. Считает по нему действие, которое ведёт робота ([строка 157](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L157)) — робот едет как обычная политика, все варианты сравниваются в одних и тех же состояниях.
3. Строит 6 других вариантов: нули ([163](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L163)), шум с тем же средним и разбросом каждого токена ([164](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L164)), первый прогноз эпизода ([167](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L167)), среднее по прошлым эпизодам задачи ([171](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L171)) и два shuffle ([174–175](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L174)), для которых видеомодель реально прогоняется на подменных кадрах из [`_FrameSource`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/vam_sanity_inference.py#L63): [`perstep_clip`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/vam_sanity_inference.py#L95) — 5 кадров из 5 случайных эпизодов (как в Exp1), [`coherent_clip`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/vam_sanity_inference.py#L99) — 5 подряд кадров одного чужого эпизода.
4. Для каждого из 7 вариантов вызывает декодер с тем же состоянием и тем же сидом ([строка 189](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L189)) и записывает статистики и действия.

**Внимание.** На каждый `cross_attn` поставлен хук — функция, которую PyTorch вызывает перед модулем и передаёт ей его входы ([строка 39](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L39)). Хук вызывает собственный `compute_qkv` модуля, считает q·k/√d ([47](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L47)), softmax и энтропию, делённую на максимум ([50](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L50)). Пересчитывать пришлось потому, что быстрое ядро веса не отдаёт.

**Активации.** [`_ctx_stats`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L57) пропускает вариант через тот же `ctx_norm` ([60](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L60)) и пишет нормы токенов и средний косинус между токенами ([72](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/sanity_pipeline.py#L72)).

**Ветка действий.** [`analyze_sanity.py`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/analyze_sanity.py#L39): величина перемещения (длина первых 3 чисел), отклонение длины столбцов поворота от 1 ([42](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/analyze_sanity.py#L42)), число NaN.

## 5. Результаты по пунктам научницы

| что | как проверено | что вышло |
|---|---|---|
| формы | все варианты той же формы; `sigma` записана на каждом шаге | формы совпадают; `sigma` = 68.325 на всех 1416 шагах |
| маски | тест (раздел 6) | маски нет; NaN ни разу |
| внимание | энтропия по блокам | настоящий 0.74; нули ровно 1.00 |
| активации | после `ctx_norm` | при нулях все токены одинаковы (косинус 1.00): LayerNorm от нуля = β |
| ветка действий | перемещение, поворот, NaN | нули и шум: поворот невалиден (0.055 против 0.006), движения почти нет; остальные варианты в норме |
| диагностика ничего не портит | действие, ведущее робота, против «настоящего» варианта | совпадение до машинной точности; успех как у базового |

При обучении декодер лишали только состояния робота ([`obs_dropout=0.2`](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_model.py#L304), [механизм](https://github.com/Coldilger/mimic-video/blob/main/model/cosmos_predict2/models/world2action_dit.py#L864)), никогда — прогноза.

**Замкнутый цикл** для вариантов, которые декодер не ломают: [`MeanWorldModelPipeline`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/closed_loop/vam_condition_inference.py#L53) ([`load_mean_context`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/closed_loop/vam_condition_inference.py#L45)), [`VAMCoherentShuffledInference`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/closed_loop/vam_condition_inference.py#L80), [`StaleWorldModelPipeline`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/closed_loop/vam_condition_inference.py#L101); все через [`closed_loop.slurm`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/closed_loop/closed_loop.slurm) на тех же 24 эпизодах. Сравнение — [`paired_stats.py`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/paired_stats.py#L86): [`load_outcomes`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/paired_stats.py#L37) читает исходы из имён видео, [`wilson`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/paired_stats.py#L66) — интервал для доли, бутстреп парной разности с ресемплингом внутри задач ([95](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/paired_stats.py#L95)), [`mcnemar_exact`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/paired_stats.py#L76) — смотрит только на пары с разными исходами и проверяет, не случаен ли их перекос.

Итог: базовый 46/96; нули, per-step shuffle, связный shuffle, среднее, устаревший — все 0/96 (−47,9 п.п. [−56,2; −39,6], p < 0,0001). Нужный объект сдвигался только при устаревшем прогнозе (12/96).

## 6. Тест масок — результат

[`mask_attention_test.py`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/mask_attention_test.py): модель без симулятора, 5 настоящих кадров Bridge, все 24 блока декодера, 19 200 токенов прогноза.

| проверка | настоящий прогноз | нулевой |
|---|---|---|
| выход ядра против ручного внимания без маски ([`manual_attention`](https://github.com/Coldilger/mimic-video/blob/main/eval/bridge/experiment1_ablation/ablation_sanity/mask_attention_test.py#L54)), макс. отн. ошибка | 1,8e-3 (точность bf16) | 0 |
| подмена одного токена на позициях 0 / 4800 / 9600 / 14400 / 19199 меняет выход настоящего модуля | да, все позиции (минимум 1,5e-3) | да |
| веса внимания | от 5e-11 до 0,26 | ровно 1/19 200 у каждого токена |
| контроль: та же подмена на позиции, закрытой маской вручную | изменение 0 | 0 |

Вывод: маски нет, ядро — обычное softmax-внимание (значит, энтропия в диагностике посчитана верно), при нулях внимание ровно равномерное. Оговорка: повтора ядра без возмущения не делали, поэтому самое маленькое изменение (1,5e-3) напрямую не отделено от возможной недетерминированности ядра; контроль с маской даёт ровно 0, так что маску тест поймать способен.

## 7. Ограничения

1. Среднее в mean — по той же задаче и включает её эпизоды: несёт «какая задача», но не «какое состояние».
2. Кадры shuffle — настоящие фото Bridge, живые кадры — рендер симулятора. Для видеомодели фото привычны; устаревший прогноз (только свои кадры) тоже даёт 0%.
3. mimic-video детерминирован (`seed=0` зашит): выборка = 24 разных эпизода на задачу, 96 всего. Диагностика Carrot — 22 эпизода (два ушли на пробный прогон).
