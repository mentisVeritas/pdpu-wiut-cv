# PDPU — кто что делает

Хакатон WIUT 2026, CV track. Дедлайн: **27 сентября 2026, 23:59 (Ташкент)**.

| Кто | GitHub | Роль до дедлайна |
|---|---|---|
| Begzad Kenesbaev | [mentisVeritas](https://github.com/mentisVeritas) | Пайплайн: YOLO, трекер, правила, Part B, репозиторий |
| Fariza Raxmanova | [farizarakhmanova](https://github.com/farizarakhmanova) | Разметка sample-видео + EDA (что видно на камере) |
| mallokodev | [httpswap](https://github.com/httpswap) | Сайт команды и живое демо |

## Fariza — разметка (можно начинать, как только скачается хоть одно видео)

```bash
python scripts/annotate.py sample_001.mp4
```

Клавиши на экране. Файл сохранится как `my_labels.json` в корне (формат `evaluate.py`).

Что писать в заметках для сайта (EDA):

- разрешение, fps, длина, день/ночь
- где полосы, стоп-линия, светофор, переход
- куда едет основной поток
- какие из 14 классов реально видны

Потом полигоны кладём в `src/scene.json` (координаты 0–1).

## mallokodev — сайт

Каркас в `website/`. Обязательные блоки: команда, подход, EDA, результаты на samples, живое демо, отчёт, ссылки. Демо = загрузка mp4 → таймлайн событий. Хост: Vercel / HF Spaces / свой сервер.

## Общий репозиторий

https://github.com/mentisVeritas/pdpu-wiut-cv

Видео в git не кладём. Каждый у себя: `bash samples/download.sh`.
